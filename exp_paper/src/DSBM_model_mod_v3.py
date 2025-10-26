import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import deque
import time
import copy
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from src.wasserstein_distance import empirical_wasserstein_distance
import matplotlib.image as mpimg

device='cpu'

class Normalizer():
    def __init__(self):
        self.fitted = False
        self.A = 1
        self.B = 0
        
    def fit(self, data):
        self.global_min = torch.min(data)
        self.global_max = torch.max(data)
        self.global_range = self.global_max - self.global_min
        self.global_range = torch.clamp(self.global_range, min=1e-8)
        
        self.A = 2 / self.global_range
        self.B = -1 - (2 * self.global_min / self.global_range)
        
        self.fitted = True
        
    def normalize(self, data):
        if not self.fitted:
            raise ValueError("fitted == False")
        return self.A * data + self.B
        
    def denormalize(self, data):
        if not self.fitted:
            raise ValueError("fitted == False")
        return (data - self.B) / self.A

# DSBM
class DSBM(nn.Module):
  def __init__(self, net_fwd=None, net_bwd=None, num_steps=1000, sig=0, eps=1e-3, first_coupling="ref"):
    super().__init__()
    self.net_fwd = net_fwd
    self.net_bwd = net_bwd
    self.net_dict = {"f": self.net_fwd, "b": self.net_bwd}
    # self.optimizer_dict = {"f": torch.optim.Adam(self.net_fwd.parameters(), lr=lr), "b": torch.optim.Adam(self.net_bwd.parameters(), lr=lr)}
    self.N = num_steps
    self.sig = sig
    self.eps = eps
    self.first_coupling = first_coupling
    self.normalizer = Normalizer()
    self.normalizer_fitted = False
  
  @torch.no_grad()
  def get_train_tuple(self, x_pairs=None, fb='', **kwargs):
    z0, z1 = x_pairs[:, 0], x_pairs[:, 1]
    t = torch.rand((z1.shape[0], 1), device=device) * (1-2*self.eps) + self.eps
    z_t = t * z1 + (1.-t) * z0
    z = torch.randn_like(z_t)
    z_t = z_t + self.sig * torch.sqrt(t*(1.-t)) * z
    if fb == 'f':
      # z1 - z_t / (1-t)
      target = z1 - z0 
      target = target - self.sig * torch.sqrt(t/(1.-t)) * z
    else:
      # z0 - z_t / t
      target = - (z1 - z0)
      target = target - self.sig * torch.sqrt((1.-t)/t) * z
    return z_t, t, target

  @torch.no_grad()
  def generate_new_dataset(self, x_pairs, prev_model=None, fb='', first_it=False):
    assert fb in ['f', 'b']

    if prev_model is None:
      assert first_it
      assert fb == 'b'
      zstart = x_pairs[:, 0]
      if self.first_coupling == "ref":
        # First coupling is x_0, x_0 perturbed
        zend = zstart + torch.randn_like(zstart) * self.sig
      elif self.first_coupling == "ind":
        zend = x_pairs[:, 1].clone()
        zend = zend[torch.randperm(len(zend))]
      else:
        raise NotImplementedError
      z0, z1 = zstart, zend
    else:
      assert not first_it
      if prev_model.fb == 'f':
        zstart = x_pairs[:, 0]
      else:
        zstart = x_pairs[:, 1]
      zend = prev_model.sample_sde(zstart=zstart, fb=prev_model.fb, train_mod=True)[-1]
      if prev_model.fb == 'f':
        z0, z1 = zstart, zend
      else:
        z0, z1 = zend, zstart
    return z0, z1

  @torch.no_grad()
  def sample_sde(self, zstart=None, N=None, fb='', first_it=False, train_mod=False):
    assert fb in ['f', 'b']
      
    if (self.normalizer_fitted == True) and (train_mod==False):
        zstart = self.normalizer.normalize(zstart)

      
    ### NOTE: Use Euler method to sample from the learned flow
    if N is None:
      N = self.N   
    dt = 1./N
    traj = [] # to store the trajectory
    z = zstart.detach().clone()
    batchsize = z.shape[0]
    
    traj.append(z.detach().clone())
    ts = np.arange(N) / N
    if fb == 'b':
      ts = 1 - ts
    for i in range(N):
      t = torch.ones((batchsize,1), device=device) * ts[i]
      pred = self.net_dict[fb](z, t)
      z = z.detach().clone() + pred * dt
      z = z + self.sig * torch.randn_like(z) * np.sqrt(dt)
      traj.append(z.detach().clone())

    if (self.normalizer_fitted == True) and (train_mod==False):
        traj[0] = self.normalizer.denormalize(traj[0])
        traj[-1] = self.normalizer.denormalize(traj[-1])

    return traj



def train_dsbm(dsbm_ipf, x_pairs, batch_size, inner_iters, prev_model=None, fb='', first_it=False, lr=1e-4, normalize=False):
  assert fb in ['f', 'b']
  dsbm_ipf.fb = fb
  optimizer = torch.optim.Adam(dsbm_ipf.net_dict[fb].parameters(), lr=lr)
  loss_curve = []
      
  dl = iter(DataLoader(TensorDataset(*dsbm_ipf.generate_new_dataset(x_pairs, prev_model=prev_model, fb=fb, first_it=first_it)), 
                       batch_size=batch_size, shuffle=True, pin_memory=False, drop_last=True))

  for i in range(inner_iters):
    try:
      z0, z1 = next(dl)
    except StopIteration:
      dl = iter(DataLoader(TensorDataset(*dsbm_ipf.generate_new_dataset(x_pairs, prev_model=prev_model, fb=fb, first_it=first_it)), 
                           batch_size=batch_size, shuffle=True, pin_memory=False, drop_last=True))
      z0, z1 = next(dl)
    
    z_pairs = torch.stack([z0, z1], dim=1)
    z_t, t, target = dsbm_ipf.get_train_tuple(z_pairs, fb=fb, first_it=first_it)
    optimizer.zero_grad()
    pred = dsbm_ipf.net_dict[fb](z_t, t)
    loss = (target - pred).view(pred.shape[0], -1).abs().pow(2).sum(dim=1)
    loss = loss.mean()
    loss.backward()
    
    if torch.isnan(loss).any():
      raise ValueError("Loss is nan")
      break
    
    optimizer.step()
    loss_curve.append(np.log(loss.item())) ## to store the loss curve

  return dsbm_ipf, loss_curve


def draw_plot(traj, z0, z1):
  
  plt.figure(figsize=(4,4))
  plt.xlim(-20,20)
  plt.ylim(-20,20)
    
  plt.scatter(z0[:, 0].cpu().numpy(), z0[:, 1].cpu().numpy(), label=r'$\pi_0$', alpha=0.15)
  plt.scatter(z1[:, 0].cpu().numpy(), z1[:, 1].cpu().numpy(), label=r'$\pi_1$', alpha=0.15)
  plt.scatter(traj[-1][:, 0].cpu().numpy(), traj[-1][:, 1].cpu().numpy(), label='Generated', alpha=0.15)
  plt.legend()
  plt.title('Distribution')
  plt.tight_layout()



def train (cfg, model, x_pairs, x_pairs_test, logs, outer_iters, lr, RESULT_DIR):
    
    inner_iters = cfg.inner_iters
    batch_size = cfg.batch_size
    dim = cfg.dim
    model_list = deque(maxlen=4)
    x0_test = x_pairs_test[:, 0]
    x1_test = x_pairs_test[:, 1]
    it = 1
    
    # train loop
    with tqdm(total=outer_iters, desc="Training Loop iter") as pbar:
        while it <= outer_iters:
            for fb in cfg.fb_sequence:
              start_time = time.time()
        
              # train
              if len(model_list) == 0:
                prev_model = None
                first_it = True
              else:
                prev_model = model_list[-1]["model"].eval()
                first_it = False
                  
              model, loss_curve = train_dsbm(model, x_pairs, batch_size, inner_iters, prev_model=prev_model, fb=fb, first_it=first_it, lr=lr)
              end_time = time.time()
    
              logs['time_list'].append(end_time-start_time)
              model_list.append({'fb': fb, 'model': copy.deepcopy(model).eval()})
              
                
              # test - только для модели b -> f
              # оцениваем на каждой 10 итерации
              if (it%10 == 0) or (len(logs['time_list'])==2):

                  # Сохраняем последнюю версию модели
                  #torch.save(model_list['model'].state_dict(), RESULT_DIR + 'model_list.pt')
                  torch.save([{'fb': m['fb'], 'model': m['model'].state_dict()} for m in model_list], RESULT_DIR + "model_list.pt")
                  
                  i = len(logs['time_list'])
                  traj = model.eval().sample_sde(zstart=x1_test, fb='b', N=cfg.num_steps)
                  
                  draw_plot(traj, z0=x0_test, z1=x1_test)
                  plt.savefig(RESULT_DIR + f"iter_{i}-b.png")
                  plt.close()
                  
                  logs['time_list_res'].append(int(np.sum(logs['time_list'])))
                  logs['result_list']['mean'].append(traj[-1].mean(0).mean(0).item())
                  logs['result_list']['var'].append(traj[-1].var(0).mean(0).item())
                  logs['result_list']['cov'].append(torch.cov(torch.cat([traj[0], traj[-1]], dim=1).T)[dim:, :dim].diag().mean(0).item())
    
                  for j, k in enumerate(logs['result_list'].keys()):

                      # По эпохам
                      plt.plot(np.arange(len(logs['result_list'][k]))*10, logs['result_list'][k], label=f"{cfg.model_name}")
                      plt.plot(np.arange(len(logs['result_list'][k]))*10, logs['optimal_result_dict'][k] * np.ones(len(logs['result_list'][k])), label="optimal", linestyle="--")
                      plt.title(k.capitalize())
                      if j == 0:
                          plt.legend()
                      plt.savefig(RESULT_DIR +  f"convergence_{k}.png")
                      plt.close()

                      # На временной шкале
                      plt.plot(logs['time_list_res'], logs['result_list'][k], label=f"{cfg.model_name}")
                      plt.plot(logs['time_list_res'], logs['optimal_result_dict'][k] * np.ones(len(logs['result_list'][k])), label="optimal", linestyle="--")
                      plt.xlabel("Время обучения, сек.")
                      plt.title(k.capitalize())
                      if j == 0:
                          plt.legend()
                      plt.savefig(RESULT_DIR +  f"convergence_{k}_inTime.png")
                      plt.close()
    
              else:
                  pass
        
              it += 1
              pbar.update(1)
              if it > outer_iters:
                break

    # Расчет основных показателей обучения
    W = empirical_wasserstein_distance(traj[-1].numpy(), x1_test.numpy())
    W_opt = empirical_wasserstein_distance(x0_test.numpy(), x1_test.numpy())

    n_parameters = sum(p.numel() for p in model.net_fwd.parameters() if p.requires_grad)

    train_stat = pd.DataFrame()
    train_stat['Количество параметров модели'] = [int(n_parameters)]
    train_stat['Время обучения, сек'] = [int(np.sum(logs['time_list']))]
    train_stat['err_mean, %'] = [np.abs(100*(logs['result_list']['mean'][-1] - logs['optimal_result_dict']['mean'])/logs['optimal_result_dict']['mean'])]
    train_stat['err_var, %'] = [np.abs(100*(logs['result_list']['var'][-1] - logs['optimal_result_dict']['var'])/logs['optimal_result_dict']['var'])]
    train_stat['err_cov, %'] = [np.abs(100*(logs['result_list']['cov'][-1] - logs['optimal_result_dict']['cov'])/logs['optimal_result_dict']['cov'])]
    train_stat['err_WD, %'] = [np.abs(100*(W - W_opt)/W_opt)]
    train_stat = train_stat.T
    train_stat.columns = ['Показатели обучения']
    print(train_stat)

    train_stat.to_csv(RESULT_DIR + 'df_train_stat.csv')
    train_stat.to_pickle(RESULT_DIR + 'df_train_stat.pkl')

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    img1 = mpimg.imread(RESULT_DIR + 'convergence_cov_inTime.png')
    axes[0].imshow(img1)
    axes[0].axis('off')
    
    img2 = mpimg.imread(RESULT_DIR + 'convergence_mean_inTime.png')
    axes[1].imshow(img2)
    axes[1].axis('off')
    
    img3 = mpimg.imread(RESULT_DIR + 'convergence_var_inTime.png')
    axes[2].imshow(img3)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
                  
    return model, logs