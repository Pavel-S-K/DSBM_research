import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from functools import partial
import copy
import ot as pot

from typing import List, Optional, Tuple
import hydra
from hydra import initialize, compose
import pytorch_lightning as pl
from omegaconf import DictConfig

import time
import os
import shutil
import datetime

device = 'cpu'


class MLP(nn.Module):
  def __init__(self, input_dim, layer_widths=[100,100,2], activate_final = False, activation_fn=F.tanh):
    super(MLP, self).__init__()
    layers = []
    prev_width = input_dim
    for layer_width in layer_widths:
      layers.append(torch.nn.Linear(prev_width, layer_width))
      prev_width = layer_width
    self.input_dim = input_dim
    self.layer_widths = layer_widths
    self.layers = nn.ModuleList(layers)
    self.activate_final = activate_final
    self.activation_fn = activation_fn
        
  def forward(self, x):
    for i, layer in enumerate(self.layers[:-1]):
      x = self.activation_fn(layer(x))
    x = self.layers[-1](x)
    if self.activate_final:
      x = self.activation_fn(x)
    return x


class ScoreNetwork(nn.Module):
  def __init__(self, input_dim, layer_widths=[100,100,2], activate_final = False, activation_fn=F.tanh):
    super().__init__()
    self.net = MLP(input_dim, layer_widths=layer_widths, activate_final=activate_final, activation_fn=activation_fn)
    
  def forward(self, x_input, t):
    inputs = torch.cat([x_input, t], dim=1)
    return self.net(inputs)


# DSBM модифицированный
class DSBM_new(nn.Module):
  def __init__(self, net_fwd=None, net_bwd=None, num_steps=1000, sig=0, eps=1e-3, first_coupling="ref"):
    super().__init__()

    # Передается две сети
    self.net_fwd = net_fwd 
    self.net_bwd = net_bwd
    self.net_dict = {"f": self.net_fwd, "b": self.net_bwd} # Определяем их порядок

    # Определяем параметры
    self.N = num_steps
    self.sig = sig
    self.eps = eps
    self.first_coupling = first_coupling
  
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
      zend = prev_model.sample_sde(zstart=zstart, fb=prev_model.fb)[-1]
      if prev_model.fb == 'f':
        z0, z1 = zstart, zend
      else:
        z0, z1 = zend, zstart
    return z0, z1

  @torch.no_grad()
  def sample_sde(self, zstart=None, N=None, fb='', first_it=False, sigm0 = 0):
    assert fb in ['f', 'b', 'fc', 'fn', 'bn']
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
          traj.append(z.detach().clone())

      
    elif(fb == 'f'):
      for i in range(N):
          t = torch.ones((batchsize,1), device=device) * ts[i]
          pred = self.net_dict[fb](z, t)
          z = z.detach().clone() + pred * dt
          traj.append(z.detach().clone())

      
    # elif(fb == 'fc'):
    #     t = torch.linspace(0, 1, 2, device=device)
    #     z = zstart.detach().clone()
        
    #     def ode_func(t, z):
    #         t_tensor = t*torch.ones((batchsize,1), device=device)
    #         return self.net_dict['f'](z, t_tensor)
    #     res = odeint(ode_func, z, t, method='dopri5',)[-1]
    #     traj.append(res.detach().clone())
      
            
    elif(fb == 'fn'):
      for i in range(N):
          t = torch.ones((batchsize,1), device=device) * ts[i]
          pred = self.net_dict['f'](z, t)
          z = z.detach().clone() + pred * dt
          z = z + sigm0 * torch.randn_like(z) * np.sqrt(dt)
          traj.append(z.detach().clone())

      
    elif fb == 'bn':
      ts = 1 - ts
      for i in range(N):
          t = torch.ones((batchsize,1), device=device) * ts[i]
          pred = self.net_dict['b'](z, t)
          z = z.detach().clone() + pred * dt
          z = z + sigm0 * torch.randn_like(z) * np.sqrt(dt)
          traj.append(z.detach().clone())
        
    return traj


def train_dsbm(dsbm_ipf, x_pairs, batch_size, inner_iters, prev_model=None, fb='', first_it=False, lr=1e-4):
  assert fb in ['f', 'b']
  dsbm_ipf.fb = fb
  optimizer = torch.optim.Adam(dsbm_ipf.net_dict[fb].parameters(), lr=lr)
  # optimizer = dsbm_ipf.optimizer_dict[fb]
  loss_curve = []
  
  dl = iter(DataLoader(TensorDataset(*dsbm_ipf.generate_new_dataset(x_pairs, prev_model=prev_model, fb=fb, first_it=first_it)), 
                       batch_size=batch_size, shuffle=True, pin_memory=False, drop_last=True))

  #for i in tqdm(range(inner_iters)):
  for i in (range(inner_iters)):
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



@torch.no_grad()
def draw_plot(traj, z0, z1, N=None):
  
  plt.figure(figsize=(4,4))
  plt.xlim(-1,1)
  plt.ylim(-1,1)
    
  plt.scatter(z0[:, 0].cpu().numpy(), z0[:, 1].cpu().numpy(), label=r'$\pi_0$', alpha=0.15)
  plt.scatter(z1[:, 0].cpu().numpy(), z1[:, 1].cpu().numpy(), label=r'$\pi_1$', alpha=0.15)
  plt.scatter(traj[-1][:, 0].cpu().numpy(), traj[-1][:, 1].cpu().numpy(), label='Generated', alpha=0.15)
  plt.legend()
  plt.title('Distribution')
  plt.tight_layout()



def train_new(cfg: DictConfig):
  # set seed for random number generators in pytorch, numpy and python.random
  if cfg.get("seed"):
    print(f"Seed: <{cfg.seed}>")
    pl.seed_everything(cfg.seed, workers=True)


  dataset_size = cfg.Gaussian_distribution.dataset_size
  test_dataset_size = cfg.Gaussian_distribution.test_dataset_size
  lr = cfg.lr
  batch_size = cfg.batch_size

  a_1 = cfg.Gaussian_distribution.mean_1
  a_2 = cfg.Gaussian_distribution.mean_2
  var_1 = cfg.Gaussian_distribution.var_1
  var_2 = cfg.Gaussian_distribution.var_2
  dim = cfg.Gaussian_distribution.dim
    
  initial_model = Normal(a_1 * torch.ones((dim, )), np.sqrt(var_1))
  target_model =  Normal(a_2 * torch.ones((dim, )), np.sqrt(var_2))
  
  x0 = initial_model.sample([dataset_size])
  x1 = target_model.sample([dataset_size])
  x_pairs = torch.stack([x0, x1], dim=1).to(device)
  
  x0_test = initial_model.sample([test_dataset_size])
  x1_test = target_model.sample([test_dataset_size])
  x0_test = x0_test.to(device)
  x1_test = x1_test.to(device)

  torch.save({'x0': x0, 'x1': x1, 'x0_test': x0_test, 'x1_test': x1_test}, "data.pt")

  x_test_dict = {'f': x0_test, 'b': x1_test}
  
  net_split = cfg.net_name.split("_")
  if net_split[0] == "mlp":
    if net_split[1] == "small":
      net_fn = partial(ScoreNetwork, input_dim=dim+1, layer_widths=[128, 128, dim], activation_fn=hydra.utils.get_class(cfg.activation_fn)())  # hydra.utils.get_method(cfg.activation_fn))  # 
    else:
      net_fn = partial(ScoreNetwork, input_dim=dim+1, layer_widths=[256, 256, dim], activation_fn=hydra.utils.get_class(cfg.activation_fn)())  # hydra.utils.get_method(cfg.activation_fn))  # 
  else:
    raise NotImplementedError
  
  num_steps = cfg.num_steps
  sigma = cfg.sigma
  inner_iters = cfg.inner_iters
  outer_iters = cfg.outer_iters

  if cfg.model_name == "dsbm":
    model = DSBM_new(net_fwd=net_fn().to(device), 
                  net_bwd=net_fn().to(device), 
                  num_steps=num_steps, sig=sigma, first_coupling=cfg.first_coupling)
    train_fn = train_dsbm
    print(f"Number of parameters: <{sum(p.numel() for p in model.net_fwd.parameters() if p.requires_grad)}>")
  else:
    raise ValueError("Wrong model_name!")


  # Training loop
  with tqdm(total=outer_iters, desc="Training Loop") as pbar:
      model_list = []
      time_list = []
      it = 1
  
      while it <= outer_iters:
          
        for fb in cfg.fb_sequence:
            
          start_time = time.time()
          first_it = (it == 1)
          if first_it:
            prev_model = None
          else:
            prev_model = model_list[-1]["model"].eval()
          model, loss_curve = train_fn(model, x_pairs, batch_size, inner_iters, prev_model=prev_model, fb=fb, first_it=first_it, lr=lr)
          end_time = time.time()
          full_time = end_time - start_time
          time_list.append(full_time)
          model_list.append({'fb': fb, 'model': copy.deepcopy(model).eval()})
    
          it += 1
          pbar.update(1)
            
          if it > outer_iters:
            break

  torch.save([{'fb': m['fb'], 'model': m['model'].state_dict()} for m in model_list], "model_list.pt")

  #return model_list, time_list


def test(cfg):

    # Загрузка тесовых данных
    data = torch.load('data.pt')
    x0_test = data['x0_test'].to(device)
    x1_test = data['x1_test'].to(device)

    # Загрузка параметров обученных моделей
    model_list = torch.load("model_list.pt")
    
    # Формируем модель заготовку
    dim = cfg.Gaussian_distribution.dim
    
    net_split = cfg.net_name.split("_")
    if net_split[0] == "mlp":
        if net_split[1] == "small":
          net_fn = partial(ScoreNetwork, input_dim=dim+1, layer_widths=[128, 128, dim], activation_fn=hydra.utils.get_class(cfg.activation_fn)())  # hydra.utils.get_method(cfg.activation_fn))  # 
        else:
          net_fn = partial(ScoreNetwork, input_dim=dim+1, layer_widths=[256, 256, dim], activation_fn=hydra.utils.get_class(cfg.activation_fn)())  # hydra.utils.get_method(cfg.activation_fn))  # 
    else:
        raise NotImplementedError
  
    num_steps = cfg.num_steps
    sigma = cfg.sigma
    inner_iters = cfg.inner_iters
    outer_iters = cfg.outer_iters

    if cfg.model_name == "dsbm":
        model = DSBM_new(net_fwd=net_fn().to(device), 
                      net_bwd=net_fn().to(device), 
                      num_steps=num_steps, sig=sigma, first_coupling=cfg.first_coupling)
    
    i = 0
    step=100
    save_directory = 'plots'
    
    
    if os.path.exists(save_directory):
        shutil.rmtree(save_directory)
    os.makedirs(save_directory, exist_ok=True)
    
    optimal_result_dict = {'mean': x0_test.mean(0).mean(0).item(), 'var': x0_test.var(0).mean(0).item(), 'cov': 0}
    result_list = {k: [] for k in optimal_result_dict.keys()}

    with tqdm(total=(len(model_list)), desc="Test loop") as pbar:
        while i < len(model_list):
            if (i%step == 0) or (i == len(model_list)-1):
                
                model.load_state_dict(model_list[i]['model'])
                traj = model.sample_sde(zstart=x1_test, fb='bn', N=1000, sigm0=cfg.sigma)
                
                draw_plot(traj, z0=x0_test, z1=x1_test)
                plt.savefig(os.path.join(save_directory, f"{i}-b.png"))
                plt.close()
        
                result_list['mean'].append(traj[-1].mean(0).mean(0).item())
                result_list['var'].append(traj[-1].var(0).mean(0).item())
                result_list['cov'].append(torch.cov(torch.cat([traj[0], traj[-1]], dim=1).T)[dim:, :dim].diag().mean(0).item())
                
            pbar.update(1)
            i+=1
    
    for j, k in enumerate(result_list.keys()):
      plt.plot(result_list[k], label=f"{cfg.model_name}-{cfg.net_name}")
      plt.plot(np.arange(len(result_list[k])), optimal_result_dict[k] * np.ones(len(result_list[k])), label="optimal", linestyle="--")
      plt.title(k.capitalize())
      if j == 0:
        plt.legend()
      plt.savefig(os.path.join(save_directory, f"convergence_{k}.png"))
      plt.close()


def main():

    # Загрузка параметров эксперимента
    with initialize(version_base=None, config_path="configurations"):
        cfg: DictConfig = compose(config_name="modDSBM_conf.yaml")
     
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    
    original_dir = os.getcwd() # Запоминаем корневую директорию
    
    # Создание директории текущего эксперимента
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S") # Формируем уникальную временную метку эксперимента
    
    experiments_dir = f'{cfg.paths.experiments_dir_name}_{timestamp}'
    
    if os.path.exists(experiments_dir):
        shutil.rmtree(experiments_dir)
    os.makedirs(experiments_dir, exist_ok=True)
    os.chdir(experiments_dir)
    
    # Train-test loops
    train_new(cfg)
    test(cfg)

    os.chdir(original_dir) # Возврат в корневую папку

if __name__ == "__main__":
    main()