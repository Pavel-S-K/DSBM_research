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

from typing import List, Optional, Tuple
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from hydra import initialize, compose

from torchdiffeq import odeint
import torch.optim as optim

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
  def sample_sde(self, zstart=None, N=None, fb='', first_it=False):
    assert fb in ['f', 'b']
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

    return traj



def get_bridge():

    with initialize(version_base=None, config_path="Bridges/configurations"):
        
    
        cfg: DictConfig = compose(config_name="temp.yaml")
        
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)
    
    data = torch.load('Bridges/data.pt')
    
    x1 =  data['x1'].to(device)
    x0_test = data['x0_test'].to(device)
    x1_test = data['x1_test'].to(device)
    
    # Загрузка параметров обученных моделей
    model_list = torch.load("Bridges/model_list.pt")
    
    # Формируем модель заготовку
    dim = cfg.dim
    
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
    
    if cfg.model_name == "dsbm":
        model = DSBM(net_fwd=net_fn().to(device), 
                      net_bwd=net_fn().to(device), 
                      num_steps=num_steps, sig=sigma, first_coupling=cfg.first_coupling)
    
    model.load_state_dict(model_list[-1]['model'])

    return x1, x1_test, x0_test, model 




def get_traj_dataloader(model, x1):

    # Отбор начальных состояний для обучения
    indices = torch.randperm(x1.size(0))[:2000]
    x1 = x1[indices]
    
    # Моделирование траекторий для каждого отобранного начального условия
    N = 200 # Количество итераций SDE для построения каждой траектории
    cnt=10 # Количество имитаций для каждого начального условия
    
    traj = model.sample_sde(zstart=x1, fb='b', N=N)
    tensor_3d = torch.stack(traj)
    
    for i in range(cnt):
        traj = model.sample_sde(zstart=x1, fb='b', N=N)
        tensor_tmp = torch.stack(traj)
        
        tensor_3d = torch.cat((tensor_3d, tensor_tmp), dim=1)
    
    tensor_3d = tensor_3d.permute(1, 0, 2)

    dataset = TensorDataset(tensor_3d)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    return dataloader, tensor_3d



class NeuralODE(nn.Module):

    def __init__(self):
        super(NeuralODE, self).__init__()
        m = 64
        dim=5
        self.net = nn.Sequential(
            nn.Linear(dim, m),
            nn.SiLU(),
            nn.Linear(m, m),
            nn.SiLU(),
            nn.Linear(m, dim),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y)



def train_NeuralODE(model_bridge, NeuralODE_model, x_train, x1_test, x0_test):

    # Генерация траеткторий для обучения
    dataloader, tensor_train = get_traj_dataloader(model_bridge, x_train)

    loss_list = []
    dim = tensor_train.shape[2]

    # Определение целевых показателей
    # train
    train_target_mean = tensor_train[:,-1,:].mean(0).mean(0)
    train_target_cov = torch.cov(torch.cat([tensor_train[:,0,:], tensor_train[:,-1,:]], dim=1).T)[dim:, :dim].diag().mean(0)
    train_target_var = tensor_train[:,-1,:].var(dim=0).mean(0)

    train_optimal_result_dict = {'mean':train_target_mean.item(), 'var':  train_target_var.item(), 'cov': train_target_cov.item()}
    result_list_train = {k: [] for k in train_optimal_result_dict.keys()}

    # test
    test_target_mean = x0_test.mean(0).mean(0)
    test_target_cov = (np.sqrt(5) - 1) / 2
    test_target_var = x0_test.var(dim=0).mean(0)

    test_optimal_result_dict = {'mean':test_target_mean.item(), 'var':  test_target_var.item(), 'cov': test_target_cov}
    result_list_test = {k: [] for k in test_optimal_result_dict.keys()}


    # Train-loop
    num_epochs = 5
    
    optimizer =optim.RMSprop(NeuralODE_model.parameters(), lr=1e-4,weight_decay=0)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    
    
    for epoch in range(num_epochs):
        NeuralODE_model.train()
        total_loss = 0.0
        with tqdm(total=len(dataloader), desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as pbar:
            for batch in dataloader:
                optimizer.zero_grad()  # Обнуляем градиенты
                
                # Получаем текущую обучающую пару
                y0 = batch[0][:, 0, :]  # Начальное состояние
                t = torch.linspace(0, 1, steps=201)  # Временные точки
        
                # Прогон
                pred = odeint(NeuralODE_model, y0, t)
                pred = pred.permute(1, 0, 2)  # Согласование размерности
    
                # Вычисление функции потерь
                # MAE по траекторям (точкам)
                loss_base = torch.mean(torch.abs(pred - batch[0]))
    
                # MAE по дисперсии конечного распределения
                pred_final_var = pred[:, -1, :].var(dim=0).mean(0)
                loss_var = 1*torch.abs(pred_final_var - train_target_var)
                
                # MAE по кросс-ковариации
                dim = pred.shape[2]
                initial_state = pred[:, 0, :] 
                final_state = pred[:, -1, :]
                combined = torch.cat([initial_state, final_state], dim=1)
                cov_matrix = torch.cov(combined.T)
                cov_pred = cov_matrix[:dim, dim:].diag().mean()
                
                loss_cov = 3*torch.abs(cov_pred - train_target_cov)
    
                # Полная ошибка
                loss = (loss_base + loss_var + loss_cov)
                
                loss.backward()
                optimizer.step()
                
                loss_list.append(loss.item())
                total_loss += loss.item()
    
                pbar.update(1)  # Увеличиваем прогресс-бар на 1
                
        epoch_loss = total_loss / len(dataloader)
        scheduler.step(epoch_loss)
    
        # Расчет метрик качества на train и test
        with torch.no_grad():
            # на train
            pred_y = odeint(NeuralODE_model, x_train, t)
            result_list_train['mean'].append(pred_y[-1].mean(0).mean(0).item())
            result_list_train['var'].append(pred_y[-1].var(0).mean(0).item())
            result_list_train['cov'].append(torch.cov(torch.cat([pred_y[0], pred_y[-1]], dim=1).T)[dim:, :dim].diag().mean(0).item())
    
            # на test
            pred_y = odeint(NeuralODE_model, x1_test, t)
            result_list_test['mean'].append(pred_y[-1].mean(0).mean(0).item())
            result_list_test['var'].append(pred_y[-1].var(0).mean(0).item())
            result_list_test['cov'].append(torch.cov(torch.cat([pred_y[0], pred_y[-1]], dim=1).T)[dim:, :dim].diag().mean(0).item())

        
    return NeuralODE_model, result_list_train, result_list_test, train_optimal_result_dict, test_optimal_result_dict




def main():

    # Загружаем обученный мост и train-test датасеты
    x_train, x1_test, x0_test, model_bridge = get_bridge()

    # Инициализация NeuralODE
    NeuralODE_model = NeuralODE() 

    # Обучение NeuralODE
    NeuralODE_model, result_list_train, result_list_test, train_optimal_result_dict, test_optimal_result_dict = train_NeuralODE(model_bridge, NeuralODE_model, x_train, x1_test, x0_test)

    # Сохранение результатов
    # Создание директории текущего эксперимента
    save_directory = 'plots_and_results'
    
    if os.path.exists(save_directory):
        shutil.rmtree(save_directory)
    os.makedirs(save_directory, exist_ok=True)

    # train
    for i, k in enumerate(result_list_train.keys()):
      plt.plot(result_list_train[k])
      plt.plot(np.arange(len(result_list_train[k])), train_optimal_result_dict[k] * np.ones(len(result_list_train[k])), label="optimal", linestyle="--")
      plt.title(k.capitalize())
      if i == 0:
        plt.legend()
      plt.savefig(save_directory + f"/train_convergence_{k}.png")
      plt.close()

    # test
    for i, k in enumerate(result_list_test.keys()):
      plt.plot(result_list_test[k])
      plt.plot(np.arange(len(result_list_test[k])), test_optimal_result_dict[k] * np.ones(len(result_list_test[k])), label="optimal", linestyle="--")
      plt.title(k.capitalize())
      if i == 0:
        plt.legend()
      plt.savefig(save_directory + f"/test_convergence_{k}.png")
      plt.close()
    
    # Сохранение NeuralODE
    torch.save(NeuralODE_model, save_directory+'/NeuralODE.pth')


if __name__ == "__main__":
    main()