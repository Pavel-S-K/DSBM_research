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
from omegaconf import DictConfig, OmegaConf
from hydra import initialize, compose

from torchdiffeq import odeint
import torch.optim as optim

import time
import os
import shutil
import datetime
import sys

from src.ScoreNetwork import ScoreNetwork
from src.DSBM_model import DSBM, train_dsbm
sys.path.append('..')


device = 'cpu'



def get_bridge(dir_name):
    
    BRIDGE_DIR = f'../Bridges/{dir_name}/'

    # Импорт параметров моста
    with initialize(version_base=None, config_path= BRIDGE_DIR):
        
        cfg_bridge: DictConfig = compose(config_name="config.yaml")
        
    if cfg_bridge.get("seed"):
        pl.seed_everything(cfg_bridge.seed, workers=True)


    # Импорт данных, соответствующих мосту
    data = torch.load(BRIDGE_DIR + 'data.pt')
    
    x1_train =  data['x1'].to(device)
    x0_test = data['x0_test'].to(device)
    x1_test = data['x1_test'].to(device)
    
    # Загрузка параметров обученных моделей
    model_list = torch.load(BRIDGE_DIR + "model_list.pt")
    


    # Собираем мост
    dim = cfg_bridge.dim
    
    net_split = cfg_bridge.net_name.split("_")
    if net_split[0] == "mlp":
        if net_split[1] == "small":
          net_fn = partial(ScoreNetwork, input_dim=dim+1, layer_widths=[128, 128, dim], activation_fn=hydra.utils.get_class(cfg_bridge.activation_fn)())
        else:
          net_fn = partial(ScoreNetwork, input_dim=dim+1, layer_widths=[256, 256, dim], activation_fn=hydra.utils.get_class(cfg_bridge.activation_fn)())
    else:
        raise NotImplementedError
    
    num_steps = cfg_bridge.num_steps
    sigma = cfg_bridge.sigma
    
    if cfg_bridge.model_name == "dsbm":
        model_bridge = DSBM(net_fwd=net_fn().to(device), 
                            net_bwd=net_fn().to(device), 
                            num_steps=num_steps, sig=sigma, first_coupling=cfg_bridge.first_coupling)
    
    model_bridge.load_state_dict(model_list[-1]['model'])
    model_bridge = model_bridge.eval()

    return x1_train, x1_test, x0_test, model_bridge, cfg_bridge



def get_traj_dataloader(model_bridge, x1, num_steps, cnt_samples):

    # Отбор начальных состояний для обучения
    indices = torch.randperm(x1.size(0))[:2000]
    x1 = x1[indices]
    
    # Моделирование траекторий для каждого отобранного начального условия
    N = num_steps # Количество итераций SDE для построения каждой траектории
    cnt = cnt_samples # Количество имитаций для каждого начального условия
    
    traj = model_bridge.sample_sde(zstart=x1, fb='b', N=N)
    tensor_3d = torch.stack(traj)
    
    for i in range(cnt-1):
        traj = model_bridge.sample_sde(zstart=x1, fb='b', N=N)
        tensor_tmp = torch.stack(traj)
        
        tensor_3d = torch.cat((tensor_3d, tensor_tmp), dim=1)
    
    tensor_3d = tensor_3d.permute(1, 0, 2)

    dataset = TensorDataset(tensor_3d)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    return dataloader, tensor_3d