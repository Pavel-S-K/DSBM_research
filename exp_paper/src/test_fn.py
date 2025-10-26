import sys
sys.path.append('..')
from src.generate_gaussian_cloud import generate_gaussian_cloud
from src.wasserstein_distance import empirical_wasserstein_distance
import torch
import numpy as np
import pandas as pd
import time
from tqdm import tqdm


def test_fn(cfg, model, N_samples):

    results_SWD = []
    device = 'cpu'

    dim = cfg.dim
    x0_vars_diag = list(cfg.x0_vars_diag) * dim
    x0_cov_pairs = list(cfg.x0_cov_pairs) * dim
    x0_mean = list(cfg.x0_mean) * dim
    
    x1_vars_diag = list(cfg.x1_vars_diag) * dim
    x1_cov_pairs = list(cfg.x1_cov_pairs) * dim
    x1_mean =list(cfg.x1_mean) * dim
    
    # train
    x0 = generate_gaussian_cloud(
        vars_diag=x0_vars_diag,  
        cov_pairs=x0_cov_pairs, 
        mean=x0_mean, 
        dataset_size=cfg.dataset_size, 
        plot=False
    )  

    
    x0 = torch.tensor(x0).to(torch.float32).to(device)
    
    for i in tqdm(range(N_samples)):
    
        # Формируем начальное распределение
        x1 = generate_gaussian_cloud(
            vars_diag=x1_vars_diag,  
            cov_pairs=x1_cov_pairs, 
            mean=x1_mean, 
            dataset_size=cfg.dataset_size, 
            plot=False
        )  

        
        x1 = torch.tensor(x1).to(torch.float32).to(device)
    
        # Инференс
        start_time = time.time()
        traj = model.eval().sample_sde(zstart=x1, fb='b', N=cfg.num_steps)
        end_time = time.time()
    
        # Расчет статистик
        end_mean = traj[-1].mean(0).mean(0).item()
        end_var =  traj[-1].var(0).mean(0).item()
        traj_cov = torch.cov(torch.cat([traj[0], traj[-1]], dim=1).T)[dim:, :dim].diag().mean(0).item()
    
        mean_opt = x0.mean(0).mean(0).item()
        var_opt =  x0.var(0).mean(0).item()
        
        W = empirical_wasserstein_distance(traj[-1].numpy(), x1.numpy())
        W_opt = empirical_wasserstein_distance(x0.numpy(), x1.numpy())
    
        results_SWD.append({
                'sample_№': i,
                'inference_time, сек.': end_time - start_time,
                '2-WD_opt': W_opt,
                '2-WD': W,
                'delta_W, %': 100*np.abs(W_opt-W)/W_opt,
                'mean_opt': mean_opt,
                'mean': end_mean,
                'delta_mean, %': 100*np.abs((mean_opt-end_mean)/mean_opt),
                'var_opt': var_opt,
                'var': end_var,
                'delta_var, %': 100*np.abs(var_opt-end_var)/var_opt,
                'cov': traj_cov
            })
    
    df_SWD = pd.DataFrame(results_SWD)

    return df_SWD