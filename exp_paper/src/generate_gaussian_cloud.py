import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def generate_cloud(mean, cov, n_points):  
    mean = np.asarray(mean)  
    cov = np.asarray(cov)  
    assert mean.shape[0] == cov.shape[0] == cov.shape[1], "Размерности не совпадают"  
    return np.random.multivariate_normal(mean, cov, n_points)  

def build_cov_from_vars_and_covs(vars_diag, cov_pairs):  
    n = len(vars_diag)  
    cov = np.diag(vars_diag)  
    for i, j, v in cov_pairs:  
        cov[i,j] = v  
        cov[j,i] = v  
    return cov  

def generate_gaussian_cloud(vars_diag, cov_pairs, mean=[0,0,0], dataset_size=1000, plot=False):  
    
 
    cov = build_cov_from_vars_and_covs(vars_diag, cov_pairs)

    if plot == True:
        print("Ковариационная матрица:\n", cov)  
    
    # Проверка на положительную определенность  
    eigvals = np.linalg.eigvalsh(cov)  
    if np.any(eigvals <= 1e-9):  
        print("\n!!! ВНИМАНИЕ: Матрица не является положительно определенной. Результат может быть некорректным.")  
        print("Собственные значения:", eigvals)  
        return  
    if plot == True:
        eigenvalues, eigenvectors = np.linalg.eigh(cov)  
        # Сортируем по убыванию собственных значений  
        idx = eigenvalues.argsort()[::-1]   
        eigenvalues = eigenvalues[idx]  
        eigenvectors = eigenvectors[:,idx]  
        
        main_axis = eigenvectors[:, 0]  
        print(f"\nНаправление главной оси (максимальной дисперсии): {np.round(main_axis, 2)}")  
        print(f"Дисперсия вдоль этой оси (собственное значение): {eigenvalues[0]:.2f}\n")  
    
    # 2. Генерируем точки  
    points = generate_cloud(mean, cov, dataset_size)  

    if plot == True:
        # 3. Визуализируем  
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))  
        
        max_lim = np.max(np.abs(points)) * 1.1  
        
        # Проекция на XY  
        sns.scatterplot(ax=axes[0], x=points[:, 0], y=points[:, 1], alpha=0.6)  
        axes[0].set_title('Проекция на X-Y')  
        axes[0].set_xlabel('X'); axes[0].set_ylabel('Y')  
        axes[0].set_xlim(-max_lim, max_lim); axes[0].set_ylim(-max_lim, max_lim)  
        axes[0].set_aspect('equal', adjustable='box')  
        axes[0].grid(True)  
        # Рисуем проекцию главной оси  
        axes[0].arrow(0, 0, main_axis[0], main_axis[1], head_width=0.2, head_length=0.2, fc='r', ec='r', length_includes_head=True)  
    
        # Проекция на XZ  
        sns.scatterplot(ax=axes[1], x=points[:, 0], y=points[:, 2], alpha=0.6)  
        axes[1].set_title('Проекция на X-Z')  
        axes[1].set_xlabel('X'); axes[1].set_ylabel('Z')  
        axes[1].set_xlim(-max_lim, max_lim); axes[1].set_ylim(-max_lim, max_lim)  
        axes[1].set_aspect('equal', adjustable='box')  
        axes[1].grid(True)  
        axes[1].arrow(0, 0, main_axis[0], main_axis[2], head_width=0.2, head_length=0.2, fc='r', ec='r', length_includes_head=True)  
        
        # Проекция на YZ  
        sns.scatterplot(ax=axes[2], x=points[:, 1], y=points[:, 2], alpha=0.6)  
        axes[2].set_title('Проекция на Y-Z')  
        axes[2].set_xlabel('Y'); axes[2].set_ylabel('Z')  
        axes[2].set_xlim(-max_lim, max_lim); axes[2].set_ylim(-max_lim, max_lim)  
        axes[2].set_aspect('equal', adjustable='box')  
        axes[2].grid(True)  
        axes[2].arrow(0, 0, main_axis[1], main_axis[2], head_width=0.2, head_length=0.2, fc='r', ec='r', length_includes_head=True)  
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])  
        plt.show()  

    return points
