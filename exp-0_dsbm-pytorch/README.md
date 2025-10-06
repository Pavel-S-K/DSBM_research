<<<<<<< HEAD
# Реализация алгоритма DSBM построения диффузионного моста


Реализация алгоритма DSBM построеняи диффузионного моста между двумя гауссианами. Исследование сходимости.
=======
# Исследование алгоритма DSBM построения диффузионного моста

- Источник алгоритма https://github.com/yuyang-shi/dsbm-pytorch/
- Исследование принципа работы алгоритма DSBM построеняи диффузионного моста между двумя гауссианами. Исследование сходимости.
>>>>>>> 581260f6ecb9a72dda5115fafca40e34a1f31ab4


## 📁 Файловая структура

```  
exp-0_dsbm-pytorch/
├── results/                    # Trained bridge models
│   ├── gaussian_dim-5.../
│   │   ├── data.pt            # Training/test data
│   │   ├── model_list.pt      # Bridge_model parameters
│   │   └── *.png              # Bridge training metrics plots
│   └── ...
├── configurations/             
│   └── gaussian.yaml         # Configuration file
├── notebooks/                # Jupyter notebooks
│   └── DSBM_research.ipynb   # Main training script
├── src/                      # Scripts
│   └── ...
└── requirements.txt          # Python dependencies
```  

## 🎯 Установка и запуск
1. `Python 3.12`
1. Для запуска проекта вам потребуются следующие библиотеки. Установите их с помощью pip:
```bash  
pip install -r requirements.txt   
```

3. Основной способ запуска — через Jupyter Notebook `notebooks/DSBM_research.ipynb`.

## Основные результаты

<<<<<<< HEAD
В директории `result` находятся результаты построения диффузионных мостов для нескольких вариантов гауссиан
=======
В директории `result` находятся результаты построения диффузионных мостов для нескольких вариантов гауссиан
>>>>>>> 581260f6ecb9a72dda5115fafca40e34a1f31ab4
