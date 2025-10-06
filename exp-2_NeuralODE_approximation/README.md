# Аппроксимация диффузионного моста


Исследование возможности аппроксимации SDE диффузионного моста с помощью нейронных обыкновенных дифференциальных уравнений (NeuralODE).


## 📁 Файловая структура

```  
exp-2_NeuralODE/
├── Bridges/                    # Trained bridge models
│   ├── gaussian_dim-5.../
│   │   ├── data.pt            # Training/test data
│   │   ├── model_list.pt      # Bridge_model parameters
│   │   └── *.png              # Bridge training metrics plots
│   └── ...
├── results/                  # NeuralODE training results
│   ├── NeuralODE.pt          # Final trained model
│   ├── train_*.png           # Training metrics
│   └── test_*.png            # Test metrics
├── notebooks/                # Jupyter notebooks
│   └── NeuralODE_train.ipynb # Main training script
├── src/                      # Scripts
└── requirements.txt          # Python dependencies
```  

## 🎯 Установка и запуск
1. `Python 3.12`
1. Для запуска проекта вам потребуются следующие библиотеки. Установите их с помощью pip:
```bash  
pip install -r requirements.txt   
```

3. Основной способ запуска — через Jupyter Notebook `notebooks/NeuralODE_train.ipynb`.


## 🔧 Конфигурация (алгоритм)

1. В директории `Bridges/` находятся директории с заранее предобученными мостами, параметрами обучения (`config.yaml`), целевыми датасетами (гауссианами) и метриками (файлы `.png`). Диффузионный мост строится между двумя многомерными гауссовскими распределениями с параметрами, указанными в имени директории соответствующего моста;
2. Обучение NeuralODE осуществляется с использованием `odeint` библиотеки `torchdiffeq`;
3. Обучение осуществляется на траекториях, полученных путем инференса SDE предобученного диффузионного моста: для каждого начального условия сэмплируется несколько траекторий, которые и составляют обучающий датасет;
4. В конце каждой эпохи обучения осуществляется инференс и оценка качества с сохранением результатов в директории `result`. 

## Основные результаты
В директории `result` находятся результаты обучения NeuralODE для нескольких диффузионных мостов
