# Ускорение процесса обучения/построения диффузионного моста.


Исследование возможности ускорения обучения/построения диффузионного моста методом DSBM путем изменения алгоритма (порядка) обучения.


## 📁 Файловая структура

```  
 exp-1_DSBM_trainModif/
├── results/                    # Trained bridge models
│   ├── gaussian_dim-.../       # results of base trainer
│   ├── gaussian_dim-..._MOD/   # results of MOD trainer
│   └── ...
│    
├── configurations/             
│   └── gaussian.yaml              # Configuration file
│    
├── notebooks/                     # Jupyter notebooks
│   ├── 1_DSBM_train.ipynb         # Main base training script
│   ├── 2_DSBM_train_mod.ipynb     # Main MOD training script
│   └── 3_DSBM_comparison.ipynb    # base-MOD comparison script
│    
├── src/                           # Scripts
│   └── ...
└── requirements.txt          # Python dependencies
```  

## 🎯 Установка и запуск
1. `Python 3.12`
1. Для запуска проекта вам потребуются следующие библиотеки. Установите их с помощью pip:
```bash  
pip install -r requirements.txt   
```

3. Основной способ запуска — через Jupyter Notebook скриптов `notebooks/ ... .ipynb`.

## 🔧 Конфигурация (алгоритм)

1. Скрипт `1_DSBM_train.ipynb`обеспечивает построение диффузионного моста с параметрами из `configurations/gaussian.yaml` с базовым алгоритмом и параметрами обучения из https://github.com/yuyang-shi/dsbm-pytorch/. В процессе обучения фиксируется время обучения кажджой эпохи и метрики качества. Все данные сохраняются в директории `result`.
2. Скрипт `2_DSBM_train_mod.ipynb`обеспечивает построение диффузионного моста с параметрами из `configurations/gaussian_MOD.yaml` с модифицированным алгоритмом (порядком обучения). В процессе обучения фиксируется время обучения кажджой эпохи и метрики качества. Все данные сохраняются в директории `result` с пометкой `_MOD`.
3. Скрипт `3_DSBM_comparison.ipynb` выгружает сохраненные результаты, полученные для одного эксперимента с использование базового алгоритма и модифицированного и сравнивает их.

## Основные результаты

1. Было проведено сравнение алгоритмов обучения для задачи построения диффузионного моста между гауссианами со следующими параметрами:
    - `gaussian_dim-5__mean-0.1_var_1`
    - `gaussian_dim-5_mean-10_var-1`.
2. Результаты построения мостов сохранены в `result`. В `3_DSBM_comparison.ipynb` произведен сравнительный анализ результатов.
3. Значительно повышение скорости обучения до достижения сравнительного качества моста:
    - для `gaussian_dim-5__mean-0.1_var_1` = **24.0  мин. -> 3.0  мин.**
    - для `gaussian_dim-5_mean-10_var-1` = **25.0  мин. -> 2.0  мин.**
4. `DSBM_train_mod` обеспечил более гладкую траекторию сходимости.