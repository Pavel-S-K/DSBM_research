{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "f520df3e-524f-41d6-b37c-739ba426f33d",
   "metadata": {},
   "source": [
    "# Аппроксимация диффузионного моста\n",
    "\n",
    "\n",
    "Исследование возможности аппроксимации SDE диффузионного моста с помощью нейронных обыкновенных дифференциальных уравнений (NeuralODE).\n",
    "\n",
    "\n",
    "## 📁 Файловая структура\n",
    "\n",
    "```  \n",
    "exp-2_NeuralODE/\n",
    "├── Bridges/                    # Trained bridge models\n",
    "│   ├── gaussian_dim-5.../\n",
    "│   │   ├── data.pt            # Training/test data\n",
    "│   │   ├── model_list.pt      # Bridge_model parameters\n",
    "│   │   └── *.png              # Bridge training metrics plots\n",
    "│   └── ...\n",
    "├── results/                  # NeuralODE training results\n",
    "│   ├── NeuralODE.pt          # Final trained model\n",
    "│   ├── train_*.png           # Training metrics\n",
    "│   └── test_*.png            # Test metrics\n",
    "├── notebooks/                # Jupyter notebooks\n",
    "│   └── NeuralODE_train.ipynb # Main training script\n",
    "└── requirements.txt          # Python dependencies\n",
    "```  \n",
    "\n",
    "## 🎯 Установка и запуск\n",
    "1. `Python 3.12`\n",
    "1. Для запуска проекта вам потребуются следующие библиотеки. Установите их с помощью pip:\n",
    "```bash  \n",
    "pip install -r requirements.txt   \n",
    "```\n",
    "\n",
    "3. Основной способ запуска — через Jupyter Notebook `notebooks/NeuralODE_train.ipynb`.\n",
    "\n",
    "\n",
    "## 🔧 Конфигурация (алгоритм)\n",
    "\n",
    "1. В директории `Bridges/` находятся директории с заранее предобученными мостами, параметрами обучения (`config.yaml`), целевыми датасетами (гауссианами) и метриками (файлы `.png`). Диффузионный мост строится между двумя многомерными гауссовскими распределениями с параметрами, указанными в имени директории соответствующего моста;\n",
    "2. Обучение NeuralODE осуществляется с использованием `odeint` библиотеки `torchdiffeq`;\n",
    "3. Обучение осуществляется на траекториях, полученных путем инференса SDE предобученного диффузионного моста: для каждого начального условия сэмплируется несколько траекторий, которые и составляют обучающий датасет;\n",
    "4. В конце каждой эпохи обучения осуществляется инференс и оценка качества с сохранением результатов в директории `result`. \n",
    "\n",
    "## Основные результаты\n",
    "\n",
    "В директории `result` находятся результаты обучения NeuralODE для нескольких диффузионных мостов"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
