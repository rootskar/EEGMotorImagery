#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import sys
import pickle
from glob import glob

from data_loader import load_data
from predict import predict_accuracy
from EEGModels import EEGNet, EEGNet_old, ShallowConvNet, DeepConvNet, DeeperConvNet, ConvNet3D, ConvNet2D, ConvNet3D_2, EEGNet_fusion
from transfer_learning import run_transfer_learning
from training_testing import run_model

# %%
# Data Loading
print("Starting job with args:")
print(sys.argv)
nr_of_subj = int(sys.argv[1]) # the number of subjects to be used from the dataset
nr_of_epochs = int(sys.argv[2]) # the number of epochs the training should be done
trial_type = int(sys.argv[3]) # what type of trials should be extracted from the data; 1 => executed trials only; 2 => imagined trials only; 3 => both trials
use_cpu = bool(sys.argv[4]) # if CPU-only mode should be used (True / False)

# Settings for transfer learning
trials_per_subject = 3 * 15 * 8
subj_for_training = 100 # The number of subjects that should be used for pre-training the model for TL
subj_for_transfer_learning = 3 # The number of subject that should be used for individual evaluation of the TL model

X, y = load_data(nr_of_subj=nr_of_subj, trial_type=trial_type, chunk_data=True, chunks=8, 
                 preprocessing=True, hp_freq=0.5, bp_low=2, bp_high=60, notch=True, 
                 hp_filter=True, bp_filter=True, artifact_removal=True, normalize=False)
#data = np.load('filt-clean-win-exec-80-103.npz', allow_pickle=True)
#X = data['data']
#y = data['labels']

# %%
# Data Splitting for Transfer Learning
X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
training_trials = subj_for_training * trials_per_subject
X_train = X[:training_trials]
y_train = y[:training_trials]

X_tl = []
y_tl = []
start_idx = training_trials
for i in range(subj_for_transfer_learning):
    end_idx = start_idx + trials_per_subject
    X_tl.append(X[start_idx:end_idx])
    y_tl.append(y[start_idx:end_idx])
    start_idx = end_idx

print("X_train shape:")
print(X_train.shape)
print("y_train shape:")
print(y_train.shape)

# Make directories for model binaries
DIR = ['./model', './history']
for directory in DIR:
    if not os.path.exists(directory):
        os.makedirs(directory)

# %%
# Model Training

"""
@input - Name of the model (String); number of classification classes (int); number of samples (int); if using cpu mode (boolean)

Switch method to return the model with input name and parameters 

@output - Model object
"""
def determine_model(model_name, nb_classes, samples, cpu = False):
    switcher = {
        "EEGNet": EEGNet(nb_classes, Samples = samples, cpu = cpu),
        "ShallowConvNet": ShallowConvNet(nb_classes, Samples = samples, cpu = cpu),
        "ConvNet2D": ConvNet2D(nb_classes, Samples = samples, cpu = cpu),
        "DeepConvNet": DeepConvNet(nb_classes, Samples = samples, cpu = cpu),
        "DeeperConvNet": DeeperConvNet(nb_classes, Samples = samples, cpu = cpu)
    }

    return switcher.get(model_name, "Invalid model name")

# Evaluating the EEGNet fusion model
model_name = 'EEGNet_fusion'
model = EEGNet_fusion(2, Samples=80)
disabled_layers = [(0, 8), (14, 22), (28, 36)]
# test model with all subjects
run_model(X, y, model, model_name = model_name, multi_branch=True, classes=2, samples=80, kfold=True, kfold_n=10, val_split=0.1)
# test model with transfer learning
run_transfer_learning(model_name, model, X_train, y_train, X_tl, y_tl, subj_for_training, disabled_layers, kfold=True, kfold_n=10, multi_branch=True, pre_train=True)

# Evaluating the EEGNet Model
model_name = 'EEGNet'
model = EEGNet(2, Samples=80)
disabled_layers = [(0, 8), (14, 22), (28, 36)]
# test model with all subjects
run_model(X, y, model, model_name = model_name, multi_branch=True, classes=2, samples=80, kfold=True, kfold_n=10, val_split=0.1)
# test model with transfer learning
run_transfer_learning(model_name, model, X_train, y_train, X_tl, y_tl, subj_for_training, disabled_layers, kfold=True, kfold_n=10, multi_branch=True, pre_train=True)

# Evaluating the ShallowConvNet Model
model_name = 'ShallowConvNet'
model = ShallowConvNet(2, Samples=80)
disabled_layers = [(0, 8)]
# test model with all subjects
run_model(X, y, model, model_name = model_name, classes=2, samples=80, kfold=True, kfold_n=10, val_split=0.1)
# test model with transfer learning
run_transfer_learning(model_name, model, X_train, y_train, X_tl, y_tl, subj_for_training, disabled_layers, kfold=True, kfold_n=10, multi_branch=True, pre_train=True)

# Evaluating the DeepConvNet Model
model_name = 'DeepConvNet'
model = DeepConvNet(2, Samples=80)
disabled_layers = [(0, 20)]
# test model with all subjects
run_model(X, y, model, model_name = model_name, classes=2, samples=80, kfold=True, kfold_n=10, val_split=0.1)
# test model with transfer learning
run_transfer_learning(model_name, model, X_train, y_train, X_tl, y_tl, subj_for_training, disabled_layers, kfold=True, kfold_n=10, multi_branch=True, pre_train=True)
