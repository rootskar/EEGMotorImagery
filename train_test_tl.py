#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import sys
import pickle
from glob import glob

from data_loader import load_data
from tensorflow.keras.utils import to_categorical
from predict import predict_accuracy
from EEGModels import EEGNet, ShallowConvNet, DeepConvNet, DeeperConvNet, ConvNet3D, ConvNet2D, ConvNet3D_2, EEGNet_fusion
from transfer_learning import run_transfer_learning
from training_testing import run_model

"""
Required dependencies: Python >= 3.3, Tensorflow >= 1.4, Numpy >= 1.18.1, scikit-learn >= 0.22, pyEDFlib >= 0.1.15, Gumpy (https://github.com/gumpy-bci/gumpy)
The program can be run from the CLI with the following required arguments:
1.) The numbr of subjects to be used from the dataset
2.) The number of epochs the training of models should be done
3.) What type of trials should be extracted from the data; 1 => executed trials only; 2 => imagined trials only; 3 => both trials
4.) If CPU-only mode should be used (True / False)

Example: python train_test_tl.py 109 100 1 False
"""

# %%
# Settings
print("Starting job with args:")
print(sys.argv)
nr_of_subj = int(sys.argv[1])
nr_of_epochs = int(sys.argv[2])
trial_type = int(sys.argv[3])
use_cpu = bool(sys.argv[4])

nb_classes = 2 # number of target classes
use_kfold = True # if using kfold cross validation
kfold_n = 10 # number of folds in kfold validation

# Settings for transfer learning
trials_per_subject = 3 * 15 * 8
trials_for_6_subjects = 6*trials_per_subject
subj_for_training = 100 # The number of subjects that should be used for pre-training the model for TL
subj_for_transfer_learning = 3 # The number of subject that should be used for individual evaluation of the TL model

# Loading data from files
X, y = load_data(nr_of_subj=nr_of_subj, trial_type=trial_type, chunk_data=True, chunks=8, 
                 preprocessing=True, hp_freq=0.5, bp_low=2, bp_high=60, notch=True, 
                 hp_filter=False, bp_filter=True, artifact_removal=True, normalize=False)
np.savez_compressed('executed_preprocessed_nohp.npz', data=X, labels=y)
#data = np.load('executed_preprocessed.npz', allow_pickle=True)
#X = data['data']
#y = data['labels']

# %%
# Data formatting
X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
y = to_categorical(y, nb_classes)

# Extracting 100-subject data
training_trials = subj_for_training * trials_per_subject
X_train = X[:training_trials]
y_train = y[:training_trials]

# Extracting first 6 subject data
X_6 = X[:trials_for_6_subjects]
y_6 = y[:trials_for_6_subjects]

# Extracting data of last 3 subjects for transfer learning
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
        "DeepConvNet": DeepConvNet(nb_classes, Samples = samples, cpu = cpu),
        "EEGNet_fusion": EEGNet_fusion(nb_classes, Samples = samples, cpu = cpu)
    }

    return switcher.get(model_name, "Invalid model name")

models = [('EEGNet_fusion', [(0, 8), (14, 22), (28, 36)]), ('EEGNet',  [(0, 8), (14, 22), (28, 36)]), 
('ShallowConvNet',  [(0, 8)]), ('DeepConvNet', [(0, 20)])]

for model_name, disabled_layers in models:
    model = determine_model(model_name, nb_classes, X.shape[3])
    multi_branch = True if model_name == 'EEGNet_fusion' else False

    # test model with all subjects 1-6
    run_model(X_6, y_6, model, model_name=model_name, multi_branch=multi_branch, classes=nb_classes, samples=80, use_kfold=use_kfold, kfold_n=kfold_n)
    
    # test model with all subjects
    run_model(X, y, model, model_name=model_name, multi_branch=multi_branch, classes=nb_classes, samples=80, use_kfold=use_kfold, kfold_n=kfold_n)
    
    # test model with transfer learning
    #run_transfer_learning(model_name, model, X_train, y_train, X_tl, y_tl, subj_for_training, disabled_layers, different_test_sizes=True, use_kfold=use_kfold, kfold_n=kfold_n, multi_branch=True, pre_train=False)
