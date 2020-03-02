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
from EEGModels import EEGNet, ShallowConvNet, DeepConvNet, EEGNet_fusion
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

# Settings
print("Starting job with args:")
print(sys.argv)
nr_of_subj = int(sys.argv[1])
nr_of_epochs = int(sys.argv[2])
trial_type = int(sys.argv[3])

nb_classes = 2 # number of target classes
use_kfold = True # if using kfold cross validation
kfold_n = 10 # number of folds in kfold validation

# Settings for transfer learning
trials_per_subject = 3 * 15 * 8
subj_for_training = 100 # The number of subjects that should be used for pre-training the model for TL
subj_for_transfer_learning = 3 # The number of subject that should be used for individual evaluation of the TL model

# Loading data from files
X, y = load_data(nr_of_subj=nr_of_subj, trial_type=trial_type, chunk_data=True, chunks=8, 
                 preprocessing=True, hp_freq=0.5, bp_low=2, bp_high=60, notch=True, 
                 hp_filter=False, bp_filter=True, artifact_removal=True, normalize=False)

# methods for saving/loading the data to/from files
#np.savez_compressed(file_name, data=X, labels=y)
#file_name = 'executed_preprocessed_nohp.npz' if trial_type == 1 else 'imagined_preprocessed_nohp.npz'
#data = np.load(file_name, allow_pickle=True)
#X = data['data']
#y = data['labels']

# Data formatting
samples = X.shape[2]
X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
y = to_categorical(y, nb_classes)

# Extracting 100-subject data
trials_for_100_subjects = subj_for_training * trials_per_subject
X_100 = X[:trials_for_100_subjects]
y_100 = y[:trials_for_100_subjects]

# Extracting first 6 subject data
trials_for_6_subjects = 6*trials_per_subject
X_6 = X[:trials_for_6_subjects]
y_6 = y[:trials_for_6_subjects]

# Extracting first 20 subject data
trials_for_20_subjects = 20*trials_per_subject
X_20 = X[:trials_for_20_subjects]
y_20 = y[:trials_for_20_subjects]

# Extracting data of last 3 subjects for transfer learning
X_tl = []
y_tl = []
start_idx = trials_for_100_subjects
for i in range(subj_for_transfer_learning):
    end_idx = start_idx + trials_per_subject
    X_tl.append(X[start_idx:end_idx])
    y_tl.append(y[start_idx:end_idx])
    start_idx = end_idx

print("X_train shape:")
print(X_100.shape)
print("y_train shape:")
print(y_100.shape)

# Make directories for model binaries
DIR = ['./model', './history']
for directory in DIR:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Model Training

"""
@input - Name of the model (String); number of classification classes (int); number of samples (int); if using cpu mode (boolean)

Switch method to return the model with input name and parameters 

@output - Model object
"""
def determine_model(model_name, nb_classes, samples):
    switcher = {
        "EEGNet": EEGNet(nb_classes, Samples = samples),
        "ShallowConvNet": ShallowConvNet(nb_classes, Samples = samples),
        "DeepConvNet": DeepConvNet(nb_classes, Samples = samples),
        "EEGNet_fusion": EEGNet_fusion(nb_classes, Samples = samples)
    }

    return switcher.get(model_name, "Invalid model name")

models = [('EEGNet_fusion', [(0, 8), (14, 22), (28, 36)]), ('EEGNet',  [(0, 8), (14, 22), (28, 36)]), 
('ShallowConvNet',  [(0, 8)]), ('DeepConvNet', [(0, 20)])]

for model_name, disabled_layers in models:
    model = determine_model(model_name, nb_classes, samples)
    multi_branch = True if model_name == 'EEGNet_fusion' else False

    # test model with subjects 1-6
    run_model(X_6, y_6, model, model_name=model_name, multi_branch=multi_branch, classes=nb_classes, samples=80, use_kfold=use_kfold, kfold_n=kfold_n)

    # for imagined tasks test model with subjects 1-20
    if trial_type == 2:
        run_model(X_20, y_20, model, model_name=model_name, multi_branch=multi_branch, classes=nb_classes, samples=80, use_kfold=use_kfold, kfold_n=kfold_n)
        
    # test model with all subjects
    run_model(X, y, model, model_name=model_name, multi_branch=multi_branch, classes=nb_classes, samples=80, use_kfold=use_kfold, kfold_n=kfold_n)

    # test model with transfer learning
    run_transfer_learning(model_name, model, X_100, y_100, X_tl, y_tl, subj_for_training, disabled_layers, samples=samples, classes=nb_classes, 
                        different_test_sizes=True, use_kfold=use_kfold, kfold_n=kfold_n, multi_branch=multi_branch, pre_train=True)
