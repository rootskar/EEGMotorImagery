#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Karel Roots"

import os
import sys

import numpy as np
from EEGModels import get_models
from data_loader import load_data
from experiment import Experiment
from mcnemar import mcnemar_test
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from training_testing import run_experiment
from run_type import RunType

"""
Required dependencies: 
Python == 3.7.7, Tensorflow == 2.1.0, Numpy >= 1.18.1, scikit-learn >= 0.22.1, pyEDFlib >= 0.1.17
statsmodels >= 0.11.1, Gumpy (https://github.com/gumpy-bci/gumpy)
The program can be run from the CLI with the following required arguments:
1.) The number of subjects to be used from the dataset (int)
2.) The number of epochs the training of models should be done (int)
3.) The number of target classes in the classification (int)
4.) What type of trials should be extracted from the data; 1 => executed trials only; 2 => imagined trials only
5.) If CPU-only mode should be used (True / False)

Example: python run_experiments.py 109 100 2 1 True
"""


# Settings
if len(sys.argv) < 6:
    raise AttributeError("Input requires 6 arguments: number of subjects (int), number of training epochs (int), " +
                         "number of classes (int), trial type (0 or 1, int), if cpu mode should be used (boolean)")

print("Starting job with args:")
print(sys.argv)
nr_of_subj = int(sys.argv[1])
nr_of_epochs = int(sys.argv[2])
nb_classes = int(sys.argv[3])
trial_type = RunType.Executed if int(sys.argv[4]) == 1 else RunType.Imagined
use_cpu = True if sys.argv[5] == 'True' else False

# Settings for transfer learning
trials_per_subject = 3 * 15 * 8
subj_for_training = 100  # The number of subjects that should be used for pre-training the model for TL
subj_for_transfer_learning = 3  # The number of subject that should be used for individual evaluation of the TL model

# Loading data from files
X, y = load_data(nr_of_subj=nr_of_subj, trial_type=trial_type, chunk_data=True, chunks=8, cpu_format=use_cpu,
                 preprocessing=True, hp_freq=0.5, bp_low=2, bp_high=60, notch=True,
                 hp_filter=False, bp_filter=True, artifact_removal=True)

# methods for saving/loading the data to/from files
# file_name = 'executed.npz' if trial_type == RunType.Executed else 'imagined.npz'
# np.savez_compressed(file_name, data=X, labels=y)
# print("Loaded data from file %s" % (file_name))
# data = np.load(file_name, allow_pickle=True)
# X = data['data']
# y = data['labels']

# Data formatting
if use_cpu:
    print("Using CPU")
    K.set_image_data_format('channels_last')
    samples = X.shape[1]
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
else:
    print("Using GPU")
    K.set_image_data_format('channels_first')
    samples = X.shape[2]
    X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
y = to_categorical(y, nb_classes)

print("X shape: {}".format(X.shape))
print("y shape: {}".format(y.shape))

# Make directories for model binaries
DIR = ['./model', './history']
for directory in DIR:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Perform experiments
experiments = []

# test models with all subjects
experiment_103sub = Experiment(trial_type, '103sub', get_models(trial_type, nb_classes, samples, use_cpu), nr_of_epochs,
                               0.125, 0.2)
experiments.append(run_experiment(X, y, experiment_103sub))

# Calculate Mcnemar's test statistic and p-value for all experiments
models = ['EEGNet', 'ShallowConvNet', 'DeepConvNet']
for experiment in experiments:
    fusion_eqs = experiment.get_model('EEGNet_fusion').get_equals()
    eqs_list = []
    
    # evaluate EEGNet Fusion against the state-of-the-art models under evaluation
    for model_name in models:
        print("EEGNet Fusion vs {}".format(model_name))
        model_eqs = experiment.get_model(model_name).get_equals()
        mcnemar_test(fusion_eqs, model_eqs)
        eqs_list.append(model_eqs)
    
    # save equals lists in .npz file for future analysis
    np.savez_compressed(experiment.get_exp_type() + '_' + experiment.get_trial_type().name + '_eq_values.npz',
                        fusion=fusion_eqs, eegnet=eqs_list[0], shallow=eqs_list[1], deep=eqs_list[2])
