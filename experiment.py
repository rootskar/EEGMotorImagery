#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Karel Roots"


"""
Class for holding information about the experiment to be performed.
Includes trial type (0 or 1), experiment type (String), list of models to evaluate, 
nr of epochs to train the classifiers, percentage of data to use for validation and
percentage of data to use for testing.
"""


class Experiment(object):

    def __init__(self, trial_type, exp_type, models, epochs, val_split, test_split):
        self.trial_type = trial_type
        self.exp_type = exp_type
        self.models = models
        self.epochs = epochs
        self.val_split = val_split
        self.test_split = test_split

    def get_model(self, model):
        return self.models[model]

    def get_models(self):
        return self.models

    def get_trial_type(self):
        return self.trial_type

    def get_exp_type(self):
        return self.exp_type

    def get_epochs(self):
        return self.epochs

    def get_val_split(self):
        return self.val_split

    def get_test_split(self):
        return self.test_split
