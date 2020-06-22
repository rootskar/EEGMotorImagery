#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Karel Roots"


"""
Class for holding data about a given model under evaluation.
Includes the model name (String), trial type (0 - Executed or 1 - Imagined),
the layers to disable during transfer learning (List), the Tensorflow model object and
if the model uses multiple branches.

After the experiment has been run on the model, the data about model accuracy can be accessed.
"""


class Model(object):

    def __init__(self, model_name, trial_type, disabled_layers, model, multi_branch=False):
        self.model_name = model_name + '_' + trial_type.name
        self.trial_type = trial_type
        self.disabled_layers = disabled_layers
        self.multi_branch = multi_branch
        self.model = model
        self.equals = []
        self.accuracy = 0

    def get_model(self):
        return self.model

    def get_name(self):
        return self.model_name

    def get_disabled_layers(self):
        return self.disabled_layers

    def get_type(self):
        return self.trial_type

    def get_mb(self):
        return self.multi_branch

    def set_equals(self, equals):
        self.equals = equals

    def get_equals(self):
        return self.equals

    def set_accuracy(self, acc):
        self.accuracy = acc

    def get_accuracy(self):
        return self.accuracy
