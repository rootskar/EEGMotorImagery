#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from tensorflow.keras.models import Model
from statistics import mean


"""
@input - model (Object); testing data (List); testing labels (List); model name (String); optional args
         
Method that predicts target values with given model and calculates the accuracy of the predicitions by mean value of correct answers.

@output - Accuracy value (float)
"""
def predict_accuracy(model, X_test, y_test, model_name, multi_branch=False, tl=False, n=0, subj=1, kfold_n=1, train_size=0.8):
    if multi_branch:
        probs = model.predict([X_test, X_test, X_test])
    else:
        probs = model.predict(X_test)
        
    preds = probs.argmax(axis = -1)  
    acc = np.mean(preds == y_test.argmax(axis=-1))

    if tl:
        print("Transfer learning classification accuracy for kfold size %d ; train_size %d ; iteration %d ; model %s ; subject %d : %f " % (kfold_n, train_size, n, model_name, subj, acc))
    else:
        print("Classification accuracy for %s : %f " % (model_name, acc))

    return acc