#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Karel Roots"

import time
from glob import glob
import numpy as np

from predict import predict_accuracy
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras import callbacks
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import precision_score, recall_score, f1_score


"""
@input - model (Object); model name (String), training data (List); training labels (List); validation data (List); 
validation labels (List); testing data (List); testing labels (List); if model is using multiple branches (bool); 
the number of epochs to train the model (int); if the model should be tested on testing data (bool).

Method that creates a new model and trains and validates it on the input data, while saving the best weights over 
all training epochs. After training the saved weights for best validation loss are loaded and used for 
evaluation on test data.

@output - Trained model; Accuracy value (float); Truth values, 
if the prediction at index i was equal to the target label (List)
"""


def train_test_model(model, model_name, X_train, y_train, X_val, y_val, X_test, y_test, multi_branch, nr_of_epochs,
                     test_model=True):
    MODEL_LIST = glob('./model/*')
    new_model_name = './model/' + str(model_name) + '_' + str(len(MODEL_LIST)) + '.h5'
    print("New model name: " + new_model_name)
    acc = 0
    equals = []

    # Callbacks for saving best model, early stopping when validation accuracy does not increase and reducing
    # learning rate on plateau
    callbacks_list = [callbacks.ModelCheckpoint(new_model_name,
                                                save_best_only=True,
                                                monitor='val_loss'),
                      # callbacks.EarlyStopping(monitor='val_acc', patience=25),
                      callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)]

    model.compile(loss=binary_crossentropy, optimizer=Adam(lr=0.001), metrics=['accuracy'])

    training_start = time.time()
    if multi_branch:
        history = model.fit([X_train, X_train, X_train], y_train, batch_size=64, shuffle=True, epochs=nr_of_epochs,
                            validation_data=([X_val, X_val, X_val], y_val), verbose=False, callbacks=callbacks_list)
    else:
        history = model.fit(X_train, y_train, batch_size=64, shuffle=True, epochs=nr_of_epochs,
                            validation_data=(X_val, y_val), verbose=False, callbacks=callbacks_list)
    
    training_total_time = time.time() - training_start
    print("Model {} total training time was {} seconds".format(model_name, training_total_time))
    print("That is {} seconds per sample".format(training_total_time/X_train.shape[0]))
    print("Train shape: {}. Test shape: {}".format(X_train.shape, X_test.shape))

    # test model predictions
    if test_model:
        model.load_weights(new_model_name)
        testing_start = time.time()
        acc, equals, preds = predict_accuracy(model, X_test, y_test, new_model_name, multi_branch=multi_branch)
        testing_total_time = time.time() - training_start
        print("Model {} total testing time was {} seconds".format(model_name, testing_total_time))
        print("That is {} seconds per sample".format(testing_total_time/X_test.shape[0]))
        
        rounded_labels = np.argmax(y_test, axis=1)

        precision_left = precision_score(rounded_labels, preds, average='binary', pos_label=0)
        print('Precision for left hand: %.3f' % precision_left)

        recall_left = recall_score(rounded_labels, preds, average='binary', pos_label=0)
        print('Recall for left hand: %.3f' % recall_left)

        f1_left = f1_score(rounded_labels, preds, pos_label=0, average='binary')
        print('F1-Score for right hand: %.3f' % f1_left)

        precision_right = precision_score(rounded_labels, preds, average='binary', pos_label=1)
        print('Precision for right hand: %.3f' % precision_right)

        recall_right = recall_score(rounded_labels, preds, average='binary', pos_label=1)
        print('Recall for right hand: %.3f' % recall_right)
        
        f1_right = f1_score(rounded_labels, preds, pos_label=1, average='binary')
        print('F1-Score for right hand: %.3f' % f1_right)


    return model, acc, equals


"""
@input - data (List); target labels (List); experiment (Object); optional args
         
Method that splits the input data into training and testing sets and evaluates the model according to the experiment specifications.

@output - Experiment object with experimental results
"""


def run_experiment(X, y, experiment, use_cpu=False, test_model=True):
    # Set the data format
    if use_cpu:
        K.set_image_data_format('channels_last')
    else:
        K.set_image_data_format('channels_first')

    # training/validation/test set split
    test_split = experiment.get_test_split()
    if test_split == 0:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=experiment.get_val_split(), random_state=42)
        X_test, y_test = [], []
        test_model = False
    else:
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_split, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,
                                                          test_size=experiment.get_val_split(), random_state=42)

    for model in experiment.get_models().values():
        _model = model.get_model()
        model_name = model.get_name() + '_' + experiment.get_exp_type()
        multi_branch = model.get_mb()

        # splitting training/testing sets
        _model, acc, equals = train_test_model(_model, model_name, X_train, y_train, X_val, y_val, X_test, y_test,
                                               multi_branch, experiment.get_epochs(), test_model=test_model)
        _model.save('./model/' + str(model_name) + '_best.h5')

        model.set_accuracy(acc)
        model.set_equals(equals)

    return experiment
