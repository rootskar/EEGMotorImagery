#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Karel Roots"

from glob import glob

from EEGModels import get_models
from experiment import Experiment
from predict import predict_accuracy
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras import callbacks
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from training_testing import run_experiment

"""
@input - training data (List); training labels (List); validation data (List); validation labels (List);
        testing data (List); testing labels (List); model (Tensorflow Model object); model name (String); 
        index of subject under evaluation (int); layer indexes ranges to disable for TL (tuple) ; optional args
         
Method that loads the best weights for the input model name (if it has been saved before with pretraining),
trains the model with transfer learning approach by disabling specified layers and evaluates the model accuracy on 
test data.

@output - Accuracy value (float); Truth values, if the prediction at index i was equal to the target label (List)
"""


def train_evaluate_transfer(X_train, y_train, X_val, y_val, X_test, y_test, model, model_name, subj, disabled_layers,
                            multi_branch=False, train_size=0.75, nr_of_epochs=100):
    # load the model with best weights from pretraining
    model.load_weights('./model/' + str(model_name) + '_100sub_best.h5')

    MODEL_LIST = glob('./model/*')
    tl_model_name = './model/' + str(model_name) + '_TL_subj_' + str(subj) + '_train_size_' + str(
        int(train_size * 100)) + '_' + str(len(MODEL_LIST)) + '.h5'
    print("New model name: " + tl_model_name)

    # Disable layers for TL
    for r in disabled_layers:
        for layers in (model.layers)[r[0]:r[1]]:
            layers.trainable = False

    # Callbacks for saving the best model with lowest validation loss and reducing learning rate when plateau is reached
    callbacks_list = [callbacks.ModelCheckpoint(tl_model_name, save_best_only=True, monitor='val_loss'),
                      # callbacks.EarlyStopping(monitor='val_acc', patience=25),
                      callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)]

    model.compile(loss=binary_crossentropy, optimizer=Adam(lr=0.001), metrics=['accuracy'])
    if multi_branch:
        history = model.fit([X_train, X_train, X_train], y_train, shuffle=True, batch_size=64,
                            epochs=nr_of_epochs, validation_data=([X_val, X_val, X_val], y_val), verbose=False,
                            callbacks=callbacks_list)
    else:
        history = model.fit(X_train, y_train, shuffle=True, batch_size=64, epochs=nr_of_epochs,
                            validation_data=(X_val, y_val), verbose=False, callbacks=callbacks_list)

    # Test the model
    model.load_weights(tl_model_name)
    return predict_accuracy(model, X_test, y_test, tl_model_name, multi_branch=multi_branch, tl=True,
                            subj=subj, train_size=train_size)


"""
@input - Experiment object; data (List); target labels (List); TL data (List); TL labels (List); 
         nr of subjects used for training (int); trial type (0 or 1); number of classification classes (int);
         number of data points in one sample (int); optional args;
         
Method that runs baseline training/testing, evaluates the test subjects before transfer learning and 
after transfer learning over the number of testing subjects

@output - List of experiment objects with experimental results
"""


def run_tl_experiment(experiment, X, y, X_tl, y_tl, subj_for_training, trial_type, nb_classes, samples,
                      use_cpu=False, pre_train=False):
    # Set the data format
    if use_cpu:
        K.set_image_data_format('channels_last')
    else:
        K.set_image_data_format('channels_first')

    if pre_train:
        # pre-train the base model
        experiment_100sub = Experiment(experiment.get_trial_type(), '100sub',
                                       get_models(trial_type, nb_classes, samples, use_cpu),
                                       experiment.get_epochs(), 0.1, 0)
        run_experiment(X, y, experiment_100sub)

    tl_experiments = []
    # test pre-trained model before TL on each new subject separately
    for i in range(len(X_tl)):
        experiment_pre_tl = Experiment(experiment.get_trial_type(), 'pre_tl_' + str(101 + i),
                                       get_models(trial_type, nb_classes, samples, use_cpu),
                                       experiment.get_epochs(), experiment.get_val_split(), experiment.get_test_split())

        for model in experiment_pre_tl.get_models().values():
            _model = model.get_model()
            model_name = model.get_name()
            subj_nr = subj_for_training + i + 1
            _model.load_weights('./model/' + str(model_name) + '_100sub_best.h5')
            acc, equals = predict_accuracy(_model, X_tl[i], y_tl[i], model_name, tl=True, subj=subj_nr,
                                           train_size=0, multi_branch=model.get_mb())
            model.set_accuracy(acc)
            model.set_equals(equals)
        tl_experiments.append(experiment_pre_tl)

    # use transfer learning for each new subject separately
    for i in range(len(X_tl)):
        experiment_post_tl = Experiment(experiment.get_trial_type(), 'post_tl_' + str(101 + i),
                                        get_models(trial_type, nb_classes, samples, use_cpu),
                                        experiment.get_epochs(), experiment.get_val_split(),
                                        experiment.get_test_split())
        for model in experiment_post_tl.get_models().values():
            _model = model.get_model()
            model_name = model.get_name()

            # training/validation/test set split
            X_train_test, X_val, y_train_test, y_val = train_test_split(X_tl[i], y_tl[i],
                                                                        test_size=experiment.get_val_split(),
                                                                        random_state=42)
            X_train, X_test, y_train, y_test = train_test_split(X_train_test, y_train_test,
                                                                test_size=experiment.get_test_split(), random_state=42)

            train_size = int((X_train.shape[0] / X_tl[i].shape[0]) * 100)

            acc, equals = train_evaluate_transfer(X_train, y_train, X_val, y_val, X_test, y_test, _model, model_name,
                                                  subj_for_training + i + 1,
                                                  model.get_disabled_layers(), multi_branch=model.get_mb(),
                                                  nr_of_epochs=experiment.get_epochs(), train_size=train_size)
            model.set_accuracy(acc)
            model.set_equals(equals)

        tl_experiments.append(experiment_post_tl)

    return tl_experiments
