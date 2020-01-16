#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from data_loader import load_data
from EEGModels import EEGNet, EEGNet_old, ShallowConvNet, DeepConvNet, DeeperConvNet, ConvNet3D 
import os
import sys

# %%
# data loading
print("Starting job with args:")
print(sys.argv)
nr_of_subj = int(sys.argv[1])
nr_of_epochs = int(sys.argv[2])
trial_type = int(sys.argv[3]) # 1 => executed trials only; 2 => imagined trials only; 3 => both trials

X, y = load_data(nr_of_subj=nr_of_subj, nr_of_epochs=nr_of_epochs, trial_type=trial_type)

# %%
# data preprocessing
X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])

# %%
# defining model

from tensorflow.keras import backend as K
from tensorflow.keras import callbacks

K.set_image_data_format('channels_first')

# Make directories for model binaries
DIR = ['./model', './history']
for directory in DIR:
    if not os.path.exists(directory):
        os.makedirs(directory)


# %%
# train model

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.losses import categorical_crossentropy
from sklearn.model_selection import KFold
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from statistics import mean

def train_test_model(model, X_train, X_test, y_train, y_test, model_name):
    MODEL_LIST = glob('./model/*')
    model_name = './model/model' + str(len(MODEL_LIST)) + '.h5'
    print("New model name: " + model_name)

    callbacks_list = [callbacks.ModelCheckpoint(model_name,
                                        save_best_only=True,
                                        monitor='val_loss'),
            callbacks.EarlyStopping(monitor='val_acc', patience=10),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5),
            callbacks.TensorBoard(log_dir='./my_log_dir',
                                histogram_freq=0,
                                write_graph=True,
                                write_images=True)]

    model.compile(loss=categorical_crossentropy, optimizer=Adam(lr=0.001), metrics=['accuracy'])
    history = model.fit(X_train, y_train, batch_size=64, epochs=nr_of_epochs, validation_split=0.04, verbose=True, callbacks=callbacks_list)

    # %%
    # test model

    #model.load_weights(model_name)
    #result = model.evaluate(X_test, y_test, batch_size=64, verbose=False)
    #print(result)

    # %%
    # alternative prediction based testing

    probs       = model.predict(X_test)
    preds       = probs.argmax(axis = -1)  
    acc         = np.mean(preds == y_test.argmax(axis=-1))
    print("Classification accuracy for %s : %f " % (model_name, acc))

    return acc

def run_model(model, X, y, model_name = 'Noname', kfold=False, kfold_n=2):
    
    if kfold:
        kfold = KFold(kfold_n, True, 42)
        accs = []
        
        for train_idx, test_idx in kfold.split(X):
            X_train, X_test = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            y_train = to_categorical(y_train, nr_of_subj)
            y_test = to_categorical(y_test, nr_of_subj)

            result = train_test_model(model, X_train, X_test, y_train, y_test, model_name)
            accs.append(result)
        
        print("Average classification accuracy for %s : %f " % (model_name, mean(accs)))

    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
        y_train = to_categorical(y_train, nr_of_subj)
        y_test = to_categorical(y_test, nr_of_subj)

        print("Train/test shapes:")
        print(X_train.shape)
        print(y_train.shape)
        print(X_test.shape)
        print(y_test.shape)

        train_test_model(model, X_train, X_test, y_train, y_test, model_name)


model = EEGNet(2, Samples = 640, kfold=True, kfold_n=4)
run_model(model, X, y, model_name="EEGNet")

model = ShallowConvNet(2, Samples = 640, kfold=True, kfold_n=4)
run_model(model, X, y, model_name="ShallowConvNet")

