#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow import Session, global_variables_initializer
from tensorflow.keras import backend as K
from tensorflow.keras import callbacks
from sklearn.model_selection import KFold, train_test_split
from statistics import mean
from glob import glob
from predict import predict_accuracy


# Set the data format
K.set_image_data_format('channels_first')

"""
@input - model (Object); training data (List); training labels (List); testing data (List); testing labels (List);
         model name (String); optional args
         
Method that creates a new model and trains it on the input data. 
After training the saved weights for best validation loss are loaded and used for evaluation on test data.

@output - Accuracy value (float)
"""
def train_test_model(model, X_train, y_train, X_val, y_val, X_test, y_test, model_name, nr_of_epochs=100, multi_branch=False):
    MODEL_LIST = glob('./model/*')
    new_model_name = './model/' + str(model_name) + '_' + str(len(MODEL_LIST)) + '.h5'
    print("New model name: " + new_model_name)

    # Callbacks for saving best model, early stopping when validation accuracy does not increase and reducing learning rate on plateau
    callbacks_list = [callbacks.ModelCheckpoint(new_model_name,
                                        save_best_only=True,
                                        monitor='val_loss'),
            #callbacks.EarlyStopping(monitor='val_acc', patience=25),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)]

    model.compile(loss=binary_crossentropy, optimizer=Adam(lr=0.001), metrics=['accuracy'])

    if multi_branch:
        history = model.fit([X_train, X_train, X_train], y_train, batch_size=64, shuffle=True, epochs=nr_of_epochs, validation_data=([X_val, X_val, X_val], y_val), verbose=False, callbacks=callbacks_list)
    else:
        history = model.fit(X_train, y_train, batch_size=64, shuffle=True, epochs=nr_of_epochs, validation_data=(X_val, y_val), verbose=False, callbacks=callbacks_list)

    # test model predictions

    model.load_weights(new_model_name)
    return model, predict_accuracy(model, X_test, y_test, new_model_name, multi_branch=multi_branch)


"""
@input - data (List); target labels (List); optional args
         
Method that splits the input data into training and testing sets and evaluates the model.
If the use_kfold argument is True, the model is kfold cross-validated and the average cross-validation accuracy is printed.

@output - Model object
"""
def run_model(X, y, model, model_name='Noname', classes=2, samples=640, test_split_size=0.2, use_kfold=False, kfold_n=2, val_split=0.04, multi_branch=False):
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_split_size, random_state=42)

    if use_kfold:
        kfold = KFold(kfold_n, True, 42)
        accs = []
        best_acc = 0
        
        for train_idx, val_idx in kfold.split(X_train_val):
            X_val, y_val = X_train_val[val_idx], y_train_val[val_idx]
            X_train, y_train = X_train_val[train_idx], y_train_val[train_idx]
            
            model, result = train_test_model(model, X_train, y_train, X_val, y_val, X_test, y_test, model_name, multi_branch=multi_branch)

            if result > best_acc:
                best_acc = result
                model.save('./model/' + str(model_name) + '_best.h5')

            accs.append(result)

            # reset weights for next iteration
            K.get_session().run(global_variables_initializer())
        
        print("Average classification accuracy for %s : %f " % (model_name, mean(accs)))

    else:
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_split, random_state=42)
        model, _ = train_test_model(model, X_train, y_train, X_val, y_val, X_test, y_test, model_name, multi_branch=multi_branch)
        model.save('./model/' + str(model_name) + '_best.h5')
