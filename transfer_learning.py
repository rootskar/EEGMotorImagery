import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import callbacks
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from tensorflow import Session, global_variables_initializer
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from statistics import mean

from training_testing import run_model
from predict import predict_accuracy


"""
@input - training data (List); training labels (List); testing data (List); testing labels (List); model (Object);
         model name (String); index of subject (int); layer indexes ranges to disable for TL (tuple) ; optional args
         
Method that loads the best weights for the input model name (if it has been saved before with pretraining),
reshapes the arrays into correct input shapes for the model, converts labels to categorical,
trains the model with transfer learning approach by disabling specified layers and evaluates the model accuracy on test data.

@output - Accuracy value (float)
"""
def train_evaluate_transfer(X_train, y_train, X_test, y_test, model, model_name, subj, disabled_layers, 
                            multi_branch=False, nr_of_epochs=100, val_split=0.1, samples=80, classes=2, n=0, kfold_n=1, train_size=0.75):
    model.load_weights('./model/' + str(model_name) + '_best.h5')
    X_train = X_train.reshape((-1, 1, 64, samples))
    X_test = X_test.reshape((-1, 1, 64, samples))
    y_train = to_categorical(y_train, classes)
    y_test = to_categorical(y_test, classes)

    model_name = './model/' + str(model_name) + '_TL_subj_' + str(subj) + '.h5'
    print("New model name: " + model_name)

    # Disable layers for TL
    for r in disabled_layers:
        for layers in (model.layers)[r[0]:r[1]]:
            layers.trainable = False

    # Callbacks for saving the best model with lowest validation loss and reducing learning rate when plateau is reached
    callbacks_list = [callbacks.ModelCheckpoint(model_name, save_best_only=True, monitor='val_loss'),
                    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)]

    model.compile(loss=binary_crossentropy, optimizer=Adam(lr=0.001), metrics=['accuracy'])
    if multi_branch:
        history = model.fit([X_train, X_train, X_train], y_train, shuffle=True, batch_size=64, 
                            epochs=nr_of_epochs, validation_split=val_split, verbose=False, callbacks=callbacks_list)
    else:
        history = model.fit(X_train, y_train, shuffle=True, batch_size=64, epochs=nr_of_epochs, validation_split=val_split, verbose=False, callbacks=callbacks_list)

    # %%
    # Test the model
    model.load_weights(model_name)
    return predict_accuracy(model, X_test, y_test, model_name, multi_branch=multi_branch, tl=True, n=n, subj=subj, kfold_n=kfold_n, train_size=train_size)


"""
@input - data (List); target labels (List); model name (String); index of subject (int); layer indexes ranges to disable for TL (tuple) ; optional args
         
Method that loads the best weights for the input model name (if it has been saved before with pretraining),
splits the input data into training and testing sets and evaluates the model.
If the kfold argument is True, the model is kfold cross-validated and the average cross-validation accuracy is printed.

@output - Model object
"""
def run_transfer(X, y, model_name, subj, disabled_layers, nr_of_epochs=100, multi_branch=False, test_different_training_size=False,
                 val_split=0.1, test_size=0.25, samples=80, classes=2, use_kfold=False, kfold_n=2):
    model = load_model('./model/' + str(model_name) + '_best.h5')

    if use_kfold:
        kfold = KFold(kfold_n, True, 42)
        accs = []
        for train_idx, test_idx in kfold.split(X):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            train_size = int((X_train.shape[0]/X.shape[0])*100)

            acc = train_evaluate_transfer(X_train, y_train, X_test, y_test, model, model_name, subj, disabled_layers,
                                        multi_branch=multi_branch, nr_of_epochs=nr_of_epochs, samples=samples, classes=classes, n=len(accs), kfold_n=kfold_n, train_size=train_size)
            accs.append(acc)
            # reload model weights
            model.load_weights('./model/' + str(model_name) + '_best.h5')

            # switches the training and testing set
            if test_different_training_size and test_size != 2:
                acc = train_evaluate_transfer(X_test, y_test, X_train, y_train, model, model_name, subj, disabled_layers,
                                            multi_branch=multi_branch, nr_of_epochs=nr_of_epochs, samples=samples, classes=classes, n=len(accs), kfold_n=kfold_n, train_size=train_size)
                accs.append(acc)
                # reload model weights
                model.load_weights('./model/' + str(model_name) + '_best.h5')

        print("Average TL classification accuracy for %s : %f " % (model_name, mean(accs)))

    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        train_size = int((X_train.shape[0]/X.shape[0])*100)
        train_evaluate_transfer(X_train, y_train, X_test, y_test, model, model_name, subj, disabled_layers,
                                multi_branch=multi_branch, nr_of_epochs=nr_of_epochs, samples=samples, classes=classes, train_size=train_size)

    return model


"""
@input - model name (String); model (Object); data (List); target labels (List); TL data (List); TL labels (List); 
         nr of subjects used for training (int); layer indexes ranges to disable for TL (tuple) ; optional args
         
Method that runs baseline training/testing and transfer learning over the number of testing subjects.
If different_test_sizes is True, then the TL is evaluated using 25%, 50% and 75% training data.

@output - None
"""
def run_transfer_learning(model_name, model, X, y, X_tl, y_tl, subj_for_training, disabled_layers, 
                          multi_branch=False, test_size=0.25, different_test_sizes=False, use_kfold=False, kfold_n=10, pre_train=False):
    if pre_train:
        # pre-train the base model
        run_model(X, y, model, multi_branch=multi_branch, model_name = model_name, classes=2, samples=80, use_kfold=use_kfold, kfold_n=kfold_n, val_split=0.1)

    # evaluate predictions for each subject before TL
    for i in range(len(X_tl)):
        subj_nr = subj_for_training+i+1
        predict_accuracy(model, X_tl[i], y_tl[i], model_name, multi_branch=True, tl=True, n=0, subj=subj_nr, kfold_n=1, train_size=0)

    # use transfer learning for each new subject separately
    for i in range(len(X_tl)):
        subj_nr = subj_for_training+i+1
        if different_test_sizes:
            if use_kfold:
                size_list = [2, 4]
                test_diff_size_kfold = True
            else:
                size_list = [0.25, 0.5, 0.75]
                test_diff_size_kfold = False
                for size in size_list:
                    run_transfer(X_tl[i], y_tl[i], model_name, subj_nr, disabled_layers, test_different_training_size=test_diff_size_kfold, multi_branch=multi_branch, test_size=size, use_kfold=use_kfold, kfold_n=size)
        else:
            run_transfer(X_tl[i], y_tl[i], model_name, subj_nr, disabled_layers, multi_branch=multi_branch, use_kfold=use_kfold, kfold_n=kfold_n)
