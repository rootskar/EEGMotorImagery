# Implementation code for the paper "Fusion Convolutional Neural Network with Transfer Learning for Cross-Subject EEG Motor Imagery Classification"

## Dependencies
Python >= 3.3, Tensorflow >= 1.4, Numpy >= 1.18.1, scikit-learn >= 0.22, pyEDFlib >= 0.1.15, Gumpy (https://github.com/gumpy-bci/gumpy), mlxtend >= 0.17

## Running
The program can be run from the CLI with the following required arguments:

1.) The number of subjects to be used from the dataset

2.) The number of epochs the training of models should be done

3.) What type of trials should be extracted from the data; 1 => executed trials only; 2 => imagined trials only; 3 => both trials

4.) If the training/testing is done using CPU mode (True/False). If this is False, the machine should be able to use tensorflow with GPU

Example: python train_test_tl.py 109 100 1 False

## License
Copyright Karel Roots 2020

This work is licensed under the Apache License, Version 2.0. See [LICENSE](https://github.com/rootskar/EEGMotorImagery/edit/master/LICENSE) for the full license text.
