# Implementation code for the paper "Fusion Convolutional Neural Network for Cross-Subject EEG Motor Imagery Classification"

## Dependencies Required
* Python 3.7
* Tensorflow 2.1.0
* SciKit-learn 0.22.1
* Gumpy (https://github.com/gumpy-bci/gumpy)
* SciPy 1.4.1
* Numpy 1.18.1
* mlxtend 0.17.2
* statsmodels 0.11.1
* pyEDFlib 0.1.17

If you have python, virtualenv and pip installed, you can use the "install.bat" script in Windows or "install.sh" script in Linux to create a virtual environment in the running folder with all the dependencies installed.

## Running
The program can be run from the CLI with the following required arguments:

1.) The number of subjects to be used from the dataset (integer)

2.) The number of epochs the training of models should be done (integer)

3.) The number of target classes in the classification (integer)

4.) What type of trials should be extracted from the data (1 or 2, where 1 => executed trials only and 2 => imagined trials only)

5.) If CPU-only mode should be used (True / False). Note that for GPU mode you will need to have CUDA installed.

Example: python run_experiments.py 109 100 2 1 True

## License
Copyright Karel Roots 2020

This work is licensed under the Apache License, Version 2.0. See [LICENSE](https://github.com/rootskar/EEGMotorImagery/edit/master/LICENSE) for the full license text.
