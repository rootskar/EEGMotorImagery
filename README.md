# Fusion Convolutional Neural Network for Cross-Subject EEG Motor Imagery Classification

Implementation code for the paper:

Karel Roots, Yar Muhammad and Naveed Muhammad, “Fusion Convolutional Neural Network for Cross-Subject EEG Motor Imagery Classification”, In Journal of Computers 2020, 9 (3), 72; Machine Learning for EEG Signal Processing, September 5, 2020.

https://www.mdpi.com/2073-431X/9/3/72

## Abstract
Brain–computer interfaces (BCIs) can help people with limited motor abilities to interact with their environment without external assistance. A major challenge in electroencephalogram (EEG)-based BCI development and research is the cross-subject classification of motor imagery data. Due to the highly individualized nature of EEG signals, it has been difficult to develop a cross-subject classification method that achieves sufficiently high accuracy when predicting the subject’s intention. In this study, we propose a multi-branch 2D convolutional neural network (CNN) that utilizes different hyperparameter values for each branch and is more flexible to data from different subjects. Our model, EEGNet Fusion, achieves 84.1% and 83.8% accuracy when tested on the 103-subject eegmmidb dataset for executed and imagined motor actions, respectively. The model achieved statistically significantly higher results compared with three state-of-the-art CNN classifiers: EEGNet, ShallowConvNet, and DeepConvNet. However, the computational cost of the proposed model is up to four times higher than the model with the lowest computational cost used for comparison.

## Dependencies Required
* Python 3.7
* Tensorflow 2.1.0
* Pywavelets 1.1.1
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

NB! The EEG data has to be unpacked into the working directory "data" folder.

## License
Copyright Karel Roots 2020

This work is licensed under the Apache License, Version 2.0. See [LICENSE](https://github.com/rootskar/EEGMotorImagery/edit/master/LICENSE) for the full license text.
