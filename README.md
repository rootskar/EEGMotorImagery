# EEGMotorImagery
EEG motor imagery classification using transfer learning

## Dependencies
Python >= 3.3, Tensorflow >= 1.4, Numpy >= 1.18.1, scikit-learn >= 0.22, pyEDFlib >= 0.1.15, Gumpy (https://github.com/gumpy-bci/gumpy)

## Running
The program can be run from the CLI with the following required arguments:
1.) The numbr of subjects to be used from the dataset
2.) The number of epochs the training of models should be done
3.) What type of trials should be extracted from the data; 1 => executed trials only; 2 => imagined trials only; 3 => both trials
4.) If CPU-only mode should be used (True / False)

Example: python train_test_tl.py 109 100 1 False
