# Implementation code for the paper "Fusion Convolutional Neural Network with Transfer Learning for Cross-Subject EEG Motor Imagery Classification"

## Dependencies
Python >= 3.3, Tensorflow >= 1.4, Numpy >= 1.18.1, scikit-learn >= 0.22, pyEDFlib >= 0.1.15, Gumpy (https://github.com/gumpy-bci/gumpy)

## Running
The program can be run from the CLI with the following required arguments:

1.) The number of subjects to be used from the dataset

2.) The number of epochs the training of models should be done

3.) What type of trials should be extracted from the data; 1 => executed trials only; 2 => imagined trials only; 3 => both trials

4.) If the training/testing is done using CPU mode (True/False). If this is False, the machine should be able to use tensorflow with GPU

Example: python train_test_tl.py 109 100 1 False

## License

 Copyright [2020] [Karel Roots]

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
