#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import os
import sys
import traceback
from glob import glob
import pyedflib
from preprocess import preprocess_data


"""
@input - Optional args - the number of subjects to use from the dataset (int); the trial type (int 1, 2 or 3);
         the number of chunks the samples should be divided into (int); the relative path to data files location (String)
         
Method that loads the EEG Motor Imagery data from the input folder with given number of subjects, 
specified trial type (Executed movement (type 1), imagined movement (type 2) or both combined into one class (type 3)).
The data is also preprocessed using if the preprocessing argument is True.

@output - data (Numpy array); target labels (Numpy array)
"""
def load_data(nr_of_subj=109, trial_type=1, chunk_data=True, chunks=4, base_folder='files/', sample_rate=160, samples=640, 
              preprocessing=False, hp_freq=0.5, bp_low=2, bp_high=60, notch=False, hp_filter=False, bp_filter=False, artifact_removal=False, normalize=False):
    # %%
    # Get file paths
    PATH = base_folder
    SUBS = glob(PATH + 'S[0-9]*')
    FNAMES = sorted([x[-4:] for x in SUBS])
    FNAMES = FNAMES[:nr_of_subj]

    # Remove the subjects with incorrectly annotated data that will be omitted from the final dataset
    subjects = ['S038', 'S088', 'S089', 'S092', 'S100', 'S104']
    try:
        for sub in subjects:
            FNAMES.remove(sub)
    except:
        pass

    #print("Using files:")
    #print(FNAMES)
    
    """
    @input - label (String)
            
    Helper method that converts trial labels into integer representations

    @output - data (Numpy array); target labels (Numpy array)
    """
    def convertLabelToInt(str):
        if str == 'T1':
            return 0
        if str == 'T2':
            return 1
        raise Exception ("Invalid label %s" % str)


    """
    @input - data (array); number of chunks to divide the list into (int)
            
    Helper method that divides the input list into a given number of arrays

    @output - 2D array of divided input data
    """
    def divide_chunks(data, chunks):
        for i in range(0, len(data), chunks):
            yield data[i:i + chunks]


    executed_trials = '03,07,11'.split(',')
    imagined_trials = '04,08,12'.split(',')
    both_trials = executed_trials + imagined_trials
    samples_per_chunk = int(samples/chunks)

    # Determine the type of trials to be used
    if trial_type == 1:
        file_numbers = executed_trials
    elif trial_type == 2:
        file_numbers = imagined_trials
    elif trial_type == 3:
        file_numbers = both_trials
    else:
        raise Exception("Invalid trial type value %d" % trial_type)

    X = []
    y = []

    # Iterate over different subjects
    for subj in FNAMES:
        
        # Load the file names for given subject
        fnames = glob(os.path.join(PATH, subj, subj+'R*.edf'))
        fnames = [name for name in fnames if name[-6:-4] in file_numbers]

        # Iterate over the trials for each subject
        for file_name in fnames:

            # Load the file
            #print("File name " + file_name)
            loaded_file = pyedflib.EdfReader(file_name)
            annotations = loaded_file.readAnnotations()
            times = annotations[0]
            durations = annotations[1]
            tasks = annotations[2]

            # Load the data signals into a buffer
            signals = loaded_file.signals_in_file
            #signal_labels = loaded_file.getSignalLabels()
            sigbufs = np.zeros((signals, loaded_file.getNSamples()[0]))
            for i in np.arange(signals):
                sigbufs[i, :] = loaded_file.readSignal(i)

            # initialize the result arrays with preferred shapes
            if chunk_data:
                trial_data = np.zeros((15, 64, chunks, samples_per_chunk))
            else:
                trial_data = np.zeros((15, 64, samples))
            labels = []

            signal_start = 0
            k = 0

            # Iterate over tasks in the trial run
            for i in range(len(times)):
                # Collects only the 15 non-rest tasks in each run
                if k == 15:
                    break

                current_duration = durations[i]
                signal_end = signal_start + samples

                # Skipping tasks where the user was resting
                if tasks[i] == 'T0': 
                    signal_start += int(sample_rate*current_duration)
                    continue

                # Iterate over each channel
                for j in range(len(sigbufs)):
                    channel_data = sigbufs[j][signal_start:signal_end]
                    if preprocessing:
                        channel_data = preprocess_data(channel_data, sample_rate=sample_rate, ac_freq=60, 
                                                       hp_freq=hp_freq, bp_low=bp_low, bp_high=bp_high, notch=notch, 
                                                       hp_filter=hp_filter, bp_filter=bp_filter, artifact_removal=artifact_removal)
                    if chunk_data:
                        channel_data = list(divide_chunks(channel_data, samples_per_chunk))

                    # Add data for the current channel and task to the result
                    trial_data[k][j] = channel_data
                
            
                # add label(s) for the current task to the result
                if chunk_data:
                    # multiply the labels by the chunk size for chunked mode
                    labels.extend([convertLabelToInt(tasks[i])]*chunks)
                else:
                    labels.append(convertLabelToInt(tasks[i]))

                signal_start += int(sample_rate*current_duration)
                k += 1

            # Add labels and data for the current run into the final output numpy arrays
            y.extend(labels)
            X.extend(trial_data.swapaxes(1, 2).reshape((-1, 64, samples_per_chunk)))
    
    # Shape the final output arrays to the correct format
    X = np.stack(X)
    y = np.array(y).reshape((-1,1))

    print("Loaded data shapes:")
    print(X.shape, y.shape)

    return X, y