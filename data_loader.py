#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import sys
import traceback
from glob import glob
import pyedflib

def load_data(nr_of_subj = 109, nr_of_epochs = 25, trial_type = 1, crop_seconds = 0.5, base_folder = 'files/'):
    # %%
    # Get file paths
    PATH = 'files/'
    SUBS = glob(PATH + 'S[0-9]*')
    FNAMES = sorted([x[-4:] for x in SUBS])
    FNAMES = FNAMES[:nr_of_subj]

    try:
        FNAMES.remove('S038', 'S088', 'S089', 'S092', 'S100', 'S104')
    except:
        pass

    print("Using files:")
    print(FNAMES)

    # %%
    # load files

    samples = 640
    sample_rate = 160

    def convertLabelToInt(str):
        if str == 'T1':
            return 0
        if str == 'T2':
            return 1
        return -1


    def divide_chunks(l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]

    executed_trials = '03,07,11'.split(',')
    imagined_trials = '04,08,12'.split(',')
    both_trials = executed_trials + imagined_trials

    if trial_type == 1:
        file_numbers = executed_trials
    elif trial_type == 2:
        file_numbers = imagined_trials
    elif trial_type == 2:
        file_numbers = both_trials
    else:
        raise Exception("Invalid trial type value")

    X = []
    y = []

    for subj in FNAMES:
        
        fnames = glob(os.path.join(PATH, subj, subj+'R*.edf'))
        fnames = [name for name in fnames if name[-6:-4] in file_numbers]

        for file_name in fnames:

            print("File name " + file_name)
            loaded_file = pyedflib.EdfReader(file_name)

            annotations = loaded_file.readAnnotations()
            times = annotations[0]
            durations = annotations[1]
            tasks = annotations[2]

            signals = loaded_file.signals_in_file
            signal_labels = loaded_file.getSignalLabels()
            sigbufs = np.zeros((signals, loaded_file.getNSamples()[0]))

            for i in np.arange(signals):
                sigbufs[i, :] = loaded_file.readSignal(i)

            trial_data = np.zeros((30, 64, int(samples)))
            labels = np.zeros(30)

            signal_start = 0

            for i in range(len(times)):

                current_duration = durations[i]
                signal_end = signal_start + samples

                # skipping rest states
                if tasks[i] == 'T0': 
                    signal_start += int(sample_rate*current_duration)
                    continue

                for j in range(len(sigbufs)):
                    trial_data[i][j] = sigbufs[j][signal_start:signal_end]

                signal_start += int(sample_rate*current_duration)
                
                labels[i] = convertLabelToInt(tasks[i])

            y += labels
            X += trial_data
        
    X = np.stack(X)
    y = np.array(y).reshape((-1,1))

    print("Loaded shapes:")
    print(X.shape, y.shape)

    return X, y