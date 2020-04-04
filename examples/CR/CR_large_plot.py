import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import scipy as sp
import scipy.io
import h5py

if __name__ == '__main__':

    DATA_FOLDER = os.path.join("data", "3-4TU-large")


    # In[Panic]
    task_num = 1
    subject_num = 2
    filename = f'dataEEG_T{task_num}_{subject_num}.mat'
    h5_data = h5py.File(os.path.join(DATA_FOLDER, filename), 'r')
    labels = pd.read_csv(os.path.join(DATA_FOLDER, 'labels.csv'), header=None)  # Channel labels extracted manually from Matlab (hard to do it in Python)
    labels.columns = ['Channel name']

    # In[Load big data]

    fs = np.array(h5_data['dataEEG']['sampleRate'])
    #data = np.array(h5_data['dataEEG']['data'], dtype=np.float32).transpose()
    #harm = np.array(h5_data['dataEEG']['ExcitedHarm'])
    #R, P, T, C = data.shape  # Multisine Realization (7), Period (210), Time (2048), Channel (132)
    # In[]

    #h5_data = sp.io.loadmat(os.path.join(DATA_FOLDER, 'ica_mat.mat'), 'r')
    h5_data = h5py.File(os.path.join(DATA_FOLDER, 'ica_mat.mat'), 'r')

