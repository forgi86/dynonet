import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import scipy as sp
import scipy.io
import h5py

if __name__ == '__main__':

    # Set seed for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)

    DATA_FOLDER = os.path.join("data", "Benchmark_EEG_medium")

    # Load dataset
    mat_data = sp.io.loadmat(os.path.join(DATA_FOLDER, "Benchmark_EEG_medium.mat"))

    data = mat_data['EEGdata']
    N = mat_data['EEGdata'].size  # participants
    # R: number of multisine realizations (7)
    # P: number of recorded periods per realization (210)
    # S: number of samples per period (2048(

    R, P, S = mat_data['EEGdata'][0][0][0][0][0].shape  # realizations, periods, samples

    U = np.zeros((N, R, P, S))
    Y = np.zeros((N, R, P, S))

    for idx in range(N):
        U[idx, :] = mat_data['EEGdata'][idx][0][0][0][0]
        Y[idx, :] = mat_data['EEGdata'][idx][0][0][0][1]

    filename = os.path.join(DATA_FOLDER, "Benchmark_EEG_medium.h5")
    hf = h5py.File(filename, 'w')
    hf.create_dataset('U', data=U)
    hf.create_dataset('Y', data=Y)
    hf.close()

    time = np.arange(S)
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(time, Y[0, 0, 0, :])
    ax[0].set_ylabel("Output (?)")
    ax[0].grid(True)

    ax[1].plot(time, U[0, 0, 0, :])
    ax[1].grid(True)
    ax[1].set_xlabel("Time (?)")
    ax[1].set_ylabel("Input (?)")
