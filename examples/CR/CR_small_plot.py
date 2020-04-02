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

    # Load dataset
    mat_data = sp.io.loadmat(os.path.join("data", "Benchmark_EEG_small", "Benchmark_EEG_small.mat"))

    u = mat_data['data'][0][0][0]    # input, normalized handle angle. Tensor structure: (P, R, S)
    y = mat_data['data'][0][0][1]    # output, ICA component with highest SNR (normalized)
    msg = mat_data['data'][0][0][1]  # msg

    B = u.shape[0]  # number of participants (10)
    R = u.shape[1]  # number of realizations (7)
    N = u.shape[2]  # number of time samples (256)

    fs = 256.0  # Sampling frequency (Hz)
    ts = 1/fs
    time = np.arange(N)*ts

    # One sample
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(time, y[0, 0, :])
    ax[0].set_ylabel("Output (activity)")
    ax[0].grid(True)

    ax[1].plot(time, u[0, 0, :])
    ax[1].grid(True)
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Input (angle)")

    # All curves
    fig, ax = plt.subplots(7, 2, sharex=True)
    for idx_b in range(B):
        for idx_r in range(R):
            ax[idx_r, 0].plot(time, u[idx_b, idx_r, :])
            ax[idx_r, 1].plot(time, y[idx_b, idx_r, :])
#            ax[0].set_ylabel("Output (activity)")

#            ax[1].plot(time, U[0, 0, :])
#            ax[1].grid(True)
#            ax[1].set_xlabel("Time (s)")
#            ax[1].set_ylabel("Input (angle)")

    # Save in a more convenient hdf format
    filename = os.path.join("data", "small", "Benchmark_EEG_medium.h5")
    hf = h5py.File(filename, 'w')
    hf.create_dataset('U', data=u)
    hf.create_dataset('Y', data=y)
    hf.close()

    U = np.fft.fft(u, axis=2)
    Y = np.fft.fft(y, axis=2)
    freq_full = np.fft.fftfreq(N)

    freq = freq_full[0:N//2]
    freq_int = np.arange(N//2)
    U_abs = np.abs(U[:, :, 0:N//2])
    Y_abs = np.abs(Y[:, :, 0:N//2])

    fig, ax = plt.subplots(7, 2, sharex=True)
    for idx_b in range(B):
        for idx_r in range(R):
            ax[idx_r, 0].plot(freq_int, U_abs[idx_b, idx_r, :])
            ax[idx_r, 1].plot(freq_int, Y_abs[idx_b, idx_r, :])

