import matplotlib
matplotlib.use("TkAgg")
import os
import pandas as pd
import numpy as np
import scipy as sp
import scipy.io
import scipy.signal
import torch
import torch.optim as optim
import time
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.join("..", ".."))


if __name__ == '__main__':

    # In[Set seed for reproducibility]
    np.random.seed(0)
    torch.manual_seed(0)

    # In[Load dataset]
    emps_data = sp.io.loadmat(os.path.join("data", "DATA_EMPS.mat"))
    q_ref = emps_data['qg'].astype(np.float32)
    q_meas = emps_data['qm'].astype(np.float32)
    u_in = emps_data['vir'].astype(np.float32)
    time_exp = emps_data['t'].astype(np.float32)
#    d_N = emps_data['pulses_N']
    ts = np.mean(np.diff(time_exp.ravel()))#time_exp[1] - time_exp[0]

    # Design a differentiator filter to estimate unmeasured velocities from noisy, measured positions
    fs = 1/ts       # Sample rate, Hz
    cutoff = 10.0    # Desired cutoff frequency, Hz
    trans_width = 100  # Width of transition from pass band to stop band, Hz
    n_taps = 32      # Size of the FIR filter.
    taps = scipy.signal.remez(n_taps, [0, cutoff, cutoff + trans_width, 0.5 * fs], [2 * np.pi * 2 * np.pi * 10 * 1.5, 0], Hz=fs, type='differentiator')

    # Filter positions to estimate velocities
    x_est = np.zeros((q_ref.shape[0], 2), dtype=np.float32)
    x_est[:, 0] = q_meas[:, 0]
    v_est = np.convolve(x_est[:, 0], taps, 'same')  # signal.lfilter(taps, 1, y_meas[:,0])*2*np.pi
    x_est[:, 1] = np.copy(v_est)
    x_est[0:n_taps, [1]] = x_est[n_taps + 1, [1]]
    x_est[-n_taps:, [1]] = x_est[-n_taps - 1, [1]]


    # Simulation plot
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(6, 7.5))
    ax[0].plot(time_exp, q_ref,  'k',  label='$q_{\mathrm{ref}}$')
    ax[0].plot(time_exp, q_meas, 'k', label='$q_{\mathrm{meas}}$')
    ax[0].legend(loc='upper right')
    ax[0].grid(True)
    ax[0].set_ylabel("Position (m)")

    ax[1].plot(time_exp, x_est[:, 1],  'k--',  label='$v_{\mathrm{est}}$')
    ax[1].grid(True)
    ax[1].set_ylabel("Velocity (m/s)")

    ax[2].plot(time_exp, u_in, 'k*', label='$u_{in}$')
    ax[2].set_xlabel("Time (s)")
    ax[2].set_ylabel("Input (V)")
    ax[2].grid(True)
    ax[2].set_xlabel("Time (s)")
    plt.show()
