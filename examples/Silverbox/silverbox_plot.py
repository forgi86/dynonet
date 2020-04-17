import torch
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


if __name__ == '__main__':

    # Set seed for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)

    # Settings
    add_noise = True
    lr = 2e-4
    num_iter = 1
    test_freq = 100
    n_fit = 100000
    decimate = 1
    n_batch = 1
    n_b = 3
    n_f = 3

    # Column names in the dataset
    COL_U = ['V1']
    COL_Y = ['V2']

    # Load dataset
    df_X = pd.read_csv(os.path.join("data", "SNLS80mV.csv"))

    # Extract data
    y = np.array(df_X[COL_Y], dtype=np.float32)
    u = np.array(df_X[COL_U], dtype=np.float32)
    u = u-np.mean(u)
    fs = 10**7/2**14
    N = y.size
    ts = 1/fs
    t = np.arange(N)*ts

    # Fit data
    y_fit = y[:n_fit:decimate]
    u_fit = u[:n_fit:decimate]
    t_fit = t[0:n_fit:decimate]

    # In[Plot]
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(t_fit, y_fit, 'k', label="$u")
    ax[0].grid()
    ax[1].plot(t_fit, u_fit, 'k', label="$y$")
    ax[1].grid()
    plt.legend()





