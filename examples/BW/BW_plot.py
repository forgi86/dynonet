import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py

if __name__ == '__main__':

    # In[Load dataset]

    h5_filename = 'train.h5'
    #h5_filename = 'test.h5'

    signal_name = 'multisine'
    #signal_name = 'sinesweep' # available in test

    h5_data = h5py.File(os.path.join("data", "Test signals", h5_filename), 'r')
    dataset_list = h5_data.keys()
    y = np.array(h5_data[signal_name]['y']).transpose()  # MATLAB saves data in column major order...
    if y.ndim == 2:
        y = y[..., None]
    u = np.array(h5_data[signal_name]['u']).transpose()
    if u.ndim == 2:
        u = u[..., None]

    fs = np.array(h5_data[signal_name]['fs']).item()

    N = y.shape[1]
    ts = 1.0/fs
    t = np.arange(N)*fs

    # In[Plot dataset]
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(y[0, :, 0])
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Displacement (mm)')
    ax[0].grid(True)
    ax[1].plot(y[0, :, 0])
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('Force (N)')
    ax[1].grid(True)
    plt.show()
