import torch
import pandas as pd
import numpy as np
import os
from torchid.linearsiso import LinearDynamicalSystem
import matplotlib.pyplot as plt
import time
import torch.nn as nn

import util.metrics

if __name__ == '__main__':


    # Column names in the dataset
    COL_F = ['fs']
    COL_U = ['u']
    COL_Y = ['y']
    COL_R = ['r']

    # Load dataset
    df_X = pd.read_csv(os.path.join("data", "WH_CombinedZeroMultisineSinesweep.csv"))

    # Extract data
    y = np.array(df_X[COL_Y], dtype=np.float32)
    u = np.array(df_X[COL_U], dtype=np.float32)
    r = np.array(df_X[COL_R], dtype=np.float32)
    fs = np.array(df_X[COL_F].iloc[0], dtype = np.float32)
    N = y.size
    ts = 1/fs
    t = np.arange(N)*ts


    # In[Plot]
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(t, y, 'k', label="$y$")
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(t, u, 'k', label="$u$")
    ax[1].plot(t, r, 'r', label="$r$")
    ax[1].legend()
    ax[1].grid()

