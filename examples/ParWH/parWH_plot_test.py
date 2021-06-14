import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':

    N = 16384  # number of samples per period
    #M = 20  # number of random phase multisine realizations
    P = 2  # number of periods
    nAmp = 5 #  number of different amplitudes

    # Column names in the dataset
    COL_F = ['fs']
    TAG_U = 'u'
    TAG_Y = 'y'

    # Load dataset

    #df_X = pd.read_csv(os.path.join("data", "ParWHData_Validation_Level1.csv"))
    df_X = pd.read_csv(os.path.join("data", "ParWHData_ValidationArrow.csv"))

    #df_X.columns = ['amplitude', 'fs', 'lines'] + [TAG_U + str(i) for i in range(M)] + [TAG_Y + str(i) for i in range(M)] + ['?']

    # Extract data
    y = np.array(df_X['y'], dtype=np.float32)
    u = np.array(df_X['u'], dtype=np.float32)
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
    ax[1].legend()
    ax[1].grid()
    plt.show()

