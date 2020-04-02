import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':


    # Column names in the dataset
    COL_F = 'fs'
    TAG_R = 'r'
    TAG_U = 'u'
    TAG_Y = 'y'

    COL_U = 'u1'
    COL_Y = 'y1'
    COL_R = 'r1'

    # Load dataset

    df_names = ['WH_CombinedZeroMultisineSinesweep', 'WH_MultisineFadeOut', 'WH_SineInput_meas', 'WH_Triangle2_meas', 'WH_ZeroMeas'] + ["WH_TestDataset"]
    df_list = []
    for dataset_name in df_names:
        dataset_filename = dataset_name + '.csv'
        df_dataset = pd.read_csv(os.path.join("data", dataset_filename))
        idx_u = list(df_dataset.keys()).index('u')
        n_real = idx_u
        col_names = [TAG_R + str(i) for i in range(n_real)] + [TAG_U + str(i) for i in range(n_real)] + [TAG_Y + str(i) for i in range(n_real)] + [COL_F] + ['?']
        df_dataset.columns = col_names
        df_list.append(df_dataset)
    dict_df = dict(zip(df_names, df_list))

    df_X = dict_df['WH_Triangle2_meas']
    y_meas = np.array(df_X[[COL_Y]], dtype=np.float32)
    u = np.array(df_X[[COL_U]], dtype=np.float32)
    r = np.array(df_X[[COL_U]], dtype=np.float32)
    fs = np.array(df_X[COL_F].iloc[0], dtype = np.float32)
    N = y_meas.size
    ts = 1/fs
    t = np.arange(N)*ts


    # In[Plot]
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(t, y_meas, 'k', label="$y$")
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(t, u, 'k', label="$u$")
    ax[1].plot(t, r, 'r', label="$r$")
    ax[1].legend()
    ax[1].grid()

