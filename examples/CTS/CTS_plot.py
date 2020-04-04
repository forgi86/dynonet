import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
import os
import sys
sys.path.append(os.path.join("..", ".."))


if __name__ == '__main__':

    matplotlib.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    matplotlib.rc('text', usetex=True)

    plot_input = False

    dataset_type = 'test'
    #dataset_type = 'id'

    model_name = 'model_SS_128step'
    hidden_name = 'hidden_SS_128step'

    #model_name = 'model_SS_soft'
    #hidden_name = 'hidden_SS_soft'

    # Load dataset
    df_data = pd.read_csv(os.path.join("data", "dataBenchmark.csv"))
    if dataset_type == 'id':
        u = np.array(df_data[['uEst']]).astype(np.float32)
        y = np.array(df_data[['yEst']]).astype(np.float32)
    elif dataset_type == 'test':
        u = np.array(df_data[['uVal']]).astype(np.float32)
        y = np.array(df_data[['yVal']]).astype(np.float32)

    ts_meas = df_data['Ts'][0].astype(np.float32)
    time_exp = np.arange(y.size).astype(np.float32) * ts_meas


    # In[Plot results]
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(9, 5.5))

    ax[0].plot(time_exp, y, 'k', label='$y$')
    ax[0].legend(loc='upper right')
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Voltage (V)")
    ax[0].set_ylim([1.0, 12.0])
    ax[0].set_xlim([0, 1024*4 + 500])
    ax[0].grid(True)


    ax[1].plot(time_exp, u, 'k', label='$u$')
    ax[1].legend(loc='upper right')
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Voltage (V)")
    ax[1].grid(True)
