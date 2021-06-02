import torch
import pandas as pd
import numpy as np
import os
from dynonet.lti import MimoLinearDynamicalOperator
from dynonet.static import MimoChannelWiseNonLinearity
import matplotlib.pyplot as plt
import time
import torch.nn as nn


if __name__ == '__main__':

    # Set seed for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)

    # Settings
    add_noise = True
    lr = 1e-3
    num_iter = 40000
    test_freq = 100
    n_batch = 1
    n_b = 2
    n_a = 2

    # Column names in the dataset
    COL_T = ['time']
    COL_X = ['V_C', 'I_L']
    COL_U = ['V_IN']
    COL_Y = ['V_C']

    # Load dataset
    df_X = pd.read_csv(os.path.join("data", "RLC_data_id_nl.csv"))
    t = np.array(df_X[COL_T], dtype=np.float32)
    #y = np.array(df_X[COL_Y], dtype=np.float32)
    x = np.array(df_X[COL_X], dtype=np.float32)
    u = np.array(df_X[COL_U], dtype=np.float32)

    # scale state
    x = x/np.array([100.0, 10.0])

    # Add measurement noise
    std_noise_V = add_noise * 0.1
    #y_nonoise = np.copy(1 + x[:, [0]] + x[:, [0]]**2)
    y_nonoise = np.copy(x[:, [0, 1]]) #np.copy(1 + x[:, [0]] ** 3)
    y_noise = y_nonoise + np.random.randn(*y_nonoise.shape) * std_noise_V

    # Prepare data
    u_torch = torch.tensor(u[None, :, :], dtype=torch.float, requires_grad=False)
    y_meas_torch = torch.tensor(y_noise[None, :, :], dtype=torch.float)
    y_true_torch = torch.tensor(y_nonoise[None, :, :], dtype=torch.float)
    G1 = MimoLinearDynamicalOperator(in_channels=1, out_channels=2, n_b=n_b, n_a=n_a, n_k=1)
    nn_static = MimoChannelWiseNonLinearity(channels=2, n_hidden=10) #StaticChannelWiseNonLin(in_channels=2, out_channels=2, n_hidden=10)
    G2 = MimoLinearDynamicalOperator(in_channels=2, out_channels=2, n_b=n_b, n_a=n_a, n_k=1)

    # Setup optimizer

    optimizer = torch.optim.Adam([
        {'params': G1.parameters(),    'lr': lr},
        {'params': nn_static.parameters(), 'lr': lr},
        {'params': G2.parameters(), 'lr': lr},
    ], lr=lr)


    # In[Train]
    LOSS = []
    start_time = time.time()
    for itr in range(0, num_iter):

        optimizer.zero_grad()

        # Simulate
        y_lin = G1(u_torch)
        y_nl = nn_static(y_lin)
        #y_hat = G2(y_nl)
        y_hat = y_nl

        # Compute fit loss
        err_fit = y_meas_torch - y_hat
        loss_fit = torch.mean(err_fit**2)
        loss = loss_fit

        LOSS.append(loss.item())
        if itr % test_freq == 0:
            print(f'Iter {itr} | Fit Loss {loss_fit:.4f}')

        # Optimize
        loss.backward()
        optimizer.step()

    train_time = time.time() - start_time
    print(f"\nTrain time: {train_time:.2f}") # 182 seconds

    # In[Detach]
    y_hat = y_hat.detach().numpy()[0, :, :]

    # In[Plot]
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(t, y_nonoise[:, 0], 'k', label="$y$")
    ax[0].plot(t, y_noise[:, 0], 'r', label="$y_{noise}$")
    ax[0].plot(t, y_hat[:, 0], 'b', label="$\hat y$")
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(t, y_nonoise[:, 1], 'k', label="$y$")
    ax[1].plot(t, y_noise[:, 1], 'r', label="$y_{noise}$")
    ax[1].plot(t, y_hat[:, 1], 'b', label="$\hat y$")
    ax[1].legend()
    ax[1].grid()

    plt.figure()
    plt.plot(LOSS)
    plt.grid(True)

