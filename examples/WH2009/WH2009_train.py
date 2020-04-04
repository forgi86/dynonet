import torch
import pandas as pd
import numpy as np
import os
from torchid.module.LTI import LinearSiso
from torchid.module.static import StaticSisoNonLin
import matplotlib.pyplot as plt
import time
import torch.nn as nn

import util.metrics


if __name__ == '__main__':

    # Set seed for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)

    # Settings
    lr = 2e-4
    num_iter = 40000
    test_freq = 100
    n_fit = 100000 #100000
    decimate = 1
    n_batch = 1
    n_b = 3
    n_a = 3

    # Column names in the dataset
    COL_F = ['fs']
    COL_U = ['uBenchMark']
    COL_Y = ['yBenchMark']

    # Load dataset
    df_X = pd.read_csv(os.path.join("data", "WienerHammerBenchmark.csv"))

    # Extract data
    y = np.array(df_X[COL_Y], dtype=np.float32)[None, :]  # batch, time, channel
    u = np.array(df_X[COL_U], dtype=np.float32)[None, :]
    fs = np.array(df_X[COL_F].iloc[0], dtype = np.float32)
    N = y.size
    ts = 1/fs
    t = np.arange(N)*ts

    # Fit data
    y_fit = y[:n_fit:decimate]
    u_fit = u[:n_fit:decimate]
    t_fit = t[0:n_fit:decimate]


    # Prepare data
    u_fit_torch = torch.tensor(u_fit, dtype=torch.float, requires_grad=False)
    y_fit_torch = torch.tensor(y_fit, dtype=torch.float)


    # Second-order dynamical system custom defined
    G1 = LinearSiso(n_b, n_a)
    G2 = LinearSiso(n_b, n_a)

    # Static sandwitched non-linearity
    F_nl = StaticSisoNonLin()

    # Setup optimizer
    optimizer = torch.optim.Adam([
        {'params': G1.parameters(), 'lr': lr},
        {'params': G2.parameters(), 'lr': lr},
        {'params': F_nl.parameters(), 'lr': lr},
    ], lr=lr)

    # In[Train]
    LOSS = []
    start_time = time.time()
    for itr in range(0, num_iter):

        optimizer.zero_grad()

        # Simulate
        y1_lin = G1(u_fit_torch)
        y1_nl = F_nl(y1_lin)
        y_hat = G2(y1_nl)

        # Compute fit loss
        err_fit = y_fit_torch - y_hat
        loss_fit = torch.mean(err_fit**2)
        loss = loss_fit

        LOSS.append(loss.item())
        if itr % test_freq == 0:
            with torch.no_grad():
                RMSE = torch.sqrt(loss)
            print(f'Iter {itr} | Fit Loss {loss_fit:.6f} | RMSE:{RMSE:.4f}')

        # Optimize
        loss.backward()
        optimizer.step()

    train_time = time.time() - start_time
    print(f"\nTrain time: {train_time:.2f}")

    # In[Save model]
    model_name = "model_WH"
    model_folder = os.path.join("models", model_name)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    torch.save(G1.state_dict(), os.path.join(model_folder, "G1.pkl"))
    torch.save(F_nl.state_dict(), os.path.join(model_folder, "F_nl.pkl"))
    torch.save(G2.state_dict(), os.path.join(model_folder, "G2.pkl"))


    # In[To numpy]

    y_hat = y_hat.detach().numpy()[0, :, 0]
    y1_lin = y1_lin.detach().numpy()[0, :, 0]
    y1_nl = y1_nl.detach().numpy()[0, :, 0]

    # In[Plot]
    plt.figure()
    plt.plot(t_fit, y_fit, 'k', label="$y$")
    plt.plot(t_fit, y_hat, 'b', label="$\hat y$")
    plt.legend()

    plt.figure()
    plt.plot(LOSS)
    plt.grid(True)

    # In[Plot static non-linearity]

    y1_lin_min = np.min(y1_lin)
    y1_lin_max = np.max(y1_lin)

    in_nl = np.arange(y1_lin_min, y1_lin_max, (y1_lin_max- y1_lin_min)/1000).astype(np.float32).reshape(-1, 1)

    with torch.no_grad():
        out_nl = F_nl(torch.as_tensor(in_nl))

    plt.figure()
    plt.plot(in_nl, out_nl, 'b')
    plt.plot(in_nl, out_nl, 'b')
    #plt.plot(y1_lin, y1_nl, 'b*')
    plt.xlabel('Static non-linearity input (-)')
    plt.ylabel('Static non-linearity input (-)')
    plt.grid(True)

    # In[Plot]
    e_rms = util.metrics.error_rmse(y_hat, y_fit)[0]
    print(f"RMSE: {e_rms:.2f}")