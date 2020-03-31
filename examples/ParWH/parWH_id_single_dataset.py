import pandas as pd
import numpy as np
import os
import torch
from torchid.module.LTI import LinearMimo
import torch.nn as nn

import matplotlib.pyplot as plt
import time
import util.metrics

class StaticNonLin(nn.Module):

    def __init__(self):
        super(StaticNonLin, self).__init__()

        self.net_1 = nn.Sequential(
            nn.Linear(1, 10),  # 2 states, 1 input
            nn.ReLU(),
            nn.Linear(10, 1)
        )

        self.net_2 = nn.Sequential(
            nn.Linear(1, 10),  # 2 states, 1 input
            nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, u_lin):

        y_nl_1 =   self.net_1(u_lin[..., [0]])  # Process blocks individually
        y_nl_2 =   self.net_2(u_lin[..., [1]])  # Process blocks individually
        y_nl = torch.cat((y_nl_1, y_nl_2), dim=-1)

        return y_nl


if __name__ == '__main__':

    lr = 1e-4
    num_iter = 100000
    test_freq = 100
    n_batch = 1

    N = 16384  # number of samples per period
    M = 20  # number of random phase multisine realizations
    P = 2  # number of periods
    nAmp = 5 #  number of different amplitudes

    # Column names in the dataset
    COL_F = ['fs']
    TAG_U = 'u'
    TAG_Y = 'y'

    # Load dataset
    #df_X = pd.read_csv(os.path.join("data", "WH_CombinedZeroMultisineSinesweep.csv"))
    df_X = pd.read_csv(os.path.join("data", "ParWHData_Estimation_Level1.csv"))
    df_X.columns = ['amplitude', 'fs', 'lines'] + [TAG_U + str(i) for i in range(M)] + [TAG_Y + str(i) for i in range(M)] + ['?']

    # Extract data
    y = np.array(df_X['y0'], dtype=np.float32)
    u = np.array(df_X['u0'], dtype=np.float32)
    fs = np.array(df_X[COL_F].iloc[0], dtype = np.float32)
    N = y.size
    ts = 1/fs
    t = np.arange(N)*ts

    u_torch = torch.tensor(u[None, :, None],  dtype=torch.float, requires_grad=False)
    y_meas_torch = torch.tensor(y[None, :, None],  dtype=torch.float, requires_grad=False)

    # In[Set-up model]

    # First linear section
    in_channels_1 = 1
    out_channels_1 = 2
    nb_1 = 3
    na_1 = 3
    y0_1 = torch.zeros((n_batch, na_1), dtype=torch.float)
    u0_1 = torch.zeros((n_batch, nb_1), dtype=torch.float)
    G1 = LinearMimo(in_channels_1, out_channels_1, nb_1, na_1)

    # Non-linear section
    F_nl = StaticNonLin()

    # Second linear section
    in_channels_2 = 2
    out_channels_2 = 1
    nb_2 = 3
    na_2 = 3
    y0_2 = torch.zeros((n_batch, na_2), dtype=torch.float)
    u0_2 = torch.zeros((n_batch, nb_2), dtype=torch.float)
    G2 = LinearMimo(in_channels_2, out_channels_2, nb_2, na_2)

    # In[Initialize linear systems]
    with torch.no_grad():
        G1.a_coeff[:, :, 0] = -0.9
        G1.b_coeff[:, :, 0] = 0.1
        G1.b_coeff[:, :, 1] = 0.1

        G2.a_coeff[:, :, 0] = -0.9
        G2.b_coeff[:, :, 0] = 0.1
        G1.b_coeff[:, :, 1] = 0.1

    # In[Setup optimizer]
    optimizer = torch.optim.Adam([
        {'params': G1.parameters(),    'lr': lr},
        {'params': F_nl.parameters(), 'lr': lr},
        {'params': G2.parameters(), 'lr': lr},
    ], lr=lr)

    # In[Training loop]
    LOSS = []
    start_time = time.time()
    for itr in range(0, num_iter):

        optimizer.zero_grad()

        # Simulate
        y_lin_1 = G1(u_torch, y0_1, u0_1)
        y_nl_1 = F_nl(y_lin_1)
        y_lin_2 = G2(y_nl_1, y0_2, u0_2)

        y_hat = y_lin_2

        # Compute fit loss
        err_fit = y_meas_torch - y_hat
        loss_fit = torch.mean(err_fit**2)
        loss = loss_fit

        LOSS.append(loss.item())
        if itr % test_freq == 0:
            print(f'Iter {itr} | Fit Loss {loss_fit:.8f}')

        # Optimize
        loss.backward()
        optimizer.step()

    train_time = time.time() - start_time
    print(f"\nTrain time: {train_time:.2f}") # 182 seconds


    # In[Save model]
    if not os.path.exists("models"):
        os.makedirs("models")
    model_filename = "model_WH"

    torch.save(G1.state_dict(), os.path.join("models", f"{model_filename}_G1.pkl"))
    torch.save(F_nl.state_dict(), os.path.join("models", f"{model_filename}_F_nl.pkl"))
    torch.save(G2.state_dict(), os.path.join("models", f"{model_filename}_G2.pkl"))


    # In[detach]
    y_hat_np = y_hat.detach().numpy()[0, :, 0]

    # In[Plot]
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(t, y, 'k', label="$y$")
    ax[0].plot(t, y_hat_np, 'r', label="$y$")

    ax[0].legend()
    ax[0].grid()

    ax[1].plot(t, u, 'k', label="$u$")
    ax[1].legend()
    ax[1].grid()

    plt.figure()
    plt.plot(LOSS)


    # In[Metrics]

    idx_metric = range(0, N)
    e_rms = util.metrics.error_rmse(y[idx_metric], y_hat_np[idx_metric])
    fit_idx = util.metrics.fit_index(y[idx_metric], y_hat_np[idx_metric])
    r_sq = util.metrics.r_squared(y[idx_metric], y_hat_np[idx_metric])

    print(f"RMSE: {e_rms:.4f}V\nFIT:  {fit_idx:.1f}%\nR_sq: {r_sq:.1f}")