import torch
import pandas as pd
import numpy as np
import os
from torchid.old.linearsiso_TB import LinearDynamicalSystem
import matplotlib.pyplot as plt
import time
import torch.nn as nn

import util.metrics


class StaticNonLin(nn.Module):

    def __init__(self):
        super(StaticNonLin, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(1, 20),  # 2 states, 1 input
            nn.Tanh(),
            nn.Linear(20, 1)
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1e-3)
                nn.init.constant_(m.bias, val=0)

    def forward(self, y_lin):
        y_nl = y_lin + self.net(y_lin)
        return y_nl


if __name__ == '__main__':

    # Set seed for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)

    # Settings
    add_noise = True
    lr = 1e-3
    num_iter = 10000
    test_freq = 100
    n_fit = 40000
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
    u = u - np.mean(u)
    fs = 10**7/2**14
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


    # First dynamical system custom defined
    b1_coeff = np.array([0.0, 0.0], dtype=np.float32)
    f1_coeff = np.array([-1.0, 0.0], dtype=np.float32)
    G1 = LinearDynamicalSystem(b1_coeff, f1_coeff)
    y_init_1 = torch.zeros((n_batch, n_f), dtype=torch.float)
    u_init_1 = torch.zeros((n_batch, n_b), dtype=torch.float)

    # Second dynamical system custom defined
    b2_coeff = np.array([0.1, 0.0], dtype=np.float32)
    f2_coeff = np.array([-0.9, 0.0], dtype=np.float32)
    G2 = LinearDynamicalSystem(b2_coeff, f2_coeff)
    y_init_2 = torch.zeros((n_batch, n_f), dtype=torch.float)
    u_init_2 = torch.zeros((n_batch, n_b), dtype=torch.float)

    # Static non-linearity
    F_nl = StaticNonLin()

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
        y1_lin = G1(u_fit_torch, y_init_1, u_init_1)
        y1_nl = F_nl(y1_lin)
        y_hat = G2(y1_nl, y_init_2, u_init_2)

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

        if itr == 100:
            pass
        optimizer.step()

    train_time = time.time() - start_time
    print(f"\nTrain time: {train_time:.2f}")  # 182 seconds

    # In[To numpy]

    y_hat = y_hat.detach().numpy()
    y1_lin = y1_lin.detach().numpy()

    # In[Plot]
    plt.figure()
    plt.plot(t_fit, y_fit, 'k', label="$y$")
    plt.plot(t_fit, y_hat, 'b', label="$\hat y$")
    plt.legend()

    plt.figure()
    plt.plot(LOSS)
    plt.grid(True)

    # In[Plot static non-linearity]

    y1_lin_min = np.min(y1_lin) - 1e-6
    y1_lin_max = np.max(y1_lin) + 1e-6

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






