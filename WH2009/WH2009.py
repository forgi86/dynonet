import torch
import pandas as pd
import numpy as np
import os
from torchid.linearsiso import LinearDynamicalSystem
import matplotlib.pyplot as plt
import time
import torch.nn as nn


class StaticNonLin(nn.Module):

    def __init__(self):
        super(StaticNonLin, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(1, 20),  # 2 states, 1 input
            nn.Tanh(),
            nn.Linear(20, 1)
        )

    def forward(self, y_lin):
        y_nl = y_lin + self.net(y_lin)
        return y_nl


if __name__ == '__main__':

    # Set seed for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)

    # Settings
    add_noise = True
    lr = 1e-4
    num_iter = 20000
    test_freq = 100
    n_batch = 1
    n_b = 3
    n_f = 3

    # Column names in the dataset
    COL_F = ['fs']
    COL_U = ['uBenchMark']
    COL_Y = ['yBenchMark']

    # Load dataset
    df_X = pd.read_csv(os.path.join("data", "WienerHammerBenchmark.csv"))

    # Extract data
    y = np.array(df_X[COL_Y], dtype=np.float32)
    u = np.array(df_X[COL_U], dtype=np.float32)
    fs = np.array(df_X[COL_F].iloc[0], dtype = np.float32)
    N = y.size
    ts = 1/fs
    t = np.arange(N)*ts

    # Fit data
    n_fit = 100000
    y_fit = y[:n_fit]
    u_fit = u[:n_fit]
    t_fit = t[0:n_fit]


    # Prepare data
    u_fit_torch = torch.tensor(u_fit, dtype=torch.float, requires_grad=False)
    y_fit_torch = torch.tensor(y_fit, dtype=torch.float)
    y_0 = torch.zeros((n_batch, n_f), dtype=torch.float)
    u_0 = torch.zeros((n_batch, n_b), dtype=torch.float)

    # Second-order dynamical system custom defined
    b1_coeff = np.array([0.1, 0.0, 0.0], dtype=np.float)
    f1_coeff = np.array([-0.9, 0.0, 0.0], dtype=np.float)
    G1 = LinearDynamicalSystem(b1_coeff, f1_coeff)

    # Second-order dynamical system custom defined
    b2_coeff = np.array([0.1, 0.0, 0.0], dtype=np.float)
    f2_coeff = np.array([-0.9, 0.0, 0.0], dtype=np.float)
    G2 = LinearDynamicalSystem(b2_coeff, f2_coeff)

    # Static sandwitched non-linearity
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
        y1_lin = G1(u_fit_torch, y_0, u_0)
        y1_nl = F_nl(y1_lin)
        y_hat = G2(y1_nl, y_0, u_0)

        # Compute fit loss
        err_fit = y_fit_torch - y_hat
        loss_fit = torch.mean(err_fit**2)
        loss = loss_fit

        LOSS.append(loss.item())
        if itr % test_freq == 0:
            print(f'Iter {itr} | Fit Loss {loss_fit:.6f}')

        # Optimize
        loss.backward()
        optimizer.step()

    train_time = time.time() - start_time
    print(f"\nTrain time: {train_time:.2f}") # 182 seconds

    # In[Plot]
    plt.figure()
    plt.plot(t_fit, y_fit, 'k', label="$y$")
    plt.plot(t_fit, y_hat.detach().numpy(), 'b', label="$\hat y$")
    plt.legend()

    plt.figure()
    plt.plot(LOSS)
    plt.grid(True)



