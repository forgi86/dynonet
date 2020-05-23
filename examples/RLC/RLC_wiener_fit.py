import torch
import pandas as pd
import numpy as np
import os
from torchid.module.lti import SisoLinearDynamicOperator
from torchid.module.static import SisoStaticNonLinearity
import matplotlib.pyplot as plt
import time
import torch.nn as nn


# class StaticNonLin(nn.Module):
#
#     def __init__(self):
#         super(StaticNonLin, self).__init__()
#
#         self.net = nn.Sequential(
#             nn.Linear(1, 20),  # 2 states, 1 input
#             nn.ReLU(),
#             nn.Linear(20, 1)
#         )
#
#     def forward(self, y_lin):
#         y_nonlin = y_lin + self.net(y_lin)
#         return y_nonlin


if __name__ == '__main__':

    # Set seed for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)

    # Settings
    add_noise = True
    lr = 1e-4
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
    G = SisoLinearDynamicOperator(n_b, n_a, n_k=1)
    nn_static = SisoStaticNonLinearity()

    # Setup optimizer
    params_lin = G.parameters()
    optimizer = torch.optim.Adam([
        {'params': params_lin,    'lr': lr},
        {'params': nn_static.parameters(), 'lr': lr}
    ], lr=lr)


    # In[Train]
    LOSS = []
    start_time = time.time()
    for itr in range(0, num_iter):

        optimizer.zero_grad()

        # Simulate
        y_lin = G(u_torch)
        y_hat = nn_static(y_lin)

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

    # In[Plot]
    plt.figure()
    plt.plot(t, y_nonoise, 'k', label="$y$")
    plt.plot(t, y_noise, 'r', label="$y_{noise}$")
    plt.plot(t, y_hat.detach().numpy(), 'b', label="$\hat y$")
    plt.legend()

    plt.figure()
    plt.plot(LOSS)
    plt.grid(True)

    # In[Plot]
    plt.figure()
    plt.plot(y_lin.detach(), y_hat.detach())

    plt.figure()
    plt.plot(x[:, [0]], y_nonoise)
