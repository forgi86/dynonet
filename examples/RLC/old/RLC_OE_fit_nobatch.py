import torch
import pandas as pd
import numpy as np
import os
from torchid.old.linearsiso_nobatch import LinearDynamicalSystem
import matplotlib.pyplot as plt
import time

if __name__ == '__main__':

    # Set seed for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)

    # Settings
    add_noise = True
    lr = 1e-3
    num_iter = 3000
    test_freq = 100
    n_batch = 1
    n_b = 2
    n_f = 2

    # Column names in the dataset
    COL_T = ['time']
    COL_X = ['V_C', 'I_L']
    COL_U = ['V_IN']
    COL_Y = ['V_C']

    # Load dataset
    df_X = pd.read_csv(os.path.join("../data", "RLC_data_id_nl.csv"))
    t = np.array(df_X[COL_T], dtype=np.float32)
    y = np.array(df_X[COL_Y], dtype=np.float32)
    x = np.array(df_X[COL_X], dtype=np.float32)
    u = np.array(df_X[COL_U], dtype=np.float32)

    # Add measurement noise
    std_noise_V = add_noise * 10.0
    std_noise_I = add_noise * 1.0
    std_noise = np.array([std_noise_V, std_noise_I])
    x_noise = np.copy(x) + np.random.randn(*x.shape) * std_noise
    x_noise = x_noise.astype(np.float32)

    # Output
    y_noise = np.copy(x_noise[:, [0]])
    y_nonoise = np.copy(x[:, [0]])

    # Second-order dynamical system custom defined
    G = LinearDynamicalSystem.apply

    # Prepare data
    u_torch = torch.tensor(u, dtype=torch.float, requires_grad=False)
    y_meas_torch = torch.tensor(y_noise, dtype=torch.float)
    y_true_torch = torch.tensor(y_nonoise, dtype=torch.float)
    y_0 = torch.zeros(n_f, dtype=torch.float)
    u_0 = torch.zeros(n_b, dtype=torch.float)
    # coefficients of a 2nd order oscillator
#    b_coeff = torch.tensor([0.0706464146944544, 0], dtype=torch.float, requires_grad=True)  # b_1, b_2
#    f_coeff = torch.tensor([-1.87212998940304, 0.942776404097492], dtype=torch.float, requires_grad=True)  # f_1, f_2
    b_coeff = torch.tensor([0.0306464146944544, 0], dtype=torch.float, requires_grad=True)  # b_1, b_2
    f_coeff = torch.tensor([-1.0, 0.9], dtype=torch.float, requires_grad=True)  # f_1, f_2

    # Setup optimizer
    params_net = [b_coeff, f_coeff]
    optimizer = torch.optim.Adam([
        {'params': params_net,    'lr': lr},
    ], lr=lr)

    # In[Train]
    LOSS = []
    start_time = time.time()
    for itr in range(0, num_iter):

        optimizer.zero_grad()

        # Simulate
        y_hat = G(b_coeff, f_coeff, u_torch, y_0, u_0)

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


