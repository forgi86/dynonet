import torch
import pandas as pd
import numpy as np
import os
from dynonet.lti import SisoFirLinearDynamicalOperator
import matplotlib.pyplot as plt
import time


if __name__ == '__main__':

    # In[Set seed for reproducibility]
    np.random.seed(0)
    torch.manual_seed(0)

    # In[Settings]
    add_noise = False
    lr = 1e-4
    num_iter = 20000
    test_freq = 100
    n_batch = 1
    n_b = 256  # number of FIR coefficients

    # In[Column names in the dataset]
    COL_T = ['time']
    COL_X = ['V_C', 'I_L']
    COL_U = ['V_IN']
    COL_Y = ['V_C']

    # In[Load dataset]
    df_X = pd.read_csv(os.path.join("data", "RLC_data_id_lin.csv"))
    t = np.array(df_X[COL_T], dtype=np.float32)
    y = np.array(df_X[COL_Y], dtype=np.float32)
    x = np.array(df_X[COL_X], dtype=np.float32)
    u = np.array(df_X[COL_U], dtype=np.float32)

    # In[Add measurement noise]
    std_noise_V = add_noise * 10.0
    std_noise_I = add_noise * 1.0
    std_noise = np.array([std_noise_V, std_noise_I])
    x_noise = np.copy(x) + np.random.randn(*x.shape) * std_noise
    x_noise = x_noise.astype(np.float32)

    # In[Output]
    y_noise = np.copy(x_noise[:, [0]])
    y_nonoise = np.copy(x[:, [0]])


    # Prepare data
    u_torch = torch.tensor(u[None, ...], dtype=torch.float, requires_grad=False)
    y_meas_torch = torch.tensor(y_noise[None, ...], dtype=torch.float)
    y_true_torch = torch.tensor(y_nonoise[None, ...], dtype=torch.float)

    # In[Second-order dynamical system custom defined]
    G = SisoFirLinearDynamicalOperator(n_b)


    # In[Setup optimizer]
    optimizer = torch.optim.Adam([
        {'params': G.parameters(),    'lr': lr},
    ], lr=lr)

    # In[Train]
    LOSS = []
    start_time = time.time()
    for itr in range(0, num_iter):

        optimizer.zero_grad()

        # Simulate
        y_hat = G(u_torch)

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

    # In[Detach and reshape]
    y_hat = y_hat.detach().numpy()[0, ...]

    # In[Plot]
    plt.figure()
    plt.plot(t, y_nonoise, 'k', label="$y$")
    plt.plot(t, y_noise, 'r', label="$y_{noise}$")
    plt.plot(t, y_hat, 'b', label="$\hat y$")
    plt.legend()

    plt.figure()
    plt.plot(LOSS)
    plt.grid(True)


    # In[FIR coefficients]

    g_pars = G.b_coeff[0, 0, :].detach().numpy()
    g_pars = g_pars[::-1]
    fig, ax = plt.subplots()
    ax.plot(g_pars)
