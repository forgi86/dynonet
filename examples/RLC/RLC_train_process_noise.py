import torch
import pandas as pd
import numpy as np
import os
import control.matlab
from torchid.module.lti import SisoLinearDynamicalOperator
import matplotlib.pyplot as plt
import time


if __name__ == '__main__':

    # In[Set seed for reproducibility]
    np.random.seed(0)
    torch.manual_seed(0)

    # In[Settings]
    model_name = 'IIR_proc_noise'
    add_noise = False
    lr = 1e-4
    num_iter = 20000
    test_freq = 100
    n_batch = 1
    n_b = 2
    n_a = 2
    do_PEM = False

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
    #std_noise_V = add_noise * 10.0
    #std_noise_I = add_noise * 1.0
    #std_noise = np.array([std_noise_V, std_noise_I])
    #x_noise = np.copy(x) + np.random.randn(*x.shape) * std_noise
    #x_noise = x_noise.astype(np.float32)
    # In[Add process noise]

    ts = t[1, 0] - t[0, 0]
    n_fit = t.shape[0]

    std_v = 100
    w_v = 5e4
    tau_v = 1/w_v

    Hu = control.TransferFunction([1], [1 / w_v, 1])
    Hu = Hu * Hu
    Hud = control.matlab.c2d(Hu, ts)
    t_imp = np.arange(1000) * ts
    t_imp, y_imp = control.impulse_response(Hud, t_imp)
    #y = y[0]
    std_tmp = np.sqrt(np.sum(y_imp ** 2))  # np.sqrt(trapz(y**2,t))
    Hud = Hud / std_tmp * std_v

    n_skip_d = 0
    N_sim_d = n_fit + n_skip_d
    e = np.random.randn(N_sim_d)
    te = np.arange(N_sim_d) * ts
    _, d, _ = control.forced_response(Hud, te, e)
    d_fast = d[n_skip_d:]
    d_fast = d_fast.reshape(-1, 1)
    y_nonoise = np.copy(y)
    y_noise = y + d_fast

    # Prepare data
    u_torch = torch.tensor(u[None, ...], dtype=torch.float, requires_grad=False)
    y_meas_torch = torch.tensor(y_noise[None, ...], dtype=torch.float)
    y_true_torch = torch.tensor(y_nonoise[None, ...], dtype=torch.float)

    # In[Second-order dynamical system custom defined]
    G = SisoLinearDynamicalOperator(n_b, n_a)
    H_inv = SisoLinearDynamicalOperator(2, 2, n_k=1)

    with torch.no_grad():
        G.b_coeff[0, 0, 0] = 0.01
        G.b_coeff[0, 0, 1] = 0.0

        G.a_coeff[0, 0, 0] = -0.9
        G.b_coeff[0, 0, 1] = 0.01

    # In[Setup optimizer]
    optimizer = torch.optim.Adam([
        {'params': G.parameters(),    'lr': lr},
        {'params': H_inv.parameters(), 'lr': lr},
    ], lr=lr)

    # In[Train]
    LOSS = []
    start_time = time.time()
    for itr in range(0, num_iter):

        optimizer.zero_grad()

        # Simulate
        y_hat = G(u_torch)

        # Compute fit loss
        err_fit_v = y_meas_torch - y_hat

        if do_PEM:
            err_fit_e = err_fit_v + H_inv(err_fit_v)
            err_fit = err_fit_e
        else:
            err_fit = err_fit_v

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

    # In[Save model]

    model_folder = os.path.join("models", model_name)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    torch.save(G.state_dict(), os.path.join(model_folder, "G.pkl"))
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


