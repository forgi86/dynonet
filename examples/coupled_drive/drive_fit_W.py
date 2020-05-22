import torch
import pandas as pd
import numpy as np
import os
from torchid.module.lti import SisoLinearDynamicOperator
from torchid.module.static import SisoStaticNonLin
import time
import util.metrics
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # In[Set seed for reproducibility]
    np.random.seed(0)
    torch.manual_seed(0)

    # In[Settings]
    lr = 1e-3
    num_iter = 400000
    test_freq = 100
    n_fit = 500
    decimate = 1
    n_batch = 1
    n_b = 4
    n_a = 4

    # Column names in the dataset
    # Column names in the dataset
    COL_U = ['u1']
    COL_Y = ['z1']

    # In[Load dataset]
    df_X = pd.read_csv(os.path.join("data", "DATAPRBS.csv"))

    # Extract data
    y = np.array(df_X[COL_Y], dtype=np.float32)
    u = np.array(df_X[COL_U], dtype=np.float32)
    N = y.size
    fs = 50  # Sampling frequency (Hz)
    ts = 1/fs
    t = np.arange(N)*ts

    # Fit data
    y_fit = y[:n_fit:decimate] - 1.5
    u_fit = u[:n_fit:decimate]
    t_fit = t[0:n_fit:decimate]


    # In[Prepare data]
    u_fit_torch = torch.tensor(u_fit[None, :, :], dtype=torch.float)
    y_fit_torch = torch.tensor(y_fit[None, :, :], dtype=torch.float)


    # In[Setup model]
    G1 = SisoLinearDynamicOperator(n_b=4, n_a=4, n_k=1)
    F = SisoStaticNonLin(n_hidden=16, activation='tanh')
    G2 = SisoLinearDynamicOperator(n_b=4, n_a=4, n_k=1)

    # Setup optimizer
    optimizer = torch.optim.Adam([
        {'params': G1.parameters(), 'lr': lr},
        {'params': F.parameters(), 'lr': lr},
    ], lr=lr)

    # In[Train]
    LOSS = []
    start_time = time.time()
    for itr in range(0, num_iter):

        optimizer.zero_grad()

        # Simulate
        y_lin = G1(u_fit_torch)
        y_nl =  F(y_lin)
        y_hat = G2(y_nl)

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
    print(f"\nTrain time: {train_time:.2f}") # 182 seconds

    # In[To numpy]

    y_hat = y_hat.detach().numpy()[0, :, :]
    y_lin = y_lin.detach().numpy()[0, :, :]


    # In[Plot]
    plt.figure()
    plt.plot(t_fit, y_fit, 'k', label="$y$")
    plt.plot(t_fit, y_hat, 'b', label="$\hat y$")
    plt.legend()

    plt.figure()
    plt.plot(LOSS)
    plt.grid(True)

    # In[Plot static non-linearity]

    y1_lin_min = np.min(y_lin)
    y1_lin_max = np.max(y_lin)

    in_nl = np.arange(y1_lin_min, y1_lin_max, (y1_lin_max- y1_lin_min)/1000).astype(np.float32).reshape(-1, 1)

    with torch.no_grad():
        out_nl = F(torch.as_tensor(in_nl))

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


    # In[Analysis]
    import control

    #Gg_lin = control.TransferFunction(G.b_coeff.detach().numpy(), np.r_[1.0, G.a_coeff.detach().numpy()], ts)
    mag, phase, omega = control.bode(Gg_lin)







