import torch
import pandas as pd
import numpy as np
import os
from torchid.module.lti import SisoFirLinearDynamicalOperator
from torchid.module.static import SisoStaticNonLinearity
import matplotlib.pyplot as plt
import time
import torch.nn as nn

import util.metrics

# In[Main]
if __name__ == '__main__':

    # In[Set seed for reproducibility]
    np.random.seed(0)
    torch.manual_seed(0)

    # In[Settings]
    lr_ADAM = 1e-4
    lr_BFGS = 1e-1
    num_iter_ADAM = 100000
    num_iter_BFGS = 0
    test_freq = 100
    n_fit = 100000
    decimate = 1
    n_batch = 1
    n_b = 128
    model_name = "model_WH_FIR"

    num_iter = num_iter_ADAM + num_iter_BFGS

    # In[Column names in the dataset]
    COL_F = ['fs']
    COL_U = ['uBenchMark']
    COL_Y = ['yBenchMark']

    # In[Load dataset]
    df_X = pd.read_csv(os.path.join("data", "WienerHammerBenchmark.csv"))

    # Extract data
    y = np.array(df_X[COL_Y], dtype=np.float32)  # batch, time, channel
    u = np.array(df_X[COL_U], dtype=np.float32)
    fs = np.array(df_X[COL_F].iloc[0], dtype = np.float32)
    N = y.size
    ts = 1/fs
    t = np.arange(N)*ts

    # In[Fit data]
    y_fit = y[0:n_fit:decimate]
    u_fit = u[0:n_fit:decimate]
    t_fit = t[0:n_fit:decimate]


    # In[Prepare training tensors]
    u_fit_torch = torch.tensor(u_fit[None, :, :], dtype=torch.float, requires_grad=False)
    y_fit_torch = torch.tensor(y_fit[None, :, :], dtype=torch.float)

    # In[Prepare model]
    G1 = SisoFirLinearDynamicalOperator(n_b=n_b)
    F_nl = SisoStaticNonLinearity()
    G2 = SisoFirLinearDynamicalOperator(n_b=n_b)

    def model(u_in):
        y1_lin = G1(u_fit_torch)
        y1_nl = F_nl(y1_lin)
        y_hat = G2(y1_nl)
        return y_hat, y1_nl, y1_lin

    # In[Setup optimizer]
    optimizer_ADAM = torch.optim.Adam([
        {'params': G1.parameters(), 'lr': lr_ADAM},
        {'params': G2.parameters(), 'lr': lr_ADAM},
        {'params': F_nl.parameters(), 'lr': lr_ADAM},
    ], lr=lr_ADAM)

    optimizer_LBFGS = torch.optim.LBFGS(list(G1.parameters()) + list(G2.parameters()) + list(F_nl.parameters()), lr=lr_BFGS)


    def closure():
        optimizer_LBFGS.zero_grad()

        # Simulate
        y_hat, y1_nl, y1_lin = model(u_fit_torch)

        # Compute fit loss
        err_fit = y_fit_torch - y_hat
        loss = torch.mean(err_fit**2)

        # Backward pas
        loss.backward()
        return loss


    # In[Train]
    LOSS = []
    start_time = time.time()
    for itr in range(0, num_iter):

        if itr < num_iter_ADAM:
            test_freq = 10
            loss_train = optimizer_ADAM.step(closure)
        else:
            test_freq = 10
            loss_train = optimizer_LBFGS.step(closure)

        LOSS.append(loss_train.item())
        if itr % test_freq == 0:
            with torch.no_grad():
                RMSE = torch.sqrt(loss_train)
            print(f'Iter {itr} | Fit Loss {loss_train:.6f} | RMSE:{RMSE:.4f}')

    train_time = time.time() - start_time
    print(f"\nTrain time: {train_time:.2f}")

    # In[Save model]
    model_folder = os.path.join("models", model_name)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    torch.save(G1.state_dict(), os.path.join(model_folder, "G1.pkl"))
    torch.save(F_nl.state_dict(), os.path.join(model_folder, "F_nl.pkl"))
    torch.save(G2.state_dict(), os.path.join(model_folder, "G2.pkl"))


    # In[Simulate one more time]
    with torch.no_grad():
        y_hat, y1_nl, y1_lin = model(u_fit_torch)

    # In[Detach]
    y_hat = y_hat.detach().numpy()[0, :, :]
    y1_lin = y1_lin.detach().numpy()[0, :, :]
    y1_nl = y1_nl.detach().numpy()[0, :, :]

    # In[Plot]
    plt.figure()
    plt.plot(t_fit, y_fit, 'k', label="$y$")
    plt.plot(t_fit, y_hat, 'b', label="$\hat y$")
    plt.legend()

    # In[Plot loss]
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
