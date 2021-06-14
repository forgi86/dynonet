import torch
import pandas as pd
import numpy as np
import os
from dynonet.lti import SisoLinearDynamicalOperator
from dynonet.static import SisoStaticNonLinearity
import matplotlib.pyplot as plt
import time
import torch.nn as nn
import dynonet.metrics


def normal_standard_cdf(val):
    return 1/2 * (1 + torch.erf(val/np.sqrt(2)))

# In[Main]
if __name__ == '__main__':

    # In[Set seed for reproducibility]
    np.random.seed(0)
    torch.manual_seed(0)

    # In[Settings]
    lr = 1e-4
    num_iter = 200000
    msg_freq = 100
    n_skip = 5000
    n_fit = 20000
    decimate = 1
    n_batch = 1
    n_b = 3
    n_a = 3
    C = 0.2  # threshold
    model_name = "model_WH_comparisons"

    # In[Column names in the dataset]
    COL_F = ['fs']
    COL_U = ['uBenchMark']
    COL_Y = ['yBenchMark']

    # In[Load dataset]
    df_X = pd.read_csv(os.path.join("data", "WienerHammerBenchmark.csv"))

    # Extract data
    y = np.array(df_X[COL_Y], dtype=np.float32)  # batch, time, channel
    u = np.array(df_X[COL_U], dtype=np.float32)
    fs = np.array(df_X[COL_F].iloc[0], dtype=np.float32)
    N = y.size
    ts = 1/fs
    t = np.arange(N)*ts

    # In[Compute v signal]
    v = np.empty(y.shape, dtype=np.float32)
    v[y > C] = 1.0
    v[y <= C] = -1.0

    # In[Fit data]
    v_fit = v[0:n_fit:decimate]
    y_fit = y[0:n_fit:decimate]
    u_fit = u[0:n_fit:decimate]
    t_fit = t[0:n_fit:decimate]

    # In[Prepare training tensors]
    u_fit_torch = torch.tensor(u_fit[None, :, :], dtype=torch.float, requires_grad=False)
    y_fit_torch = torch.tensor(y_fit[None, :, :], dtype=torch.float)
    v_fit_torch = torch.tensor(v_fit[None, :, :], dtype=torch.float)

    # In[Prepare model]
    G1 = SisoLinearDynamicalOperator(n_b, n_a, n_k=1)
    F_nl = SisoStaticNonLinearity(n_hidden=10, activation='tanh')
    G2 = SisoLinearDynamicalOperator(n_b, n_a)

    sigma_hat = torch.tensor(0.1, requires_grad=True)  # torch.randn(1, requires_grad = True)

    def model(u_in):
        y1_lin = G1(u_fit_torch)
        y1_nl = F_nl(y1_lin)
        y_hat = G2(y1_nl)
        return y_hat, y1_nl, y1_lin

    # In[Setup optimizer]
    optimizer = torch.optim.Adam([
        {'params': G1.parameters(), 'lr': lr},
        {'params': G2.parameters(), 'lr': lr},
        {'params': F_nl.parameters(), 'lr': lr},
        {'params': sigma_hat, 'lr': 1e-4},
    ], lr=lr)


    # In[Train]
    LOSS = []
    SIGMA = []
    start_time = time.time()
    for itr in range(0, num_iter):

        optimizer.zero_grad()

        y_hat, y1_nl, y1_lin = model(u_fit_torch)
        y_Phi_hat = normal_standard_cdf(v_fit_torch*((y_hat-C)/sigma_hat)) #:  #(1 + torch.erf(-v_fit_torch * (C - y_hat) / torch.abs(sigma_hat+1e-12) / np.sqrt(2))) / 2  # Cumulative
        y_hat_log = y_Phi_hat.log()
        loss_train = - y_hat_log.mean()

        LOSS.append(loss_train.item())
        SIGMA.append(sigma_hat.item())

        if itr % msg_freq == 0:
            with torch.no_grad():
                RMSE = torch.sqrt(loss_train)
            print(f'Iter {itr} | Fit Loss {loss_train:.5f} sigma_hat:{sigma_hat:.5f} ')

        loss_train.backward()
        optimizer.step()

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
    plt.show()

    # In[Plot loss]
    plt.figure()
    plt.plot(LOSS)
    plt.grid(True)
    plt.show()

    # In[Plot sigma]
    plt.figure()
    plt.plot(SIGMA)
    plt.grid(True)
    plt.show()

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
    plt.show()

    # In[RMSE]
    e_rms = dynonet.metrics.error_rmse(y_hat, y_fit)[0]
    print(f"RMSE: {e_rms:.2f}") # target: 1mv

    # In[v_hat]
    v_hat = np.empty_like(y_hat)
    v_hat[y_hat > C] = 1.0
    v_hat[y_hat <= C] = -1.0

    acc = np.sum(v_hat == v_fit)/(v_fit.shape[0])
    print(acc)





