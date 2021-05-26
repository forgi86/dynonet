import torch
import pandas as pd
import numpy as np
import os
from torchid.module.lti import SisoLinearDynamicalOperator
from torchid.module.static import SisoStaticNonLinearity
import matplotlib.pyplot as plt
import control.matlab
import time


import util.metrics

# In[Main]
if __name__ == '__main__':

    # In[Set seed for reproducibility]
    np.random.seed(0)
    torch.manual_seed(0)

    # In[Settings]
    lr_ADAM = 2e-4
    lr_BFGS = 1e0
    num_iter_ADAM = 40000  # ADAM iterations 20000
    num_iter_BFGS = 0  # final BFGS iterations
    msg_freq = 100
    n_skip = 5000
    n_fit = 20000
    decimate = 1
    n_batch = 1
    n_b = 8
    n_a = 8
    model_name = "model_WH_proc_noise"

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
    fs = np.array(df_X[COL_F].iloc[0], dtype=np.float32).item()
    N = y.size
    ts = 1/fs
    t = np.arange(N)*ts

    # In[Fit data]
    y_fit = y[0:n_fit:decimate]
    u_fit = u[0:n_fit:decimate]
    t_fit = t[0:n_fit:decimate]
    N_fit = y_fit.shape[0]

    # In[Add process noise]

    std_v = 0.1
    w_v = 1e3
    tau_v = 1/w_v

    Hu = control.TransferFunction([1], [1 / w_v, 1])
    Hu = Hu * Hu
    Hud = control.matlab.c2d(Hu, ts)
    t_imp = np.arange(1000) * ts
    t_imp, y_imp = control.impulse_response(Hud, t_imp)
    #y = y[0]
    std_tmp = np.sqrt(np.sum(y_imp ** 2))  # np.sqrt(trapz(y**2,t))
    Hu = Hu / std_tmp * std_v
    Hud = Hud / std_tmp * std_v

    # N_skip int(20 * tau_v // ts) # skip initial samples to get a regime sample of d
    n_skip_d = 0
    N_sim_d = n_fit + n_skip_d
    e = np.random.randn(N_sim_d)
    te = np.arange(N_sim_d) * ts
    d, u = control.forced_response(Hud, te, e)
    d_fast = d[n_skip_d:]
    d_fast = d_fast.reshape(-1, 1)
    y_fit_clean = np.copy(y_fit)
    y_fit = y_fit + d_fast

    # In[Prepare training tensors]
    u_fit_torch = torch.tensor(u_fit[None, :, :], dtype=torch.float, requires_grad=False)
    y_fit_torch = torch.tensor(y_fit[None, :, :], dtype=torch.float)

    # In[Prepare model]
    G1 = SisoLinearDynamicalOperator(n_b, n_a, n_k=1)
    F_nl = SisoStaticNonLinearity(n_hidden=10, activation='tanh')
    G2 = SisoLinearDynamicalOperator(n_b, n_a)

    H_inv = SisoLinearDynamicalOperator(2, 2, n_k=1)

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
        {'params': H_inv.parameters(), 'lr': lr_ADAM},
    ], lr=lr_ADAM)

    optimizer_LBFGS = torch.optim.LBFGS(list(G1.parameters()) + list(G2.parameters()) + list(F_nl.parameters()) + list(H_inv.parameters()), lr=lr_BFGS)

    def closure():
        optimizer_LBFGS.zero_grad()

        # Simulate
        y_hat, y1_nl, y1_lin = model(u_fit_torch)

        # Compute fit loss
        err_fit_v = y_fit_torch[:, :, :] - y_hat[:, :, :]  # simulation error loss
        err_fit_e = err_fit_v + H_inv(err_fit_v)
        err_fit_e = err_fit_e[:, n_skip:, :]

        loss = torch.mean(err_fit_e**2)*1000

        # Backward pas
        loss.backward()
        return loss


    # In[Train]
    LOSS = []
    start_time = time.time()
    for itr in range(0, num_iter):

        if itr < num_iter_ADAM:
            msg_freq = 10
            loss_train = optimizer_ADAM.step(closure)
        else:
            msg_freq = 10
            loss_train = optimizer_LBFGS.step(closure)

        LOSS.append(loss_train.item())
        if itr % msg_freq == 0:
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
    torch.save(H_inv.state_dict(), os.path.join(model_folder, "H_inv.pkl"))

    # In[Simulate one more time]
    with torch.no_grad():
        y_hat, y1_nl, y1_lin = model(u_fit_torch)

    # In[Detach]
    y_hat = y_hat.detach().numpy()[0, :, :]
    y1_lin = y1_lin.detach().numpy()[0, :, :]
    y1_nl = y1_nl.detach().numpy()[0, :, :]

    # In[Plot]
    plt.figure()
    # plt.plot(t_fit, y_fit, 'k', label="$y$")
    plt.plot(t_fit, y_fit_clean, 'r', label="$y_{clean}$")
    plt.plot(t_fit, y_hat, 'b', label="$\hat y$")
    plt.legend()
    plt.show()

    # In[Plot loss]
    plt.figure()
    plt.plot(LOSS)
    plt.grid(True)
    plt.show()

    # In[Plot static non-linearity]

    y1_lin_min = np.min(y1_lin)
    y1_lin_max = np.max(y1_lin)

    in_nl = np.arange(y1_lin_min, y1_lin_max, (y1_lin_max - y1_lin_min)/1000).astype(np.float32).reshape(-1, 1)

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

    # In[Plot]

    # Inspect process noise blocks
    n_imp = 128
    H_inv_num, H_inv_den = H_inv.get_tfdata()
    H_inv_sys = 1 + control.TransferFunction(H_inv_num, H_inv_den, ts)
    H_sys = 1/H_inv_sys

    plt.figure()
    mag_H, phase_H, omega_H = control.bode(H_sys, omega_limits=[1e1, 1e5])
    plt.suptitle("$H_inv$ bode plot")
#    plt.savefig(os.path.join("models", model_name, "G1_bode.pdf"))

    # In[]
    control.bode(H_sys)
    control.bode(Hud*1000)
    plt.show()
