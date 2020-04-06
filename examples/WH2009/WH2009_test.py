import torch
import pandas as pd
import numpy as np
import os
from torchid.module.LTI import LinearSiso
from torchid.module.static import StaticSisoNonLin

import matplotlib.pyplot as plt
import torch.nn as nn
import control
import util.metrics


if __name__ == '__main__':

    # Set seed for reproducibility
    model_name = 'model_WH_LBFGS'

    # Settings
    n_b = 3
    n_f = 3
    n_fit = 100000
    n_batch = 1

    # Column names in the dataset
    COL_F = ['fs']
    COL_U = ['uBenchMark']
    COL_Y = ['yBenchMark']

    # Load dataset
    df_X = pd.read_csv(os.path.join("data", "WienerHammerBenchmark.csv"))

    # Extract data
    y_meas = np.array(df_X[COL_Y], dtype=np.float32)
    u = np.array(df_X[COL_U], dtype=np.float32)
    fs = np.array(df_X[COL_F].iloc[0], dtype = np.float32)
    N = y_meas.size
    ts = 1/fs
    t = np.arange(N)*ts

    t_fit_start = 0
    t_fit_end = 100000
    t_test_start = 100000
    t_test_end = 188000
    t_skip = 1000

    # In[Instantiate models]

    # Create models
    G1 = LinearSiso(n_b=3, n_a=3)
    G2 = LinearSiso(n_b=3, n_a=3)
    F_nl = StaticSisoNonLin()

    model_folder = os.path.join("models", model_name)
    # Create model parameters
    G1.load_state_dict(torch.load(os.path.join(model_folder, "G1.pkl")))
    F_nl.load_state_dict(torch.load(os.path.join(model_folder, "F_nl.pkl")))
    G2.load_state_dict(torch.load(os.path.join(model_folder, "G2.pkl")))

    # In[Predict]

    u_torch = torch.tensor(u[None, :, :])
    y1_lin = G1(u_torch)
    y1_nl = F_nl(y1_lin)
    y_hat = G2(y1_nl)

    # In[Detach]
    y_hat = y_hat.detach().numpy()[0, :, :]
    y1_lin = y1_lin.detach().numpy()[0, :, :]
    y1_nl = y1_nl.detach().numpy()[0, :, :]

    # In[Plot]
    plt.figure()
    plt.plot(t, y_meas, 'k', label="$y$")
    plt.plot(t, y_hat, 'b', label="$\hat y$")
    plt.plot(t, y_meas - y_hat, 'r', label="$e$")
    plt.legend(loc='upper left')

    # In[Inspect linear model]

    G1_sys = control.TransferFunction(G1.b_coeff[0, 0, :].detach().numpy(), np.r_[1.0, G1.a_coeff[0, 0, :].detach().numpy()], ts)
    plt.figure()
    mag_G1, phase_G1, omega_G1 = control.bode(G1_sys)

    G2_sys = control.TransferFunction(G2.b_coeff[0, 0, :].detach().numpy(), np.r_[1.0, G2.a_coeff[0, 0, :].detach().numpy()], ts)
    plt.figure()
    mag_G2, phase_G2, omega_G2 = control.bode(G2_sys)

    # In[Inspect static non-linearity]

    y1_lin_min = np.min(y1_lin)
    y1_lin_max = np.max(y1_lin)

    in_nl = np.arange(y1_lin_min, y1_lin_max, (y1_lin_max- y1_lin_min)/1000).astype(np.float32).reshape(-1, 1)

    with torch.no_grad():
        out_nl = F_nl(torch.as_tensor(in_nl))

    plt.figure()
    plt.plot(in_nl, out_nl, 'b')
    plt.plot(in_nl, out_nl, 'b')
    plt.xlabel('Static non-linearity input (-)')
    plt.ylabel('Static non-linearity input (-)')
    plt.grid(True)

    # In[Metrics]
    idx_test = range(t_test_start + t_skip, t_test_end)
    e_rms = 1000*util.metrics.error_rmse(y_meas[idx_test], y_hat[idx_test])[0]
    fit_idx = util.metrics.fit_index(y_meas[idx_test], y_hat[idx_test])[0]
    r_sq = util.metrics.r_squared(y_meas[idx_test], y_hat[idx_test])[0]

    print(f"RMSE: {e_rms:.1f}V\nFIT:  {fit_idx:.1f}%\nR_sq: {r_sq:.1f}")


