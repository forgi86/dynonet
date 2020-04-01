import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import control
from torchid.module.LTI import LinearMimo
import util.metrics
from examples.ParWH.common import  ParallelWHDataset
from examples.ParWH.common import StaticMimoNonLin


if __name__ == '__main__':

    model_filename = 'model_WH'
    n_batch = 1

    n_amp = 5 #  number of different amplitudes
    n_real = 20  # number of random phase multisine realizations
    N_per_period = 16384  # number of samples per period
    n_skip = 100
    P = 2  # number of periods
    seq_len = N_per_period * P  # data points per realization


    # Column names in the dataset
    TAG_U = 'u'
    TAG_Y = 'y'

    # In[Load dataset]

    dataset_list_level = ['ParWHData_Validation_Level' + str(i) for i in range(1, n_amp+1)]
    dataset_list = dataset_list_level + ['ParWHData_ValidationArrow']

    df_X_lst = []
    for dataset_name in dataset_list:
        dataset_filename = dataset_name + '.csv'
        df_Xi = pd.read_csv(os.path.join("data", dataset_filename))
        df_X_lst.append(df_Xi)


    df_X = df_X_lst[5]

    # Extract data
    y_meas = np.array(df_X['y'], dtype=np.float32)
    u = np.array(df_X['u'], dtype=np.float32)
    fs = np.array(df_X['fs'].iloc[0], dtype = np.float32)
    N = y_meas.size
    ts = 1/fs
    t = np.arange(N)*ts

    # In[Set-up model]

    # First linear section
    in_channels_1 = 1
    out_channels_1 = 2
    nb_1 = 6
    na_1 = 6
    y0_1 = torch.zeros((n_batch, na_1), dtype=torch.float)
    u0_1 = torch.zeros((n_batch, nb_1), dtype=torch.float)
    G1 = LinearMimo(in_channels_1, out_channels_1, nb_1, na_1)
    G1.load_state_dict(torch.load(os.path.join("models", f"{model_filename}_G1.pkl")))

    # Non-linear section
    F_nl = StaticMimoNonLin()
    F_nl.load_state_dict(torch.load(os.path.join("models", f"{model_filename}_F_nl.pkl")))

    # Second linear section
    in_channels_2 = 2
    out_channels_2 = 1
    nb_2 = 6
    na_2 = 6
    y0_2 = torch.zeros((n_batch, na_2), dtype=torch.float)
    u0_2 = torch.zeros((n_batch, nb_2), dtype=torch.float)
    G2 = LinearMimo(in_channels_2, out_channels_2, nb_2, na_2)
    G2.load_state_dict(torch.load(os.path.join("models", f"{model_filename}_G2.pkl")))

    # In[Predict]
    u_torch = torch.tensor(u[None, :, None],  dtype=torch.float, requires_grad=False)

    with torch.no_grad():
        y_lin_1 = G1(u_torch, y0_1, u0_1)
        y_nl_1 = F_nl(y_lin_1)
        y_lin_2 = G2(y_nl_1, y0_2, u0_2)
        y_hat = y_lin_2

    # In[Detach]
    y_lin_1 = y_lin_1.detach().numpy()[0, :, :]
    y_nl_1 = y_nl_1.detach().numpy()[0, :, :]

    y_hat = y_hat.detach().numpy()[0, :, 0]

    # In[Plot]
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(t, y_meas, 'k', label="$y$")
    ax[0].plot(t, y_hat, 'r', label="$\hat y$")
    ax[0].plot(t, y_meas - y_hat, 'g', label="$e$")
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(t, u, 'k', label="$u$")
    ax[1].legend()
    ax[1].grid()

    # In[Inspect linear model]

    # First linear block
    a_coeff_1 = G1.a_coeff.detach().numpy()
    b_coeff_1 = G1.b_coeff.detach().numpy()
    a_poly_1 = np.empty_like(a_coeff_1, shape=(out_channels_1, in_channels_1, na_1 + 1))
    a_poly_1[:, :, 0] = 1
    a_poly_1[:, :, 1:] = a_coeff_1[:, :, :]
    b_poly_1 = np.array(b_coeff_1)
    G1_sys = control.TransferFunction(b_poly_1, a_poly_1, ts)

    plt.figure()
    mag_G1_1, phase_G1_1, omega_G1_1 = control.bode(G1_sys[0, 0])
    plt.figure()
    mag_G1_2, phase_G1_2, omega_G1_2 = control.bode(G1_sys[1, 0])

    # Second linear block
    a_coeff_2 = G2.a_coeff.detach().numpy()
    b_coeff_2 = G2.b_coeff.detach().numpy()
    a_poly_2 = np.empty_like(a_coeff_2, shape=(out_channels_2, in_channels_2, na_2 + 1))
    a_poly_2[:, :, 0] = 1
    a_poly_2[:, :, 1:] = a_coeff_2[:, :, :]
    b_poly_2 = np.array(b_coeff_2)
    G2_sys = control.TransferFunction(b_poly_2, a_poly_2, ts)

    plt.figure()
    mag_G2_1, phase_G2_1, omega_G2_1 = control.bode(G2_sys[0, 0])
    plt.figure()
    mag_G2_2, phase_G2_2, omega_G2_2 = control.bode(G2_sys[0, 1])



    # In[Inspect linear model]

    fig, ax = plt.subplots(1, 2)
    ax[0].plot(y_lin_1[:, 0], y_nl_1[:, 0], '*')
    ax[0].grid("True")
    ax[0].set_xlabel('ylin_1')
    ax[0].set_ylabel('ynl_1')
    ax[1].plot(y_lin_1[:, 1], y_nl_1[:, 1], '*')
    ax[1].grid("True")
    ax[1].set_xlabel('ylin_2')
    ax[1].set_ylabel('ynl_2')

    # In[Metrics]

    idx_test = range(n_skip, N)

    e_rms = 1000*util.metrics.error_rmse(y_meas[idx_test], y_hat[idx_test])
    mae = 1000 * util.metrics.error_mae(y_meas[idx_test], y_hat[idx_test])
    fit_idx = util.metrics.fit_index(y_meas[idx_test], y_hat[idx_test])
    r_sq = util.metrics.r_squared(y_meas[idx_test], y_hat[idx_test])
    u_rms = 1000*util.metrics.error_rmse(u, 0)

    print(f"RMSE: {e_rms:.2f}mV\nMAE: {mae:.2f}mV\nFIT:  {fit_idx:.1f}%\nR_sq: {r_sq:.1f}\nRMSU: {u_rms:.2f}mV")