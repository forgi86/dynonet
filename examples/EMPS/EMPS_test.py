import matplotlib
matplotlib.use("TkAgg")
import os
import numpy as np
import scipy as sp
import scipy.io
import torch
import matplotlib.pyplot as plt
from torchid.module.lti import MimoLinearDynamicalOperator, SisoLinearDynamicalOperator
from torchid.module.static import MimoStaticNonLinearity, MimoStaticNonLinearity
import util.metrics

if __name__ == '__main__':


    matplotlib.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    #matplotlib.rc('text', usetex=True)

    # In[Set seed for reproducibility]
    np.random.seed(0)
    torch.manual_seed(0)

    # In[Settings]

    model_name = 'EMPS_model'
    dataset = 'test'
    if dataset == 'id':
        dataset_filename = 'DATA_EMPS.mat'
    elif dataset == 'test':
        dataset_filename = 'DATA_EMPS_PULSES.mat'

    # In[Load dataset]

    emps_data = sp.io.loadmat(os.path.join("data", 'DATA_EMPS.mat'))
    y_ref = emps_data['qg'].astype(np.float32)
    y_meas = emps_data['qm'].astype(np.float32)
    u_in = emps_data['vir'].astype(np.float32)
    time_exp = emps_data['t'].astype(np.float32)
#    d_N = emps_data['pulses_N']
    ts = np.mean(np.diff(time_exp.ravel())) #time_exp[1] - time_exp[0]

    v_est = np.diff(y_meas, axis=0) / ts
    v_est = np.r_[[[0]], v_est]


    # In[Instantiate models]

    # Model blocks
    G1 = MimoLinearDynamicalOperator(1, 10, n_b=2, n_a=2, n_k=1)
    # Static sandwitched non-linearity
    F1 = MimoStaticNonLinearity(10, 5, activation='tanh')
    G2 = MimoLinearDynamicalOperator(5, 1, n_b=2, n_a=2, n_k=0)

    # Load identified model parameters
    model_folder = os.path.join("models", model_name)
    G1.load_state_dict(torch.load(os.path.join(model_folder, "G1.pkl")))
    F1.load_state_dict(torch.load(os.path.join(model_folder, "F1.pkl")))
    G2.load_state_dict(torch.load(os.path.join(model_folder, "G2.pkl")))

    # Model structure
    def model(u_in):
        y_lin_1 = G1(u_in)
        v_hat = F1(y_lin_1)
        v_hat = G2(v_hat)
        y_hat = torch.cumsum(v_hat, dim=1) * ts
        return y_hat, v_hat

    # In[Simulate]
    u_fit_torch = torch.tensor(u_in[None, :, :])
    y_hat, v_hat = model(u_fit_torch)

    # In[Detach]
    y_hat = y_hat.detach().numpy()[0, :, :]
    v_hat = v_hat.detach().numpy()[0, :, :]

    # In[Plot]
    # Simulation plot
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(6, 7.5))
    ax[0].plot(time_exp, y_meas, 'k', label='$y_{\mathrm{meas}}$')
    ax[0].plot(time_exp, y_hat, 'r', label='$y_{\mathrm{sim}}$')
    ax[0].legend(loc='upper right')
    ax[0].grid(True)
    ax[0].set_ylabel("Position (m)")

    ax[1].plot(time_exp, v_est,  'k',  label='$v_{\mathrm{est}}$')
    #ax[1].plot(time_exp, v_hat_np,  'r',  label='$v_{\mathrm{sim}}$')
    ax[1].grid(True)
    ax[1].legend(loc='upper right')
    ax[1].set_ylabel("Velocity (m/s)")

    ax[2].plot(time_exp, u_in, 'k*', label='$u_{in}$')
    ax[2].set_xlabel("Time (s)")
    ax[2].set_ylabel("Input (V)")
    ax[2].grid(True)
    ax[2].set_xlabel("Time (s)")
    plt.show()

    # In[Metrics]
    e_rms = util.metrics.error_rmse(y_meas, y_hat)[0]
    fit_idx = util.metrics.fit_index(y_meas, y_hat)[0]
    r_sq = util.metrics.r_squared(y_meas, y_hat)[0]

    print(f"RMSE: {e_rms:.2E} mm\nFIT:  {fit_idx:.1f}%\nR_sq: {r_sq:.2f}")

    # In[Plot for paper]
    t_test_start = 5900
    len_plot = 400

    plt.figure(figsize=(4, 3))
    plt.plot(time_exp, y_meas, 'k', label='$\mathbf{y}^{\mathrm{meas}}$')
    plt.plot(time_exp, y_hat, 'b', label='$\mathbf{y}$')
    plt.plot(time_exp, y_meas - y_hat, 'r', label='$\mathbf{e}$')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.ylabel("Position (m)")
    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.show()
#    plt.savefig('EMPS_timetrace.pdf')

