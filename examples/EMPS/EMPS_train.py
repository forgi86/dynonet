import matplotlib
matplotlib.use("TkAgg")
import os
import numpy as np
import scipy as sp
import scipy.io
import scipy.signal
import torch
import torch.optim as optim
import time
import matplotlib.pyplot as plt
from torchid.module.lti import MimoLinearDynamicOperator, SisoLinearDynamicOperator
from torchid.module.static import MimoStaticNonLinearity, MimoStaticNonLinearity

if __name__ == '__main__':

    # In[Set seed for reproducibility]
    np.random.seed(0)
    torch.manual_seed(0)

    # In[]
    lr = 1e-4
    num_iter = 50000
    msg_freq = 100
    model_name = 'EMPS_model'

    # In[Load dataset]
    emps_data = sp.io.loadmat(os.path.join("data", "DATA_EMPS.mat"))
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
    G1 = MimoLinearDynamicOperator(1, 10, n_b=2, n_a=2, n_k=1)
    # Static sandwitched non-linearity
    F1 = MimoStaticNonLinearity(10, 5, activation='tanh')
    G2 = MimoLinearDynamicOperator(5, 1, n_b=2, n_a=2, n_k=0)

    # Model structure
    def model(u_in):
        y_lin_1 = G1(u_in)
        v_hat = F1(y_lin_1)
        v_hat = G2(v_hat)
        y_hat = torch.cumsum(v_hat, dim=1) * ts
        return y_hat, v_hat

    # In[Optimizer]
    optimizer = torch.optim.Adam([
        {'params': G1.parameters(), 'lr': lr},
        {'params': F1.parameters(), 'lr': lr},
        {'params': G2.parameters(), 'lr': lr},
    ], lr=lr)

    # In[Prepare tensors]

    #q_meas = (q_meas - 1.23)/0.08
    u_fit_torch = torch.tensor(u_in[None, :, :])
    y_fit_torch = torch.tensor(y_meas[None, :, :])

    # In[Train]
    LOSS = []
    start_time = time.time()
    for itr in range(0, num_iter):

        optimizer.zero_grad()

        y_hat, v_hat = model(u_fit_torch)

        err_fit = y_fit_torch - y_hat
        loss = torch.mean(err_fit ** 2) * 10

        LOSS.append(loss.item())
        if itr % msg_freq == 0:
            print(f'Iter {itr} | Fit Loss {loss:.6f}')

        loss.backward()
        optimizer.step()

    train_time = time.time() - start_time

    print(f"\nTrain time: {train_time:.2f}") # 1900 seconds, loss was still going down

    # In[Save model]

    if model_name is not None:
        model_folder = os.path.join("models", model_name)
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        torch.save(G1.state_dict(), os.path.join(model_folder, "G1.pkl"))
        torch.save(F1.state_dict(), os.path.join(model_folder, "F1.pkl"))
        torch.save(G2.state_dict(), os.path.join(model_folder, "G2.pkl"))

    # In[Detach]
    y_hat_np = y_hat.detach().numpy()[0, :, 0]
    v_hat_np = v_hat.detach().numpy()[0, :, 0]

    # In[Plot loss]
    fig, ax = plt.subplots(figsize=(6, 7.5))
    ax.plot(LOSS)

    # In[Plot]
    # Simulation plot
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(6, 7.5))
    ax[0].plot(time_exp, y_meas, 'k', label='$q_{\mathrm{meas}}$')
    ax[0].plot(time_exp, y_hat_np, 'r', label='$q_{\mathrm{sim}}$')
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