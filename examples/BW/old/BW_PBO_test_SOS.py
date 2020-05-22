import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
from torchid.module.lti import MimoSecondOrderDynamicOperator
import util.metrics
from torchid.module.static import MimoStaticNonLin


if __name__ == '__main__':

    # In[Settings]
    #h5_filename = 'train.h5'
    h5_filename = 'test.h5'
    #signal_name = 'multisine'
    signal_name = 'multisine'
    #signal_name = 'sinesweep' # available in test
    model_name = "model_BW_PBO_SOS_LBFGS_refined"


    # In[Load dataset]

    h5_data = h5py.File(os.path.join("BoucWenFiles", "Test signals", h5_filename), 'r')
    dataset_list = h5_data.keys()
    y = np.array(h5_data[signal_name]['y']).transpose()  # MATLAB saves data in column major order...
    if y.ndim == 2:
        y = y[..., None]
    u = np.array(h5_data[signal_name]['u']).transpose()
    if u.ndim == 2:
        u = u[..., None]

    fs = np.array(h5_data[signal_name]['fs']).item()

    N = y.shape[1]
    ts = 1.0/fs
    t = np.arange(N)*ts


    # In[Scale data]
    scaler_y = 0.0006  # approx std(y_train)
    scaler_u = 50  # approx std(u_train)

    y = y/scaler_y
    u = u/scaler_u

    # In[Data to float 32]
    y = y.astype(np.float32)
    u = u.astype(np.float32)
    t = t.astype(np.float32)

    # In[Instantiate models]

    # Second-order dynamical system
    G1 = MimoSecondOrderDynamicOperator(1, 8)
    F1 = MimoStaticNonLin(8, 4, n_hidden=10) #torch.nn.ReLU() #StaticMimoNonLin(3, 3, n_hidden=10)
    G2 = MimoSecondOrderDynamicOperator(4, 2)
    F2 = MimoStaticNonLin(2, 1, n_hidden=10)
    G3 = MimoSecondOrderDynamicOperator(1, 1)

    model_folder = os.path.join("models", model_name)
    G1.load_state_dict(torch.load(os.path.join(model_folder, "G1.pkl")))
    F1.load_state_dict(torch.load(os.path.join(model_folder, "F1.pkl")))
    G2.load_state_dict(torch.load(os.path.join(model_folder, "G2.pkl")))
    F2.load_state_dict(torch.load(os.path.join(model_folder, "F2.pkl")))
    G3.load_state_dict(torch.load(os.path.join(model_folder, "G3.pkl")))

    # In[Prepare tensors]
    u_torch = torch.tensor(u)

    # In[Predict]
    def model(u_in):
        y1_lin = G1(u_in)
        y1_nl = F1(y1_lin)
        y2_lin = G2(y1_nl)
        y2_nl = F2(y2_lin)
        y3_lin = G3(y2_nl)

        y_hat = y3_lin
        return y_hat, y1_nl, y1_lin

    y_hat, y1_nl, y1_lin = model(u_torch)

    # In[Detach & organize]
    y_hat = y_hat.detach().numpy()[0, :, :]
    y1_lin = y1_lin.detach().numpy()[0, :, :]
    y1_nl = y1_nl.detach().numpy()[0, :, :]
    y = y[0, :, :]
    u = u[0, :, :]

    # In[Plot]
    e = y - y_hat
    plt.figure()
    plt.plot(t, y, 'k', label="$y$")
    plt.plot(t, y_hat, 'b', label="$\hat y$")
    plt.plot(t, e, 'r', label="$e$")
    plt.legend(loc='upper left')
    plt.grid(True)

    # In[Metrics]
    n_skip = 100
    e_rms = util.metrics.error_rmse(y[n_skip:], y_hat[n_skip:])[0]*scaler_y
    fit_idx = util.metrics.fit_index(y[n_skip:], y_hat[n_skip:])[0]
    r_sq = util.metrics.r_squared(y[n_skip:], y_hat[n_skip:])[0]

    print(f"RMSE: {e_rms:.2E} mm\nFIT:  {fit_idx:.1f}%\nR_sq: {r_sq:.2f}")