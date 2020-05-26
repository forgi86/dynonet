import os
import h5py
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from torchid.module.lti import MimoLinearDynamicalOperator
import util.metrics
from torchid.module.static import MimoStaticNonLinearity

if __name__ == '__main__':

    matplotlib.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})

    # In[Settings]
    #h5_filename = 'train.h5'
    h5_filename = 'test.h5'
    #signal_name = 'multisine'
    signal_name = 'multisine'
    #signal_name = 'sinesweep' # available in test
    model_name = "model_BW"


    # In[Load dataset]

    h5_data = h5py.File(os.path.join("data", "Test signals", h5_filename), 'r')
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

    # Model blocks
    G1 = MimoLinearDynamicalOperator(1, 8, n_b=3, n_a=3, n_k=1)
    F1 = MimoStaticNonLinearity(8, 4, n_hidden=10)  #torch.nn.ReLU() #StaticMimoNonLin(3, 3, n_hidden=10)
    G2 = MimoLinearDynamicalOperator(4, 4, n_b=3, n_a=3)
    F2 = MimoStaticNonLinearity(4, 1, n_hidden=10)
    G3 = MimoLinearDynamicalOperator(1, 1, n_b=2, n_a=2, n_k=1)

    # Load identified model parameters
    model_folder = os.path.join("models", model_name)
    G1.load_state_dict(torch.load(os.path.join(model_folder, "G1.pkl")))
    F1.load_state_dict(torch.load(os.path.join(model_folder, "F1.pkl")))
    G2.load_state_dict(torch.load(os.path.join(model_folder, "G2.pkl")))
    F2.load_state_dict(torch.load(os.path.join(model_folder, "F2.pkl")))
    G3.load_state_dict(torch.load(os.path.join(model_folder, "G3.pkl")))

    # Model structure
    def model(u_in):
        y1_lin = G1(u_in)
        y1_nl = F1(y1_lin)
        y2_lin = G2(y1_nl)
        y_branch1 = F2(y2_lin)

        y_branch2 = G3(u_in)
        y_hat = y_branch1 + y_branch2
        return y_hat

    # In[Simulate]
    u_torch = torch.tensor(u)
    y_hat = model(u_torch)

    # In[Detach & organize]
    y_hat = y_hat.detach().numpy()[0, :, :]
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
    n_skip = 300
    e_rms = util.metrics.error_rmse(scaler_y*y[n_skip:], scaler_y*y_hat[n_skip:])[0]
    fit_idx = util.metrics.fit_index(y[n_skip:], y_hat[n_skip:])[0]
    r_sq = util.metrics.r_squared(y[n_skip:], y_hat[n_skip:])[0]

    print(f"RMSE: {e_rms:.2E} mm\nFIT:  {fit_idx:.1f}%\nR_sq: {r_sq:.2f}")


    # In[Plot for paper]
    t_test_start = 5900
    len_plot = 400

    plt.figure(figsize=(4, 3))
    plt.plot(t[t_test_start:t_test_start+len_plot], y[t_test_start:t_test_start+len_plot], 'k', label="$\mathbf{y}^{\mathrm{meas}}$")
    plt.plot(t[t_test_start:t_test_start+len_plot], y_hat[t_test_start:t_test_start+len_plot], 'b--', label="$\mathbf{y}$")
    plt.plot(t[t_test_start:t_test_start+len_plot], e[t_test_start:t_test_start+len_plot], 'r', label="$\mathbf{e}$")
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement (mm)')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.grid(True)
    plt.savefig('BW_timetrace.pdf')