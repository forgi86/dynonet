import torch
import pandas as pd
import numpy as np
import os
from torchid.module.LTI import LinearMimo
import matplotlib.pyplot as plt
import torch.nn as nn
import control
import util.metrics


class StaticNonLin(nn.Module):

    def __init__(self):
        super(StaticNonLin, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(1, 20),  # 2 states, 1 input
            nn.ReLU(),
            nn.Linear(20, 1)
        )

        #for m in self.net.modules():
        #    if isinstance(m, nn.Linear):
        #        nn.init.normal_(m.weight, mean=0, std=1e-3)
        #        nn.init.constant_(m.bias, val=0)

    def forward(self, y_lin):
        y_nl = self.net(y_lin)
        return y_nl


if __name__ == '__main__':

    #model_name = "model_WH_trainedOnTest"
    dataset_name = "WH_TestDataset.csv"
    model_name = "model_WH"
    #dataset_name = 'WH_CombinedZeroMultisineSinesweep.csv'  # test on train!

    # Settings
    n_b = 3
    n_a = 3
    n_fit = 100000
    n_batch = 1

    # Column names in the dataset
    COL_F = 'fs'
    TAG_R = 'r'
    TAG_U = 'u'
    TAG_Y = 'y'

    COL_U = 'u0'
    COL_Y = 'y0'

    # Load dataset
    df_X = pd.read_csv(os.path.join("data", dataset_name))
    idx_u = list(df_X.keys()).index('u')
    n_real = idx_u
    col_names = [TAG_R + str(i) for i in range(n_real)] \
                + [TAG_U + str(i) for i in range(n_real)] \
                + [TAG_Y + str(i) for i in range(n_real)] + [COL_F] + ['?']
    df_X.columns = col_names

    # Extract data
    y_meas = np.array(df_X[[COL_Y]], dtype=np.float32)
    u = np.array(df_X[[COL_U]], dtype=np.float32)
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

    # Second-order dynamical system
    G1 = LinearMimo(1, 1, n_b, n_a)
    G2 = LinearMimo(1, 1, n_b, n_a)
    # Static sandwitched non-linearity
    F_nl = StaticNonLin()


    # Create model parameters
    model_folder = os.path.join("models", model_name)
    G1.load_state_dict(torch.load(os.path.join(model_folder, "G1.pkl")))
    F_nl.load_state_dict(torch.load(os.path.join(model_folder, "F1.pkl")))
    G2.load_state_dict(torch.load(os.path.join(model_folder, "G2.pkl")))

    # In[Predict]

    u_torch = torch.tensor(u[None, :])
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
    plt.grid(True)

    # In[Inspect linear model]

#    G1_sys = control.TransferFunction(G1.b_coeff.detach().numpy(), np.r_[1.0, G1.f_coeff.detach().numpy()], ts)
#    plt.figure()
#    mag_G1, phase_G1, omega_G1 = control.bode(G1_sys)

#    G2_sys = control.TransferFunction(G2.b_coeff.detach().numpy(), np.r_[1.0, G2.f_coeff.detach().numpy()], ts)
#    plt.figure()
#    mag_G2, phase_G2, omega_G2 = control.bode(G2_sys)

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

    e_rms = 1000*util.metrics.error_rmse(y_meas, y_hat)[0]
    fit_idx = util.metrics.fit_index(y_meas, y_hat)[0]
    r_sq = util.metrics.r_squared(y_meas, y_hat)[0]

    print(f"RMSE: {e_rms:.2f} mV\nFIT:  {fit_idx:.1f}%\nR_sq: {r_sq:.1f}")


