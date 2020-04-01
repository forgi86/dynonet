import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.utils.data
from torchid.module.LTI import LinearMimo
from tqdm import tqdm

import matplotlib.pyplot as plt
import time
import util.metrics


class StaticNonLin(nn.Module):

    def __init__(self):
        super(StaticNonLin, self).__init__()

        self.net_1 = nn.Sequential(
            nn.Linear(1, 20),  # 2 states, 1 input
            nn.ReLU(),
            nn.Linear(20, 1)
        )

        self.net_2 = nn.Sequential(
            nn.Linear(1, 20),  # 2 states, 1 input
            nn.ReLU(),
            nn.Linear(20, 1)
        )

    def forward(self, u_lin):

        y_nl_1 = self.net_1(u_lin[..., [0]])  # Process blocks individually
        y_nl_2 = self.net_2(u_lin[..., [1]])  # Process blocks individually
        y_nl = torch.cat((y_nl_1, y_nl_2), dim=-1)

        return y_nl


class StaticMimoNonLin(nn.Module):

    def __init__(self):
        super(StaticMimoNonLin, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 20),  # 2 states, 1 input
            nn.ReLU(),
            nn.Linear(20, 2)
        )

    def forward(self, u_lin):

        y_nl = self.net(u_lin)  # Process blocks individually
        return y_nl


if __name__ == '__main__':

    lr = 1e-4
    epochs = 1000
    test_freq = 1 # print a msg every epoch
    batch_size = 16

    n_amp = 5 #  number of different amplitudes
    n_real = 20  # number of random phase multisine realizations
    N_per_period = 16384  # number of samples per period
    P = 2  # number of periods
    seq_len = N_per_period * P  # data points per realization
    n_skip = 100  # skip first n_skip points in loss evaluation

    # Column names in the dataset
    TAG_U = 'u'
    TAG_Y = 'y'
    DF_COL = ['amplitude', 'fs', 'lines'] + [TAG_U + str(i) for i in range(n_real)] + [TAG_Y + str(i) for i in range(n_real)] + ['?']

    # Load dataset
    dataset_list_level = ['ParWHData_Estimation_Level' + str(i) for i in range(1, n_amp+1)]

    df_X_lst = []
    for dataset_name in dataset_list_level:
        dataset_filename = dataset_name + '.csv'
        df_Xi = pd.read_csv(os.path.join("data", dataset_filename))
        df_Xi.columns = DF_COL
        df_X_lst.append(df_Xi)

    data_mat = np.empty((n_amp, n_real, seq_len, 2))  # Level, Realization, Time, Feat
    for amp_idx in range(n_amp):
        for real_idx in range(n_real):
            tag_u = 'u' + str(real_idx)
            tag_y = 'y' + str(real_idx)
            df_data = df_X_lst[amp_idx][[tag_u, tag_y]]  #np.array()
            data_mat[amp_idx, real_idx, :, :] = np.array(df_data)

    data_mat = data_mat.astype(np.float32)

    data_train = data_mat[:, :-1, :, :]
    data_val = data_mat[:, [-1], :, :] # use

    #data_mat_torch = torch.tensor(data_mat)  # A, R, T, C


    class ParallelWHDataset(torch.utils.data.Dataset):
        """Face Landmarks dataset."""

        def __init__(self, data):
            """
            Args:
                data (torch.Tensor): Tensor with data organized in.
            """
            self.data = torch.tensor(data)
            self.n_amp, self.n_real, self.seq_len, self.n_channels = data.shape
            self.len = self.n_amp * self.n_real
            self._data = self.data.view(self.n_amp * self.n_real, self.seq_len, self.n_channels)

        def __len__(self):
            return self.len

        def __getitem__(self, idx):
            return self._data[idx, :, [0]], self._data[idx, :, [1]]



#    u_torch = torch.tensor(u[None, :, None],  dtype=torch.float, requires_grad=False)
#    y_meas_torch = torch.tensor(y[None, :, None],  dtype=torch.float, requires_grad=False)

    # In[Set-up model]

    # First linear section
    in_channels_1 = 1
    out_channels_1 = 2
    nb_1 = 6  # was 3 before...
    na_1 = 6
    y0_1 = torch.zeros((batch_size, na_1), dtype=torch.float)
    u0_1 = torch.zeros((batch_size, nb_1), dtype=torch.float)
    G1 = LinearMimo(in_channels_1, out_channels_1, nb_1, na_1)

    # Non-linear section
    F_nl = StaticMimoNonLin()

    # Second linear section
    in_channels_2 = 2
    out_channels_2 = 1
    nb_2 = 6
    na_2 = 6
    y0_2 = torch.zeros((batch_size, na_2), dtype=torch.float)
    u0_2 = torch.zeros((batch_size, nb_2), dtype=torch.float)
    G2 = LinearMimo(in_channels_2, out_channels_2, nb_2, na_2)

    # In[Initialize linear systems]
    with torch.no_grad():
        G1.a_coeff[:, :, 0] = -0.9
        G1.b_coeff[:, :, 0] = 0.1
        G1.b_coeff[:, :, 1] = 0.1

        G2.a_coeff[:, :, 0] = -0.9
        G2.b_coeff[:, :, 0] = 0.1
        G1.b_coeff[:, :, 1] = 0.1

        #G1.a_coeff[0, 0, 0] = -0.9
        #G1.a_coeff[1, 0, 0] = 0.9
        #G1.b_coeff[:, :, 0] = 0.1
        #G1.b_coeff[:, :, 1] = 0.1

        #G2.a_coeff[0, 0, 0] = -0.9
        #G2.a_coeff[0, 1, 0] = 0.9

        #G2.b_coeff[:, :, 0] = 0.1
        #G1.b_coeff[:, :, 1] = 0.1

    # In[Setup optimizer]
    optimizer = torch.optim.Adam([
        {'params': G1.parameters(),    'lr': lr},
        {'params': F_nl.parameters(), 'lr': lr},
        {'params': G2.parameters(), 'lr': lr},
    ], lr=lr)

    # In[Setup data loaders]
    train_ds = ParallelWHDataset(data_train)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_ds = ParallelWHDataset(data_val)
    valid_dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=True)

    # In[Training loop]
    LOSS = []
    start_time = time.time()
    for epoch in range(epochs):
        #loop = tqdm(train_dl)
        G1.train()
        F_nl.train()
        G1.train()

        loss_train = torch.tensor(0.0)
        for u_torch, y_meas_torch in train_dl:

            # Empty old gradients
            optimizer.zero_grad()

            # Simulate
            y_lin_1 = G1(u_torch, y0_1, u0_1)
            y_nl_1 = F_nl(y_lin_1)
            y_lin_2 = G2(y_nl_1, y0_2, u0_2)

            y_hat = y_lin_2

            # Compute fit loss
            err_fit = y_meas_torch[..., n_skip:, :] - y_hat[..., n_skip:, :]
            loss_fit = torch.mean(err_fit**2)
            loss = loss_fit

            with torch.no_grad():
                loss_train += loss

            # Statistics
            LOSS.append(loss.item())

            # Optimize
            loss.backward()
            optimizer.step()

        loss_train = loss_train/len(train_ds)
        #loop.set_description('Epoch {}/{}'.format(epoch + 1, epochs))
        #loop.set_postfix(loss=loss.item(), mse=0.9)

        #if epoch % test_freq == 0:
        #    with torch.no_grad():
        #        RMSE = 1000*torch.sqrt(loss)
        #    print(f'Epoch {epoch} | Fit Loss {loss_fit:.6f} | RMSE:{RMSE:.2f} mV')


        G1.eval()
        F_nl.eval()
        G2.eval()
        with torch.no_grad():
            loss_val = torch.tensor(0.0)
            for u_torch, y_meas_torch in valid_dl:
                # Simulate
                y_lin_1 = G1(u_torch, y0_1, u0_1)
                y_nl_1 = F_nl(y_lin_1)
                y_lin_2 = G2(y_nl_1, y0_2, u0_2)

                y_hat = y_lin_2

                # Compute fit loss
                err_val = y_meas_torch[..., n_skip:, :] - y_hat[..., n_skip:, :]
                loss_val += torch.mean(err_val ** 2)
            loss_val = loss_val/len(valid_dl)

        print(f'Epoch {epoch} | Train Loss {loss_train:.6f} | Validation Loss {loss_val:.6f}')




    train_time = time.time() - start_time
    print(f"\nTrain time: {train_time:.2f}")  # 182 seconds


    # In[Save model]
    if not os.path.exists("models"):
        os.makedirs("models")
    model_filename = "model_WH"

    torch.save(G1.state_dict(), os.path.join("models", f"{model_filename}_G1.pkl"))
    torch.save(F_nl.state_dict(), os.path.join("models", f"{model_filename}_F_nl.pkl"))
    torch.save(G2.state_dict(), os.path.join("models", f"{model_filename}_G2.pkl"))


    # In[detach]
    y_hat_np = y_hat.detach().numpy()[0, :, 0]

    # In[Plot]
#    fig, ax = plt.subplots(2, 1, sharex=True)
#    ax[0].plot(t, y, 'k', label="$y$")
#    ax[0].plot(t, y_hat_np, 'r', label="$y$")

#    ax[0].legend()
#    ax[0].grid()

#    ax[1].plot(t, u, 'k', label="$u$")
#    ax[1].legend()
#    ax[1].grid()

    plt.figure()
    plt.plot(LOSS)
    plt.grid(True)


    # In[Metrics]

#    idx_metric = range(0, N_per_period)
#    e_rms = util.metrics.error_rmse(y[idx_metric], y_hat_np[idx_metric])
#    fit_idx = util.metrics.fit_index(y[idx_metric], y_hat_np[idx_metric])
#    r_sq = util.metrics.r_squared(y[idx_metric], y_hat_np[idx_metric])

#    print(f"RMSE: {e_rms:.4f}V\nFIT:  {fit_idx:.1f}%\nR_sq: {r_sq:.1f}")