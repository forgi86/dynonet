import os
import numpy as np
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import scipy.io
import tqdm
from torchid.module.LTI import LinearMimo


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


if __name__ == '__main__':

    # Set seed for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)

    # Overall parameters
    epochs = 10000  # gradient-based optimization steps
    lr = 1e-4  # learning rate
    test_freq = 10  # print message every test_freq iterations
    batch_size = 16

    # Load dataset
    mat_data = scipy.io.loadmat(os.path.join("data", "Benchmark_EEG_small", "Benchmark_EEG_small.mat"))

    u = mat_data['data'][0][0][0].astype(np.float32)    # input, normalized handle angle. Tensor structure: (B, R, N)
    y = mat_data['data'][0][0][1].astype(np.float32)    # output, ICA component with highest SNR (normalized)
    #msg = mat_data['data'][0][0][1]  # msg


    B = u.shape[0]  # number of participants (10)
    R = u.shape[1]  # number of realizations (7)
    T = u.shape[2]  # time index (256)

    fs = 256.0  # Sampling frequency (Hz)
    ts = 1/fs
    time_vec = np.arange(T) * ts

    n_u = 1  # number of inputs
    n_y = 1  # number of outputs


    class CRSmallDataset(torch.utils.data.Dataset):
        """Face Landmarks dataset."""

        def __init__(self, u, y):
            """
            Args:
                u (torch.Tensor): Tensor with input data. # B R T
                y (torch.Tensor): Tensor with input data. # B R T
            """
            self.u = torch.tensor(u)
            self.y = torch.tensor(y)

            self.B, self.R, self.T = self.u.shape
            self.len = self.B * self.R

            self._u = self.u.view(self.len, self.T, 1)
            self._y = self.y.view(self.len, self.T, 1)

        def __len__(self):
            return self.len

        def __getitem__(self, idx):
            return self._u[idx, :], self._y[idx, :]

    train_ds = CRSmallDataset(u, y)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)


    # In[Set-up model]

    # First linear section
    in_channels_1 = 1
    out_channels_1 = 2
    nb_1 = 3
    na_1 = 3
    y0_1 = torch.zeros((batch_size, na_1), dtype=torch.float)
    u0_1 = torch.zeros((batch_size, nb_1), dtype=torch.float)
    G1 = LinearMimo(in_channels_1, out_channels_1, nb_1, na_1)

    # Non-linear section
    F_nl = StaticNonLin()

    # Second linear section
    in_channels_2 = 2
    out_channels_2 = 1
    nb_2 = 3
    na_2 = 3
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

    # In[Setup optimizer]
    optimizer = torch.optim.Adam([
        {'params': G1.parameters(),    'lr': lr},
        {'params': F_nl.parameters(), 'lr': lr},
        {'params': G2.parameters(), 'lr': lr},
    ], lr=lr)

    # In[Training loop]
    LOSS = []
    start_time = time.time()
    for epoch in range(epochs):
        #loop = tqdm.tqdm(train_dl)
        for u_torch, y_meas_torch in train_dl:

            # Empty old gradients
            optimizer.zero_grad()

            # Simulate
            y_lin_1 = G1(u_torch, y0_1, u0_1)
            y_nl_1 = F_nl(y_lin_1)
            y_lin_2 = G2(y_nl_1, y0_2, u0_2)

            y_hat = y_lin_2

            # Compute fit loss
            err_fit = y_meas_torch - y_hat
            loss_fit = torch.mean(err_fit**2)
            loss = loss_fit

            # Statistics
            LOSS.append(loss.item())

            # Optimize
            loss.backward()
            optimizer.step()

        if epoch % test_freq == 0:
            with torch.no_grad():
                RMSE = torch.sqrt(loss)
            print(f'Epoch {epoch} | Fit Loss {loss_fit:.6f} | RMSE:{RMSE:.4f}')

    train_time = time.time() - start_time
    print(f"\nTrain time: {train_time:.2f}")  # 182 seconds


    # In[Save model]
    if not os.path.exists("models"):
        os.makedirs("models")
    model_filename = "model_WH"

    torch.save(G1.state_dict(), os.path.join("models", f"{model_filename}_G1.pkl"))
    torch.save(F_nl.state_dict(), os.path.join("models", f"{model_filename}_F_nl.pkl"))
    torch.save(G2.state_dict(), os.path.join("models", f"{model_filename}_G2.pkl"))

    # In[Plot Loss]
    plt.figure()
    plt.plot(LOSS)
