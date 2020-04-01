import torch
import pandas as pd
import numpy as np
import os
from torchid.module.LTI import LinearMimo
import matplotlib.pyplot as plt
import time
import torch.nn as nn

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

    # Set seed for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)

    # Settings
    add_noise = True
    lr = 1e-3
    num_iter = 10000
    test_freq = 1
    n_fit = 100000
    n_batch = 1
    n_b = 3
    n_a = 3
    n_u = 1
    n_y = 1

    # Column names in the dataset
    COL_F = 'fs'
    TAG_R = 'r'
    TAG_U = 'u'
    TAG_Y = 'y'

    # Load dataset
    df_names = ['WH_CombinedZeroMultisineSinesweep', 'WH_MultisineFadeOut', 'WH_SineInput_meas', 'WH_Triangle2_meas', 'WH_ZeroMeas']

    df_list = []
    for dataset_name in df_names:
        dataset_filename = dataset_name + '.csv'
        df_dataset = pd.read_csv(os.path.join("data", dataset_filename))
        idx_u = list(df_dataset.keys()).index('u')
        n_real = idx_u
        col_names = [TAG_R + str(i) for i in range(n_real)] + [TAG_U + str(i) for i in range(n_real)] + [TAG_Y + str(i) for i in range(n_real)] + [COL_F] + ['?']
        df_dataset.columns = col_names
        df_list.append(df_dataset)
    dict_df = dict(zip(df_names, df_list))

    dict_data = dict()
    for key in dict_df:
        df_dataset = dict_df[key]
        idx_u = list(df_dataset.keys()).index('u0')
        n_real = idx_u
        N = df_dataset.shape[0]
        COL_U = [TAG_U + str(i) for i in range(n_real)]
        COL_Y = [TAG_Y + str(i) for i in range(n_real)]

        data_u = np.array(df_dataset[COL_U]).reshape(N, n_real, 1)
        data_u = np.ascontiguousarray(data_u.transpose(1, 0, 2)).astype(np.float32)
        data_y = np.array(df_dataset[COL_Y]).reshape(N, n_real, 1)
        data_y = np.ascontiguousarray(data_y.transpose(1, 0, 2)).astype(np.float32)

        dict_data[key] = (n_real, data_u, data_y)

#    df_X = pd.read_csv(os.path.join("data", "WH_CombinedZeroMultisineSinesweep.csv"))



    # Second-order dynamical system custom defined
    G1 = LinearMimo(1, 1, n_b, n_a)
    G2 = LinearMimo(1, 1, n_b, n_a)

    with torch.no_grad():

        G1.a_coeff[:] = torch.randn((1, 1, n_a))*0.01
        G1.b_coeff[:] = torch.randn((1, 1, n_b))*0.01

        G2.a_coeff[:] = torch.randn((1, 1, n_a))*0.01
        G2.b_coeff[:] = torch.randn((1, 1, n_a))*0.01

    # Static sandwitched non-linearity
    F_nl = StaticNonLin()

    # Setup optimizer
    optimizer = torch.optim.Adam([
        {'params': G1.parameters(), 'lr': lr},
        {'params': G2.parameters(), 'lr': lr},
        {'params': F_nl.parameters(), 'lr': lr},
    ], lr=lr)

    # In[Train]
    LOSS = []
    start_time = time.time()
    for itr in range(0, num_iter):

        dataset_name = np.random.choice(df_names)
        batch_size, data_u, data_y = dict_data[dataset_name]
        u_fit_torch = torch.tensor(data_u)
        y_fit_torch = torch.tensor(data_y)

        optimizer.zero_grad()

        # Simulate
        y1_lin = G1(u_fit_torch)
        y1_nl = F_nl(y1_lin)
        y_hat = G2(y1_nl)

        # Compute fit loss
        err_fit = y_fit_torch - y_hat
        loss_fit = torch.mean(err_fit**2)
        loss = loss_fit

        LOSS.append(loss.item())
        if itr % test_freq == 0:
            with torch.no_grad():
                RMSE = torch.sqrt(loss)
            print(f'Iter {itr} | Fit Loss {loss_fit:.6f} | RMSE:{RMSE:.4f}')

        # Optimize
        loss.backward()

        if itr == 100:
            pass
        optimizer.step()

    train_time = time.time() - start_time
    print(f"\nTrain time: {train_time:.2f}") # 182 seconds

    # In[Save model]
    if not os.path.exists("models"):
        os.makedirs("models")
    model_filename = "model_WH"

    torch.save(G1.state_dict(), os.path.join("models", f"{model_filename}_G1.pkl"))
    torch.save(F_nl.state_dict(), os.path.join("models", f"{model_filename}_F_nl.pkl"))
    torch.save(G2.state_dict(), os.path.join("models", f"{model_filename}_G2.pkl"))

    # In[To numpy]

    y_hat = y_hat.detach().numpy()
    y1_lin = y1_lin.detach().numpy()
    y1_nl = y1_nl.detach().numpy()

    # In[Plot]
    #plt.figure()
    #plt.plot(t_fit, y_fit, 'k', label="$y$")
    #plt.plot(t_fit, y_hat, 'b', label="$\hat y$")
    #plt.legend()

    plt.figure()
    plt.plot(LOSS)
    plt.grid(True)

    # In[Plot static non-linearity]

    y1_lin_min = np.min(y1_lin)
    y1_lin_max = np.max(y1_lin)

    in_nl = np.arange(y1_lin_min, y1_lin_max, (y1_lin_max- y1_lin_min)/1000).astype(np.float32).reshape(-1, 1)

    with torch.no_grad():
        out_nl = F_nl(torch.as_tensor(in_nl))

    plt.figure()
    plt.plot(in_nl, out_nl, 'b')
    plt.plot(in_nl, out_nl, 'b')
    #plt.plot(y1_lin, y1_nl, 'b*')
    plt.xlabel('Static non-linearity input (-)')
    plt.ylabel('Static non-linearity input (-)')
    plt.grid(True)

    # In[Plot]
    e_rms = util.metrics.error_rmse(y_hat, y_fit)[0]
    print(f"RMSE: {e_rms:.2f}")





