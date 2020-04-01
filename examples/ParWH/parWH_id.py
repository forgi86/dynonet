import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.utils.data
from torchid.module.LTI import LinearMimo
from examples.ParWH.common import  ParallelWHDataset
from examples.ParWH.common import StaticMimoNonLin
from tqdm import tqdm

import matplotlib.pyplot as plt
import time
import util.metrics

if __name__ == '__main__':

    lr = 1e-2
    epochs = 2000
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

        G1.a_coeff[:] = torch.randn((out_channels_1, in_channels_1, na_1))*0.01
        G1.b_coeff[:] = torch.randn((out_channels_1, in_channels_1, nb_1))*0.01

        G2.a_coeff[:] = torch.randn((out_channels_2, in_channels_2, na_2))*0.01
        G2.b_coeff[:] = torch.randn((out_channels_2, in_channels_2, nb_2))*0.01

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

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.1, min_lr=1e-6, verbose=True)

    # In[Setup data loaders]
    train_ds = ParallelWHDataset(data_train)  # 19*5=95 samples
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    valid_ds = ParallelWHDataset(data_val)  # 5 samples
    valid_dl = torch.utils.data.DataLoader(valid_ds, batch_size=batch_size, shuffle=True)

    # In[Training loop]
    LOSS_ITR = []
    LOSS_TRAIN = []
    LOSS_VAL = []
    start_time = time.time()
    for epoch in range(epochs):
        #loop = tqdm(train_dl)
        G1.train()
        F_nl.train()
        G1.train()

        train_loss = torch.tensor(0.0)
        for ub, yb in train_dl:

            bs = ub.shape[0]  # length of this batch (normally batch_size, except the last of the epoch)

            # Empty old gradients
            optimizer.zero_grad()

            # Simulate
            y_lin_1 = G1(ub, y0_1, u0_1)
            y_nl_1 = F_nl(y_lin_1)
            y_lin_2 = G2(y_nl_1, y0_2, u0_2)

            y_hat = y_lin_2

            # Compute fit loss
            err_fit = yb[..., n_skip:, :] - y_hat[..., n_skip:, :]
            loss = torch.mean(err_fit**2)

            with torch.no_grad():
                train_loss += loss * bs

            # Statistics
            LOSS_ITR.append(loss.item())

            # Optimize
            loss.backward()
            optimizer.step()


        # Model in evaluation mode
        G1.eval()
        F_nl.eval()
        G2.eval()

        # Metrics
        with torch.no_grad():

            train_loss = train_loss / len(train_ds)
            RMSE_train = 1000*torch.sqrt(train_loss)
            LOSS_TRAIN.append(train_loss.item())

            val_loss = torch.tensor(0.0)
            for ub, yb in valid_dl:

                bs = ub.shape[0]
                # Simulate
                y_lin_1 = G1(ub, y0_1, u0_1)
                y_nl_1 = F_nl(y_lin_1)
                y_lin_2 = G2(y_nl_1, y0_2, u0_2)

                y_hat = y_lin_2

                # Compute fit loss
                err_val = yb[..., n_skip:, :] - y_hat[..., n_skip:, :]
                val_loss += torch.mean(err_val ** 2) * bs

            val_loss = val_loss / len(valid_ds)
            LOSS_VAL.append(val_loss.item())

        scheduler.step(val_loss)

        print(f'Epoch {epoch} | Train Loss {train_loss:.6f} | Validation Loss {val_loss:.6f} | Train RMSE: {RMSE_train:.2f}')




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
    plt.plot(LOSS_ITR)
    plt.grid(True)


    plt.figure()
    plt.plot(LOSS_TRAIN, 'r')
    plt.plot(LOSS_VAL, 'g')
    plt.grid(True)

    # In[Metrics]

#    idx_metric = range(0, N_per_period)
#    e_rms = util.metrics.error_rmse(y[idx_metric], y_hat_np[idx_metric])
#    fit_idx = util.metrics.fit_index(y[idx_metric], y_hat_np[idx_metric])
#    r_sq = util.metrics.r_squared(y[idx_metric], y_hat_np[idx_metric])

#    print(f"RMSE: {e_rms:.4f}V\nFIT:  {fit_idx:.1f}%\nR_sq: {r_sq:.1f}")