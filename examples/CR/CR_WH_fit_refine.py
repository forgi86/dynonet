import os
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
import scipy.io
from examples.CR.common import CRSmallDataset
from torchid.module.LTI import LinearMimo
from torchid.module.static import StaticMimoNonLin, StaticSisoNonLin

import util.metrics

if __name__ == '__main__':

    # In[Set seed for reproducibility]
    np.random.seed(0)
    torch.manual_seed(0)

    # In[Load and save folders]
    model_load_name = None#"model_PWH"
    model_save_name = "model_PWH" #"model_PWH_refined"

    # In[Overall parameters]
    epochs = 100000  # gradient-based optimization steps
    lr = 1e-3  # learning rate
    test_freq = 10  # print message every test_freq iterations
    batch_size = 16

    # In[Constants]
    fs = 256.0  # Sampling frequency (Hz)
    ts = 1/fs
    n_u = 1  # number of inputs
    n_y = 1  # number of outputs

    # In[Load dataset]
    mat_data = scipy.io.loadmat(os.path.join("data", "Benchmark_EEG_small", "Benchmark_EEG_small.mat"))

    # In [Extract data]
    u = mat_data['data'][0][0][0].astype(np.float32)    # input, normalized handle angle. Tensor structure: (B, R, N)
    y = mat_data['data'][0][0][1].astype(np.float32)    # output, ICA component with highest SNR (normalized)

    B = u.shape[0]  # number of participants (10)
    R = u.shape[1]  # number of realizations (7)
    T = u.shape[2]  # time index (256)

    time_vec = np.arange(T) * ts

    # Add a channel index (even though signals are scalar)
    u = u[..., None]
    y = y[..., None]

    # In[Train and test datasets]
    u_train = u[:, :-1, :, :]
    y_train = y[:, :-1, :, :]

    y_test = y[:, [-1], :, :]
    u_test = u[:, [-1], :, :]

    # In[Create Dataset and DataLoader objects]
    train_ds = CRSmallDataset(u_train, y_train)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    test_ds = CRSmallDataset(u_test, y_test)
    # In[Set-up model]

    # First linear model
    in_channels_1 = 1
    out_channels_1 = 6
    nb_1 = 3
    na_1 = 3
    G1 = LinearMimo(in_channels_1, out_channels_1, nb_1, na_1)

    # Static non-linearity
    F1 = StaticMimoNonLin(out_channels_1, 6)

    # Second linear model
    in_channels_2 = 6
    out_channels_2 = 1
    nb_2 = 3
    na_2 = 3
    G2 = LinearMimo(in_channels_2, out_channels_2, nb_2, na_2)

    # Save initial parameters (if available)
    if model_load_name is not None:
        model_folder = os.path.join("models", model_load_name)
        G1.load_state_dict(torch.load(os.path.join(model_folder, "G1.pkl")))
        F1.load_state_dict(torch.load(os.path.join(model_folder, "F1.pkl")))
        G2.load_state_dict(torch.load(os.path.join(model_folder, "G2.pkl")))

    def model(u_in):
        y_lin_1 = G1(u_in)
        y_nl_1 = F1(y_lin_1)
        y_lin_2 = G2(y_nl_1)
        y_hat = y_lin_2
        return y_hat, y_lin_2, y_nl_1, y_lin_1

    # In[Setup optimizer]
    optimizer = torch.optim.Adam([
        {'params': G1.parameters(),    'lr': lr},
        {'params': F1.parameters(), 'lr': lr},
        {'params': G2.parameters(), 'lr': lr},
    ], lr=lr)

    # In[Training loop]
    LOSS_ITR = []
    LOSS_TRAIN = []
    start_time = time.time()
    for epoch in range(epochs):
        #loop = tqdm.tqdm(train_dl)
        train_loss = torch.tensor(0.0)
        for ub, yb in train_dl:

            bs = ub.shape[0]
            # Clean old gradients
            optimizer.zero_grad()

            # Simulate
            y_hat, y_lin_2, y_nl_1, y_lin_1 = model(ub)

            # Compute fit loss
            err_fit = yb - y_hat
            loss_fit = torch.mean(err_fit**2)
            loss = loss_fit

            # Statistics
            with torch.no_grad():
                train_loss += loss * bs
                LOSS_ITR.append(loss.item())

            # Optimize
            loss.backward()
            optimizer.step()

        # Metrics
        with torch.no_grad():
            train_loss = train_loss / len(train_ds)
            LOSS_TRAIN.append(train_loss.item())
            if epoch % test_freq == 0:
                with torch.no_grad():
                    RMSE = torch.sqrt(train_loss)
                print(f'Epoch {epoch} | Train Loss {train_loss:.6f} | RMSE:{RMSE:.4f}')

    train_time = time.time() - start_time
    print(f"\nTrain time: {train_time:.2f}")  # 182 seconds


    # In[Save model]

    if model_save_name is not None:
        model_folder = os.path.join("models", model_save_name)
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        torch.save(G1.state_dict(), os.path.join(model_folder, "G1.pkl"))
        torch.save(F1.state_dict(), os.path.join(model_folder, "F1.pkl"))
        torch.save(G2.state_dict(), os.path.join(model_folder, "G2.pkl"))

    # In[Simulate for all batches]
    y_train_hat, _, _, _ = model(train_ds._u)
    y_train_hat = y_train_hat.detach().numpy().reshape(B, R-1, T, 1)

    y_test_hat, _, _, _ = model(test_ds._u)
    y_test_hat = y_test_hat.detach().numpy().reshape(B, 1, T, 1)

    # In[Plot Loss]
    plt.figure()
    plt.plot(LOSS_ITR)

    # In[Plot signals]
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(time_vec, y_train[0, 0, :])
    ax[0].plot(time_vec, y_train_hat[0, 0, :])


    # In[Metrics]

    e_rms_train = util.metrics.error_rmse(y_train, y_train_hat, time_axis=-2)
    r_sq_train = util.metrics.r_squared(y_train, y_train_hat, time_axis=-2)

#    print(f"RMSE: {e_rms:.2f}mV\nMAE: {mae:.2f}mV\nFIT:  {fit_idx:.1f}%\nR_sq: {r_sq:.1f}\nRMSU: {u_rms:.2f}mV")
