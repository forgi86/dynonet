import os
import numpy as np
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import scipy.io
from examples.CR.common import CRSmallDataset
from torchid.module.lti import MimoLinearDynamicOperator
from torchid.module.static import MimoStaticNonLin, SisoStaticNonLin

if __name__ == '__main__':

    # Set seed for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)

    # Overall parameters
    epochs = 100000  # gradient-based optimization steps
    lr = 1e-3  # learning rate
    test_freq = 10  # print message every test_freq iterations
    batch_size = 16

    # Load dataset
    mat_data = scipy.io.loadmat(os.path.join("../data", "Benchmark_EEG_small", "Benchmark_EEG_small.mat"))

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



    train_ds = CRSmallDataset(u, y)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)


    # In[Set-up model]

    # First linear section
    in_channels_1 = 1
    out_channels_1 = 6
    nb_1 = 3
    na_1 = 3
    y0_1 = torch.zeros((batch_size, na_1), dtype=torch.float)
    u0_1 = torch.zeros((batch_size, nb_1), dtype=torch.float)
    G1 = MimoLinearDynamicOperator(in_channels_1, out_channels_1, nb_1, na_1)

    # Non-linear section
    F_nl = MimoStaticNonLin(out_channels_1, 6)

    # Second linear section
    in_channels_2 = 6
    out_channels_2 = 1
    nb_2 = 3
    na_2 = 3
    y0_2 = torch.zeros((batch_size, na_2), dtype=torch.float)
    u0_2 = torch.zeros((batch_size, nb_2), dtype=torch.float)
    G2 = MimoLinearDynamicOperator(in_channels_2, out_channels_2, nb_2, na_2)

    # In[Setup optimizer]
    optimizer = torch.optim.Adam([
        {'params': G1.parameters(),    'lr': lr},
        {'params': F_nl.parameters(), 'lr': lr},
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
            # Empty old gradients
            optimizer.zero_grad()

            # Simulate
            y_lin_1 = G1(ub, y0_1, u0_1)
            y_nl_1 = F_nl(y_lin_1)
            y_lin_2 = G2(y_nl_1, y0_2, u0_2)

            y_hat = y_lin_2

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
    model_name = "model_PWH"
    model_folder = os.path.join("../models", model_name)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    torch.save(G1.state_dict(), os.path.join(model_folder, "G1.pkl"))
    torch.save(F_nl.state_dict(), os.path.join(model_folder, "F_nl.pkl"))
    torch.save(G2.state_dict(), os.path.join(model_folder, "G2.pkl"))

    # In[Plot Loss]
    plt.figure()
    plt.plot(LOSS_ITR)


