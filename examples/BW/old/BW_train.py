import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
from torchid.module.LTI import LinearMimo
from torchid.module.static import StaticSisoNonLin

if __name__ == '__main__':

    # In[Settings]
    h5_filename = 'train.h5'
    #h5_filename = 'test.h5'
    signal_name = 'multisine'
    #signal_name = 'sinesweep' # available in test

    n_b = 3
    n_a = 3
    n_fit = 100000
    n_batch = 1
    lr = 1e-3
    num_iter = 400000
    msg_freq = 100

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
    t = np.arange(N)*fs


    # In[Scale data]
    y = y/np.std(y)
    u = u/np.std(u)

    # In[Data to float 32]
    y = y.astype(np.float32)
    u = u.astype(np.float32)
    t = t.astype(np.float32)

    # In[Instantiate models]

    # Second-order dynamical system
    G1 = LinearMimo(1, 1, n_b, n_a)
    G2 = LinearMimo(1, 1, n_b, n_a)

    # Static sandwitched non-linearity
    F_nl = StaticSisoNonLin()

    # Setup optimizer
    optimizer = torch.optim.Adam([
        {'params': G1.parameters(), 'lr': lr},
        {'params': G2.parameters(), 'lr': lr},
        {'params': F_nl.parameters(), 'lr': lr},
    ], lr=lr)

    # In[Prepare tensors]
    u_fit_torch = torch.tensor(u)
    y_fit_torch = torch.tensor(y)

    # In[Train]
    LOSS = []
    start_time = time.time()
    for itr in range(0, num_iter):

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
        if itr % msg_freq == 0:
            with torch.no_grad():
                RMSE = torch.sqrt(loss)
            print(f'Iter {itr} | Fit Loss {loss_fit:.6f} | RMSE:{RMSE:.4f}')

        # Optimize
        loss.backward()
        optimizer.step()

    train_time = time.time() - start_time
    print(f"\nTrain time: {train_time:.2f}") # 182 seconds

    # In[Save model]

    model_name = "model_BW_1"
    model_folder = os.path.join("models", model_name)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    torch.save(G1.state_dict(), os.path.join(model_folder, "G1.pkl"))
    torch.save(F_nl.state_dict(), os.path.join(model_folder, "F_nl.pkl"))
    torch.save(G2.state_dict(), os.path.join(model_folder, "G2.pkl"))

    # In[Detach tensors]

    y_hat = y_hat.detach().numpy()
    y1_lin = y1_lin.detach().numpy()
    y1_nl =  y1_nl.detach().numpy()

    # In[Plot signals]

    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(t, y[0, :, 0], label='$y$')
    ax[0].plot(t, y_hat[0, :, 0], label='$\hat y$')
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Displacement (mm)')
    ax[0].grid(True)
    ax[0].legend()
    ax[1].plot(t, y[0, :, 0])
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('Force (N)')
    ax[1].grid(True)
    #ax[1].legend()

    # In[Plot loss]
    fig, ax = plt.subplots()
    ax.plot(LOSS)
    plt.grid(True)

