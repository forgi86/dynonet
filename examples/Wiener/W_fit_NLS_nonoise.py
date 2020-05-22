import numpy as np
import torch
import pyro
import pyro.distributions as dist
import scipy.signal
import matplotlib.pyplot as plt
import pandas as pd
import os
import h5py
from torchid.module.lti import SisoLinearDynamicOperator
from torchid.module.static import SisoStaticNonLin
import time

if __name__ == '__main__':

    n_a = 1
    n_b = 1
    n_k = 1

    num_iter = 20000
    test_freq = 100
    lr = 1e-3

    #model_name = 'ML_noise'
    model_name = 'NLS_nonoise'
    dataset_name = 'train_nonoise'
    # In[Load data]
    filename = os.path.join('data', 'dataset.h5')
    h5_data = h5py.File(filename, 'r')
    u = np.array(h5_data[dataset_name]['u'])
    y = np.array(h5_data[dataset_name]['y'])

#    y = (y - np.mean(y[[0], :, :], axis=-2))/(np.std(y[[0], :, :], axis=-2))
    batch_size = u.shape[0]
    seq_len = u.shape[1]
    n_u = u.shape[2]
    n_y = y.shape[2]

    # In[To tensors]
    u_torch = torch.tensor(u, dtype=torch.float32)
    y_torch = torch.tensor(y, dtype=torch.float32)

    # In[Deterministic model]
    G = SisoLinearDynamicOperator(n_b, n_a, n_k=n_k)
    F = SisoStaticNonLin(n_hidden=10)

    # In[Log-likelihood]


    optimizer = torch.optim.Adam([
        {'params': G.parameters(),    'lr': lr},
        {'params': F.parameters(), 'lr': lr},
    ], lr=lr)

    # In[Train]
    LOSS = []
    start_time = time.time()
    for itr in range(0, num_iter):

        optimizer.zero_grad()

        # Simulate
        y_lin = G(u_torch)
        y_nl = F(y_lin)
        y_hat = y_nl

        # Compute fit loss
        err_fit = y_torch - y_hat
        loss_fit = torch.mean(err_fit**2)
        loss = loss_fit

        LOSS.append(loss.item())
        if itr % test_freq == 0:
            print(f'Iter {itr} | Fit Loss {loss:.4f}')

        # Optimize
        loss.backward()
        optimizer.step()

    train_time = time.time() - start_time


    # In[Save model]
    model_folder = os.path.join("models", model_name)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    torch.save(G.state_dict(), os.path.join(model_folder, "G.pkl"))
    torch.save(F.state_dict(), os.path.join(model_folder, "F.pkl"))


    # In[Detach]
    y_hat = y_hat.detach().numpy()

    # In[Predict]
    plt.plot(y_torch[0, :, 0], 'k')
    plt.plot(y_hat[0, :, 0], 'g')
    plt.plot(y_torch[0, :, 0]-y_hat[0, :, 0], 'r')

    # In[Plot loss]
    plt.figure()
    plt.plot(LOSS)

