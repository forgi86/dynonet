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

    model_name = 'ML_only_static'
    dataset_name = 'train_nonoise'
    # In[Load data]
    filename = os.path.join('data', 'dataset.h5')
    h5_data = h5py.File(filename, 'r')
    u = np.array(h5_data[dataset_name]['u'])
    y = np.array(h5_data[dataset_name]['y'])
    y0 = np.array(h5_data[dataset_name]['y0'])

    # Train on a single example
    u = u[[0], ...]
    y = y[[0], ...]

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
    model_folder = os.path.join("models", model_name)
    G.load_state_dict(torch.load(os.path.join(model_folder, "G.pkl")))
    F.load_state_dict(torch.load(os.path.join(model_folder, "F.pkl")))

    # In[Simulate]
    y_lin = G(u_torch)
    y_nl = F(y_lin)
    y_hat = y_nl

    # In[Detach]
    y_hat = y_hat.detach().numpy()

    # In[Predict]
    plt.plot(y0[0, :, 0], 'k', label='y0')
    plt.plot(y_hat[0, :, 0], 'g', label='$\hat y$')
    plt.plot(y0[0, :, 0]-y_hat[0, :, 0], 'r', label='e')
    plt.grid()
    plt.legend()


    # In[Plot loss]
    #plt.figure()
    #plt.plot(y0[0, :, 0], 'k', label='y')
    #plt.plot(y[0, :, 0], 'r', label='y0')

    #plt.grid()
    #plt.legend()