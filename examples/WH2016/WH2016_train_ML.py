import torch
import pandas as pd
import numpy as np
import os
from torchid.module.lti import SisoLinearDynamicOperator
from torchid.module.static import SisoStaticNonLin
import matplotlib.pyplot as plt
import time
import util.metrics
import pyro
import pyro.distributions as dist


if __name__ == '__main__':

    # In[Set seed for reproducibility]
    np.random.seed(0)
    torch.manual_seed(0)

    # In[Settings]
    n_a = 3
    n_b = 3
    # In[Column names in dataset]
    COL_F = 'fs'
    TAG_R = 'r'
    TAG_U = 'u'
    TAG_Y = 'y'

    # In[Load all datasets]
    #df_names = ["WH_TestDataset"]
    #df_names = ['WH_CombinedZeroMultisineSinesweep', 'WH_MultisineFadeOut', 'WH_SineInput_meas', 'WH_Triangle2_meas', 'WH_ZeroMeas']
    df_names = ['WH_CombinedZeroMultisineSinesweep']#, 'WH_MultisineFadeOut', 'WH_SineInput_meas', 'WH_Triangle2_meas']#, 'WH_MultisineFadeOut', 'WH_SineInput_meas', 'WH_Triangle2_meas']#, 'WH_ZeroMeas']

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


    batch_size, data_u, data_y = dict_data['WH_CombinedZeroMultisineSinesweep']


    data_u = torch.tensor(data_u)
    data_y = torch.tensor(data_y)

    # In[Setup model]

    G1 = SisoLinearDynamicOperator(n_b, n_a)
    F1 = SisoStaticNonLin()
    G2 = SisoLinearDynamicOperator(n_b, n_a)


    def model(u_in, y_meas):

        # we should probably register the module we wanna use...
        pyro.module("G1", G1)
        pyro.module("G2", G2)
        pyro.module("F1", F1)
        w_loc = torch.zeros_like(u_in, dtype=u_in.dtype)  # prior mean for w
        var_w = torch.ones_like(u_in, dtype=u_in.dtype)  # prior variance for w
        var_e = torch.ones_like(u_in, dtype=u_in.dtype)  # prior variance of e

        # sample from prior (value will be sampled by guide when computing the ELBO)
        w = pyro.sample("process_noise", dist.Normal(w_loc, var_w).to_event(2)) # sampled variance of w

        # From here onwards, deterministic model
        x = G1(u_in) + w
        z = F1(x)
        y_true = G2(z)  # true y, which is the mean of the y_obs distribution

        # distribution of observed y, conditioned to actually measured y
        with pyro.plate("batch_plate"):
            y_obs = pyro.sample("y_obs", dist.Normal(y_true, var_e).to_event(2), obs=y_meas) #.to_event(1)
        return y_obs

    def guide(u_in, y_meas):

        w_loc = torch.zeros_like(u_in, dtype=u_in.dtype)
        var_w = pyro.param("var_w", torch.tensor(1.0), constraint=pyro.distributions.constraints.positive)
        # a variational distribution for the process noise...
        w = pyro.sample("process_noise", dist.Normal(w_loc, var_w).to_event(1))

    model(data_u, data_y)