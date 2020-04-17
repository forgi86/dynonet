import torch
import pandas as pd
import numpy as np
import os
from torchid.module.LTI import LinearSiso
import matplotlib.pyplot as plt
import pyro
import pyro.distributions as dist



if __name__ == '__main__':

    # In[Set seed for reproducibility]
    np.random.seed(0)
    torch.manual_seed(0)

    # In[settings]
    add_noise = True
    lr = 1e-4
    num_iter = 200000
    test_freq = 100
    n_batch = 1
    n_b = 2
    n_a = 2

    # In[Column names in the dataset]
    COL_T = ['time']
    COL_X = ['V_C', 'I_L']
    COL_U = ['V_IN']
    COL_Y = ['V_C']

    # In[Load dataset]
    df_X = pd.read_csv(os.path.join("data", "RLC_data_id_lin.csv"))
    t = np.array(df_X[COL_T], dtype=np.float32)
    y = np.array(df_X[COL_Y], dtype=np.float32)
    x = np.array(df_X[COL_X], dtype=np.float32)
    u = np.array(df_X[COL_U], dtype=np.float32)

    # In[Add measurement noise]
    std_noise_V = add_noise * 10.0
    std_noise_I = add_noise * 1.0
    std_noise = np.array([std_noise_V, std_noise_I])
    x_noise = np.copy(x) + np.random.randn(*x.shape) * std_noise
    x_noise = x_noise.astype(np.float32)
    y_noise = np.copy(x_noise[:, [0]])
    y_nonoise = np.copy(x[:, [0]])

    # In[Prepare data]
    u_torch = torch.tensor(u[None, ...], dtype=torch.float, requires_grad=False)
    y_meas_torch = torch.tensor(y_noise[None, ...], dtype=torch.float)
    y_true_torch = torch.tensor(y_nonoise[None, ...], dtype=torch.float)

    # In[Setup deterministic model]

    G = LinearSiso(n_b, n_a)
    model_folder = os.path.join("models", 'IIR')
    G.load_state_dict(torch.load(os.path.join(model_folder, "G.pkl")))

    # In[Setup stochastic model]
    def model(u_in, y_meas):

        # we should probably register the module we wanna use...
        #pyro.module("G", G)

        prior_scale_e = torch.tensor(std_noise_V)#10*torch.ones_like(u_in, dtype=u_in.dtype)  # prior variance of e
        y_true = G(u_in)  # true y, which is the mean of the y_obs distribution

        # distribution of observed y, conditioned to actually measured y
        #with pyro.plate("batch_plate"):
        y_obs = pyro.sample("y_obs", dist.Normal(y_true, prior_scale_e).to_event(2), obs=y_meas)
        return y_obs


    def guide(u_in, y_meas):
        #pyro.module("G", G)
        scale_e = pyro.param("var_e", torch.tensor(std_noise_V), constraint=pyro.distributions.constraints.positive)
        y_true = G(u_in)  # true y, which is the mean of the y_obs distribution

        # distribution of observed y, conditioned to actually measured y
        #with pyro.plate("batch_plate"):
        y_obs = pyro.sample("y_obs", dist.Normal(y_true, scale_e).to_event(2))
        return y_obs


    # In[Prepare training loop]
    pyro.clear_param_store()
    svi = pyro.infer.SVI(model=model,
                         guide=guide,
                         optim=pyro.optim.Adam({"lr": 1e-4}),
                         #optim=pyro.optim.SGD({"lr": 0.00001, "momentum":0.1}),
                         loss=pyro.infer.Trace_ELBO())
    LOSS_ITR, VAR_ITR = [], []

    # In[Tranining loop]
    for itr in range(num_iter):
        #losses.append(svi.step(guess))
        LOSS_ITR.append(svi.step(u_torch, y_meas_torch))
        var_e_curr = pyro.param("var_e").item()
        VAR_ITR.append(var_e_curr)
        if itr % test_freq == 0:
            print(f'Iter {itr} | Current variance estimate {var_e_curr:.4f}')

    # In[Plot]
    plt.figure()
    plt.plot(LOSS_ITR)
    plt.title("ELBO")
    plt.xlabel("step")
    plt.ylabel("loss")

    plt.figure()
    plt.plot(VAR_ITR)
    plt.title("Param")
    plt.xlabel("step")
    plt.ylabel("Estimated variance")

    print('var_e = ', pyro.param("var_e").item())


