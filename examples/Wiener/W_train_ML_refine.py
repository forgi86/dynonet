import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import h5py
from torchid.module.lti import SisoLinearDynamicalOperator
from torchid.module.static import SisoStaticNonLinearity
import time


class StaticNonLinPoly(torch.nn.Module):
    def __init__(self, n_p):
        super(StaticNonLinPoly, self).__init__()
        self.p_coeff = torch.nn.Parameter(torch.zeros(n_p))

    def forward(self, u):
        u_sq = u**2
        y = u_sq*self.p_coeff[0] + u*self.p_coeff[1] + self.p_coeff[2]
        return y


if __name__ == '__main__':

    # In[Settings]
    num_points = 2001
    num_iter = 100
    test_freq = 10
    lr = 1e-3

    n_a = 1
    n_b = 1
    n_k = 1

#    var_w = 4.0  # need only to define a reasonable integration interval
#    var_e = 1.0

#    std_w = np.sqrt(var_w)
#    std_e = np.sqrt(var_e)
    model_name_load = 'NLS_noise'  # start from NLS fit
    model_name_save = 'ML_noise'  # Refine with ML fit
    dataset_name = 'train_noise'

    # In[Load data]
    filename = os.path.join('data', 'dataset.h5')
    h5_data = h5py.File(filename, 'r')
    u = np.array(h5_data[dataset_name]['u'])
    y = np.array(h5_data[dataset_name]['y'])
    y0 = np.array(h5_data[dataset_name]['y0'])

    # Train on a single example
    u = u[0:1, ...]
    y = y[0:1, ...]

    batch_size = u.shape[0]
    seq_len = u.shape[1]
    n_u = u.shape[2]
    n_y = y.shape[2]

    # In[To tensors]
    u_torch = torch.tensor(u, dtype=torch.float32)
    y_torch = torch.tensor(y, dtype=torch.float32)

    # In[Deterministic model]
    G = SisoLinearDynamicalOperator(n_b, n_a, n_k=n_k)
    F = SisoStaticNonLinearity(n_hidden=10)
    model_folder = os.path.join("models", model_name_load)
    G.load_state_dict(torch.load(os.path.join(model_folder, "G.pkl")))
    F.load_state_dict(torch.load(os.path.join(model_folder, "F.pkl")))

    #F = StaticNonLinPoly(n_p=3)
    #with torch.no_grad():
    #    F.p_coeff[0] = 1.0
    #    F.p_coeff[1] = 1.0
    #    F.p_coeff[2] = 0.0

    #with torch.no_grad():
    #    G.a_coeff[0, 0, 0] = 0.5
    #    G.b_coeff[0, 0, 0] = 1.0


    # In[]
    with torch.no_grad():
        x0 = G(u_torch)
        pow_x0 = torch.mean(x0**2)
        std_x0 = torch.sqrt(pow_x0)

    std_w_est = torch.tensor(4.0, requires_grad=True)
    std_e_est = torch.tensor(1.0, requires_grad=True)

    # In[Log-likelihood]
    optimizer = torch.optim.Adam([
        {'params': G.parameters(),    'lr': lr},
        {'params': F.parameters(), 'lr': lr},
        {'params': std_w_est, 'lr': lr},
        {'params': std_e_est, 'lr': lr},
    ], lr=lr)

    # In[Train]
    pi = torch.tensor(3.14)  # useless, enters as a constant in the cost function
    N = u_torch.shape[0] * u_torch.shape[1]
    LOSS = []
    start_time = time.time()

    for itr in range(num_iter):
        optimizer.zero_grad()

        var_w_est = std_w_est**2
        var_e_est = std_e_est**2

        # Simulate
        x0 = G(u_torch)
        with torch.no_grad():
            size_x0 = torch.tensor(2.0) #torch.sqrt(torch.mean(x0**2))
            #size_x0 = torch.sqrt(var_w_est)
            a_int = -size_x0 * 6
            b_int = size_x0 * 6
            size_int = b_int - a_int
            integration_points = torch.linspace(a_int.item(), b_int.item(), num_points, dtype=torch.float32)

        x_int = x0 + integration_points # x at selected integration points
        F_x0_int = F(x_int[..., None]).squeeze(-1)
        E_y = 1/var_e_est*(y_torch - F_x0_int)**2
        E_x = 1/var_w_est*(integration_points)**2  # same as x_int - x0
        E = E_y + E_x
        exp_E = torch.exp(-1/2*E)

        I = torch.mean(exp_E, dim=(-1)) * size_int  # multirectancle integration

        L = N * torch.log(2*pi) +\
            N/2*torch.log(var_w_est*var_e_est) \
            - torch.sum(torch.log(I+1*1e-12))
        L = L/N

        # Compute fit loss
        loss = L

        LOSS.append(loss.item())
        if itr % test_freq == 0:
            print(f'Iter {itr} | Fit Loss {loss:.4f} Integration Size: {size_int:.1f} Std w: {std_w_est}')

        # Optimize
        loss.backward()
        optimizer.step()

    train_time = time.time() - start_time

    # In[Save]

    model_folder = os.path.join("models", model_name_save)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    torch.save(G.state_dict(), os.path.join(model_folder, "G.pkl"))
    torch.save(F.state_dict(), os.path.join(model_folder, "F.pkl"))

    # In[Simulate noise-free]
    with torch.no_grad():
        y_lin = G(u_torch)
        y_nl = F(y_lin)
        y_hat = y_nl

    y_lin = y_lin.numpy()
    y_nl = y_nl.numpy()
    y_hat = y_hat.numpy()

    # In[Predict]
    plt.figure()
    plt.plot(y[0, :, 0], 'k')
    plt.plot(y_hat[0, :, 0], 'g')
    plt.plot(y[0, :, 0] - y_hat[0, :, 0], 'r')
    plt.show()

    plt.figure()
    plt.plot(y_lin[0, :], y_hat[0, :], '*k', label='x')
    plt.legend()
    plt.show()

    # In[Plot loss]
    plt.figure()
    plt.plot(LOSS)
    plt.show()

    # In[Inspect]
    with torch.no_grad():
        optimizer.zero_grad()

        var_w_est = std_w_est**2
        var_e_est = std_e_est**2

        # Simulate
        x0 = G(u_torch)
        with torch.no_grad():
            size_x0 = torch.tensor(2.0) #torch.sqrt(torch.mean(x0**2))
            #size_x0 = torch.sqrt(var_w_est)
            a_int = -size_x0 * 6
            b_int = size_x0 * 6
            size_int = b_int - a_int
            integration_points = torch.linspace(a_int.item(), b_int.item(), num_points, dtype=torch.float32)

        x_int = x0 + integration_points # x at selected integration points
        F_x0_int = F(x_int[..., None]).squeeze(-1)
        E_y = 1/var_e_est*(y_torch - F_x0_int)**2
        E_x = 1/var_w_est*(integration_points)**2  # same as x_int - x0
        E = E_y + E_x
        exp_E = torch.exp(-1/2*E)

        I = torch.mean(exp_E, dim=(-1)) * size_int # multirectancle integration

        L = N * torch.log(2*pi) +\
            N/2*torch.log(var_w_est*var_e_est) \
            - torch.sum(torch.log(I+1*1e-12))
        L = L/N


    # In[]

    E_xx = 1 / var_w_est * (x_int - x0) ** 2  # same as x_int - x0
    plt.figure()
    plt.plot(x_int[0, 0, :], E_y[0, 0, :])
    plt.plot(x_int[0, 0, :], E_xx[0, 0, :])
    plt.xlabel('x')
    plt.ylabel('E')
    plt.show()

    plt.figure()
    plt.plot(x_int[0, 0, :], exp_E[0, 0, :])
    plt.xlabel('x')
    plt.ylabel('E')
    plt.show()

