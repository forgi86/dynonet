import os
import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from torchid.module.LTI import LinearMimo
from torchid.module.static import StaticMimoNonLin


# Good results, but a bit slow...
if __name__ == '__main__':

    # Set seed for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)

    # In[Settings]
    h5_filename = 'train.h5'
    #h5_filename = 'test.h5'
    signal_name = 'multisine'
    #signal_name = 'sinesweep' # available in test
    model_name = "model_BW_PBO_SOS_LBFGS_mod7"

    lr_ADAM = 2e-3
    lr_BFGS = 1e0
    num_iter_ADAM = 10000 #5000 or 4000
    num_iter_BFGS = 0 #500#1000
    msg_freq = 100

    num_iter = num_iter_ADAM + num_iter_BFGS
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
    scaler_y = 0.0006  # approx std(y)
    scaler_u = 50  # approx std(u)
    y = y/scaler_y
    u = u/scaler_u

    # In[Data to float 32]
    y = y.astype(np.float32)
    u = u.astype(np.float32)
    t = t.astype(np.float32)

    # In[Instantiate models]

    # Second-order dynamical system
    G1 = LinearMimo(1, 8, n_b=3, n_a=3, n_k=1)
    F1 = StaticMimoNonLin(8, 4, n_hidden=10)  # torch.nn.ReLU() #StaticMimoNonLin(3, 3, n_hidden=10)
    G2 = LinearMimo(4, 4, n_b=3, n_a=3)
    F2 = StaticMimoNonLin(4, 1, n_hidden=10)
    G3 = LinearMimo(1, 1, n_b=2, n_a=2, n_k=1)

    def model(u_in):
        y1_lin = G1(u_in)
        y1_nl = F1(y1_lin)
        y2_lin = G2(y1_nl)
        y2_nl = F2(y2_lin)

        y_hat = y2_nl + G3(u_in)
        return y_hat, y1_nl, y1_lin


    # In[Setup optimizer and closure]
    optimizer_ADAM = torch.optim.Adam([
        {'params': G1.parameters(), 'lr': lr_ADAM},
        {'params': G2.parameters(), 'lr': lr_ADAM},
        {'params': F1.parameters(), 'lr': lr_ADAM},
        {'params': F2.parameters(), 'lr': lr_ADAM},
        {'params': G3.parameters(), 'lr': lr_ADAM},
    ], lr=lr_ADAM)

    params = list(G1.parameters()) + list(G2.parameters()) + list(G3.parameters())
    optimizer_LBFGS = torch.optim.LBFGS(params, lr=lr_BFGS)#, tolerance_grad=1e-7, line_search_fn='strong_wolfe')

    def closure():
        optimizer_LBFGS.zero_grad()

        # Simulate
        y_hat, y1_nl, y1_lin = model(u_fit_torch)

        # Compute fit loss
        err_fit = y_fit_torch[:, 300:, :] - y_hat[:, 300:, :]
        loss = torch.mean(err_fit**2)

        # Backward pas
        loss.backward()
        return loss

    # In[Prepare tensors]
    u_fit_torch = torch.tensor(u)
    y_fit_torch = torch.tensor(y)

    # In[Train]
    LOSS = []
    start_time = time.time()
    for itr in range(0, num_iter):

        optimizer_ADAM.zero_grad()

        if itr < num_iter_ADAM:
            msg_freq = 10
            loss_train = optimizer_ADAM.step(closure)
        else:
            msg_freq = 10
            loss_train = optimizer_LBFGS.step(closure)

        if itr == 5000:
            for group in optimizer_ADAM.param_groups:
                group['lr'] = 2e-4
        LOSS.append(loss_train.item())
        if itr % msg_freq == 0:
            with torch.no_grad():
                RMSE = torch.sqrt(loss_train)
            print(f'Iter {itr} | Fit Loss {loss_train:.6f} | RMSE:{RMSE:.4f}')

    train_time = time.time() - start_time
    print(f"\nTrain time: {train_time:.2f}")  # 1900 seconds, loss was still going down

    # In[Save model]

    model_folder = os.path.join("models", model_name)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    torch.save(G1.state_dict(), os.path.join(model_folder, "G1.pkl"))
    torch.save(F1.state_dict(), os.path.join(model_folder, "F1.pkl"))
    torch.save(G2.state_dict(), os.path.join(model_folder, "G2.pkl"))
    torch.save(F2.state_dict(), os.path.join(model_folder, "F2.pkl"))
    torch.save(G3.state_dict(), os.path.join(model_folder, "G3.pkl"))


    # In[Simulate one more time]
    with torch.no_grad():
        y_hat, y1_nl, y1_lin = model(u_fit_torch)

    # In[Detach tensors]

    y_hat = y_hat.detach().numpy()
    y1_lin = y1_lin.detach().numpy()
    y1_nl = y1_nl.detach().numpy()

    # In[Plot signals]

    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(t, y[0, :, 0], label='$y$')
    ax[0].plot(t, y_hat[0, :, 0], label='$\hat y$')
    ax[0].plot(t, y[0, :, 0] - y_hat[0, :, 0], label='$e$')
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Displacement (mm)')
    ax[0].grid(True)
    ax[0].legend()
    ax[1].plot(t, u[0, :, 0])
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('Force (N)')
    ax[1].grid(True)
    #ax[1].legend()

    # In[Plot loss]
    fig, ax = plt.subplots()
    ax.plot(LOSS)
    plt.grid(True)
    fig_name = 'loss.pdf'
    plt.savefig(os.path.join("models", model_name, fig_name))




