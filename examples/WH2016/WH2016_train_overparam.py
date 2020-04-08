import torch
import pandas as pd
import numpy as np
import os
from torchid.module.LTI import LinearMimo
from torchid.module.static import StaticMimoNonLin
import matplotlib.pyplot as plt
import time
import util.metrics

if __name__ == '__main__':

    # In[Set seed for reproducibility]
    np.random.seed(0)
    torch.manual_seed(0)

    # In[Settings]
    lr = 1e-3
    num_iter = 2000
    test_freq = 10
    n_batch = 1
    n_b = 3
    n_a = 3
    n_u = 1
    n_y = 1

    # In[Column names in dataset]
    COL_F = 'fs'
    TAG_R = 'r'
    TAG_U = 'u'
    TAG_Y = 'y'

    # In[Load all datasets]
    #df_names = ["WH_TestDataset"]
    #df_names = ['WH_CombinedZeroMultisineSinesweep', 'WH_MultisineFadeOut', 'WH_SineInput_meas', 'WH_Triangle2_meas', 'WH_ZeroMeas']
    df_names = ['WH_CombinedZeroMultisineSinesweep', 'WH_MultisineFadeOut', 'WH_SineInput_meas', 'WH_Triangle2_meas']#, 'WH_MultisineFadeOut', 'WH_SineInput_meas', 'WH_Triangle2_meas']#, 'WH_ZeroMeas']


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

#    df_X = pd.read_csv(os.path.join("data", "WH_CombinedZeroMultisineSinesweep.csv"))


    # In[Setup model]
    # Second-order dynamical system
    G1 = LinearMimo(1, 4, n_b, n_a, n_k=0)
    F1 = StaticMimoNonLin(4, 4, n_hidden=10) #torch.nn.ReLU() #StaticMimoNonLin(3, 3, n_hidden=10)
    G2 = LinearMimo(4, 2, n_b, n_a, n_k=0)
    F2 = StaticMimoNonLin(2, 1, n_hidden=10)
    G3 = LinearMimo(1, 1, n_b, n_a, n_k=0)


    def model(u_in):
        y1_lin = G1(u_in)
        y1_nl = F1(y1_lin)
        y2_lin = G2(y1_nl)
        y2_nl = F2(y2_lin)
        y3_lin = G3(y2_nl)

        y_hat = y3_lin
        return y_hat, y1_nl, y1_lin

    # In[Setup optimizer]
    optimizer = torch.optim.Adam([
        {'params': G1.parameters(), 'lr': lr},
        {'params': F1.parameters(), 'lr': lr},
        {'params': G2.parameters(), 'lr': lr},
        {'params': F2.parameters(), 'lr': lr},
        {'params': G3.parameters(), 'lr': lr},
    ], lr=lr)

    # In[Train]
    LOSS = []
    start_time = time.time()
    for itr in range(0, num_iter):

        dataset_name = np.random.choice(df_names)
        batch_size, data_u, data_y = dict_data[dataset_name]
        u_fit_torch = torch.tensor(data_u)
        y_fit_torch = torch.tensor(data_y)

        optimizer.zero_grad()

        # Simulate
        y_hat, y1_nl, y1_lin = model(u_fit_torch)

        # Compute fit loss
        err_fit = y_fit_torch - y_hat
        loss_fit = torch.mean(err_fit**2)
        loss = loss_fit

        LOSS.append(loss.item())
        if itr % test_freq == 0:
            with torch.no_grad():
                RMSE = torch.sqrt(loss)
            print(f'Iter {itr} | Fit Loss {loss_fit:.6f} | RMSE:{RMSE:.4f}')

        # Optimize
        loss.backward()
        optimizer.step()

    train_time = time.time() - start_time
    print(f"\nTrain time: {train_time:.2f}") # 182 seconds

    # In[Save model]
    model_name = "model_WH_over"
    model_folder = os.path.join("models", model_name)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    torch.save(G1.state_dict(), os.path.join(model_folder, "G1.pkl"))
    torch.save(F1.state_dict(), os.path.join(model_folder, "F1.pkl"))
    torch.save(G2.state_dict(), os.path.join(model_folder, "G2.pkl"))

    # In[To numpy]

    y_hat = y_hat.detach().numpy()
    y1_lin = y1_lin.detach().numpy()
    y1_nl = y1_nl.detach().numpy()

    # In[Plot]
    #plt.figure()
    #plt.plot(t_fit, y_fit, 'k', label="$y$")
    #plt.plot(t_fit, y_hat, 'b', label="$\hat y$")
    #plt.legend()

    plt.figure()
    plt.plot(LOSS)
    plt.grid(True)

    # In[Plot static non-linearity]

    y1_lin_min = np.min(y1_lin)
    y1_lin_max = np.max(y1_lin)

    in_nl = np.arange(y1_lin_min, y1_lin_max, (y1_lin_max- y1_lin_min)/1000).astype(np.float32).reshape(-1, 1)

    with torch.no_grad():
        out_nl = F1(torch.as_tensor(in_nl))

    plt.figure()
    plt.plot(in_nl, out_nl, 'b')
    plt.plot(in_nl, out_nl, 'b')
    #plt.plot(y1_lin, y1_nl, 'b*')
    plt.xlabel('Static non-linearity input (-)')
    plt.ylabel('Static non-linearity input (-)')
    plt.grid(True)

    # In[Plot]
    #e_rms = util.metrics.error_rmse(y_hat, y_hat)[0]
    #print(f"RMSE: {e_rms:.2f}")






