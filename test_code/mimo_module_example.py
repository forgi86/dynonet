import torch
from torchid.module.lti import MimoLinearDynamicalOperator
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':


    # In[Setup problem]
    n_b = 3
    n_a = 2
    n_k = 20
    in_channels = 4
    out_channels = 5
    batch_size = 32
    seq_len = 1024
    G = MimoLinearDynamicalOperator(in_channels, out_channels, n_b, n_a, n_k=n_k)

    # build first-order stable systems
    with torch.no_grad():
        G.a_coeff[:, :, :] = 0.0
        G.b_coeff[:, :, :] = 0.0
        G.a_coeff[:, :, 0] = -0.99
        G.b_coeff[:, :, 0] = 0.01

    y_0 = torch.tensor(0*np.random.randn(*(out_channels, in_channels, n_a)))
    u_0 = torch.tensor(0*np.random.randn(*(out_channels, in_channels, n_b)))
    u_in = torch.tensor(1*np.random.randn(*(batch_size, seq_len, in_channels)), requires_grad=True)

    # In[Forward pass]
    y_out = G(u_in, y_0, u_0)
    y_out_np = y_out.detach().numpy()
    #y_out = y_out.detach().numpy(),
    # In[Plot output]


    #plt.figure()
    plt.plot(y_out_np[0, :, 0], label='y')
    #plt.grid(True)

    # In[Test doc]
    in_channels, out_channels = 2, 4
    n_b, n_a, n_k = 2, 2, 1
    G = MimoLinearDynamicalOperator(in_channels, out_channels, n_b, n_a, n_k)
    batch_size, seq_len = 32, 100
    u_in = torch.ones((batch_size, seq_len, in_channels))
    y_out = G(u_in, y_0, u_0)  # shape:
