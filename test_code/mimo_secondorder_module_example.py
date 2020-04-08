import torch
from torchid.module.LTI import LinearSecondOrderMimo
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':


    # In[Setup problem]
    n_b = 3
    n_a = 2
    in_channels = 4
    out_channels = 5
    batch_size = 32
    seq_len = 1024
    G = LinearSecondOrderMimo(in_channels, out_channels)


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