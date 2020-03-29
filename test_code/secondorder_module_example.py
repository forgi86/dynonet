import torch
from torchid.linearsiso import LinearDynamicalSystem, SecondOrderOscillator
import numpy as np
import matplotlib.pyplot as plt


# copied from torch.autograd.gradcheck
def istuple(obj):
    # Usually instances of PyStructSequence is also an instance of tuple
    # but in some py2 environment it is not, so we have to manually check
    # the name of the type to determine if it is a namedtupled returned
    # by a pytorch operator.
    t = type(obj)
    return isinstance(obj, tuple) or t.__module__ == 'torch.return_types'

# copied from torch.autograd.gradcheck
def _as_tuple(x):
    if istuple(x):
        return x
    elif isinstance(x, list):
        return tuple(x)
    else:
        return x,


if __name__ == '__main__':


    # In[Setup problem]
    n_batch = 1
    n_b = 2
    n_f = 2
    N = 10000000
    u_in = torch.rand((N, n_batch), requires_grad=True)
    y_0 = torch.zeros((n_batch, n_f), requires_grad=False)
    u_0 = torch.zeros((n_batch, n_b), requires_grad=True)

    # coefficients of a 2nd order oscillator
    b_coeff = np.array([0.0706464146944544])  # b_1, b_2
    rho = np.array(1.0)
    psi = np.array(0.0)

    G = SecondOrderOscillator(b_coeff, rho, psi)

    # In[Forward pass]
    y_out = G(u_in, y_0, u_0)

    # In[Plot output]
    plt.figure()
    plt.plot(y_out.detach().numpy(), label='\hat y')
    plt.grid(True)