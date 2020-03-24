import torch
from torchid.linearsiso import LinearDynamicalSystem
import numpy as np
import matplotlib.pyplot as plt
import time

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
    f_coeff = np.array([-1.87212998940304, 0.942776404097492])  # f_1, f_2

    # In[Trace]
    time_start = time.time()
    G = LinearDynamicalSystem(b_coeff, f_coeff)
    G_traced = torch.jit.trace(G, (u_in, y_0, u_0))
    # In[Forward pass]

    time_start = time.time()
    y_out = G_traced(u_in, y_0, u_0)
    z = y_out
    L = torch.sum(z)
    L.backward()
    time_full = time.time() - time_start

    print(f"Time forward + backward: {time_full:.2f}")

    # In[Plot output]
    plt.figure()
    plt.plot(y_out.detach().numpy(), label='\hat y')
    plt.grid(True)

    # In[Test]
    y_out_np = y_out.detach().numpy()
    grad_out = np.ones_like(y_out_np)
    f_np = np.concatenate(([1.0], f_coeff))
    b_np = b_coeff
    import scipy.signal
    grad_u = scipy.signal.lfilter(b_np, f_np, grad_out, axis=0)
    grad_u = grad_u[::-1, :]



