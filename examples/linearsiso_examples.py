import torch
from torchid.linearsiso import LinearDynamicalSystem
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import gradcheck
from torch.autograd.gradcheck import get_numerical_jacobian, get_analytical_jacobian


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

    G = LinearDynamicalSystem.apply

    # In[Setup problem]
    n_batch = 1
    n_b = 2
    n_f = 2
    N = 100
    u_in = torch.ones((N, n_batch), dtype=torch.double, requires_grad=False)
    y_0 = torch.zeros((n_batch, n_f), dtype=torch.double)
    u_0 = torch.zeros((n_batch, n_b), dtype=torch.double)

    # coefficients of a 2nd order oscillator
    b_coeff = torch.tensor([0.0706464146944544], dtype=torch.double, requires_grad=True)  # b_1, b_2
    f_coeff = torch.tensor([-1.87212998940304, 0.942776404097492], dtype=torch.double, requires_grad=True)  # f_1, f_2
    inputs = (b_coeff, f_coeff, u_in, y_0, u_0)

    # In[Forward pass]
    y_out = G(*inputs)

    # In[Finite difference derivatives computation]
    def G_fun(input):
        return _as_tuple(G(*input))[0]
    numerical = get_numerical_jacobian(G_fun, inputs)

    # In[Autodiff derivatives computation]
    analytical, reentrant, correct_grad_sizes = get_analytical_jacobian(inputs, y_out)
    torch.max(numerical[0]- analytical[0])


    # In[Plot output]
    plt.figure()
    plt.plot(y_out.detach().numpy(), label='\hat y')
    plt.grid(True)

    # In[Plot derivatives]

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(numerical[0][0, :], 'b', label='$\\tilde{b}_1$')
    #ax[0].plot(numerical[0][1, :], 'k', label='$\\tilde{b}_2$')
    ax[0].grid(True)
    ax[0].legend()

    ax[1].plot(numerical[1][0, :], 'b', label='$\\tilde{f}_1$')
    ax[1].plot(numerical[1][1, :], 'k', label='$\\tilde{f}_2$')
    ax[1].grid(True)
    ax[1].legend()

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(numerical[0][0, :], 'b', label='$\\tilde{b}_1$')
    ax[0].plot(analytical[0][0, :], 'b*', label='$\\tilde{b}_1$')

    #ax[0].plot(numerical[0][1, :], 'k', label='$\\tilde{b}_2$')
    #ax[0].plot(analytical[0][1, :], 'k*', label='$\\tilde{b}_2$')

    ax[0].grid(True)
    ax[0].legend()

    ax[1].plot(numerical[1][0, :], 'b', label='$\\tilde{f}_1$')
    ax[1].plot(analytical[1][0, :], 'b*', label='$\\tilde{f}_1$')

    ax[1].plot(numerical[1][1, :], 'k', label='$\\tilde{f}_2$')
    ax[1].plot(analytical[1][1, :], 'k*', label='$\\tilde{f}_2$')

    ax[1].grid(True)
    ax[1].legend()

    # In[Plot derivatives delayed]

    # delayed sensitivities match!

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(numerical[0][0, 0:-1], 'b', label='$\\tilde{b}_1$')
    ax[0].plot(numerical[0][1, 1:], 'k', label='$\\tilde{b}_2$')
    ax[0].grid(True)
    ax[0].legend()

    ax[1].plot(numerical[1][0, 0:-1], 'b', label='$\\tilde{f}_1$')
    ax[1].plot(numerical[1][1, 1:], 'k', label='$\\tilde{f}_2$')
    ax[1].grid(True)
    ax[1].legend()


    # In[builtin gradient check]
    test = gradcheck(G, inputs, eps=1e-6, atol=1e-4, raise_exception=True)
