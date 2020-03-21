import torch

class SecondOrderDynamical(torch.autograd.Function):
    r"""Applies a linear second-order filter to the incoming data: :math:`y = G(u)`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> G = SecondOrderDynamical.apply
        >>> n_b = 2
        >>> n_f = 2
        >>> N = 500
        >>> y_0 = torch.zeros(n_f, dtype=torch.double)
        >>> u_0 = torch.zeros(n_b, dtype=torch.double)
        >>> b_coeff = torch.tensor([0.0706464146944544, 0], dtype=torch.double, requires_grad=True)  # b_1, b_2
        >>> f_coeff = torch.tensor([-1.87212998940304, 0.942776404097492], dtype=torch.double, requires_grad=True)  # f_1, f_2
        >>> inputs = (b_coeff, f_coeff, u_in, y_0, u_0)
        >>> Y = G(*inputs)
        >>> print(Y.size())
        torch.Size([500, 1])
    """

    @staticmethod
    def forward(ctx, b_coeff, f_coeff, u_in, y_0, u_0):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """

        y_out = torch.empty_like(u_in)
        y_out[0:1, :] = y_0[0:1]
        for idx in range(2, len(y_out)):
            y_out[idx] = b_coeff[0]*u_in[idx-1, :] + b_coeff[1]*u_in[idx-2, :] + \
                -f_coeff[0]*y_out[idx-1, :] -f_coeff[1]*y_out[idx-2, :]

        ctx.save_for_backward(b_coeff, f_coeff, u_in, y_0, u_0, y_out)
        return y_out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        b_coeff, f_coeff, u_in, y_0, u_0, y_out = ctx.saved_tensors
        grad_b = grad_f = grad_u = grad_y0 = grad_u0 = None

        N = u_in.shape[0]
        n_b = b_coeff.shape[0]  # number of coefficient of polynomial B(q)
        n_f = f_coeff.shape[0]  # number of coefficient of polynomial F(q)

        # compute forward sensitivity w.r.t. b_i parameters
        sens_b = torch.zeros((N, n_b), dtype=u_in.dtype) # empty_like better...
        for idx in range(2, N):
            sens_b[idx, 0] = u_in[idx-1] -f_coeff[0]*sens_b[idx-1, 0] -f_coeff[1]*sens_b[idx-2, 0]
            sens_b[idx, 1] = u_in[idx-2] -f_coeff[0]*sens_b[idx-1, 1] -f_coeff[1]*sens_b[idx-2, 1]

        # compute gradient w.r.t. b
        grad_b = grad_output.t().matmul(sens_b)

        # compute forward sensitivity w.r.t. the f_i parameters
        sens_f = torch.zeros((N, n_b), dtype=u_in.dtype) # empty_like better...
        for idx in range(2, N):
            sens_f[idx, 0] = -y_out[idx-1] -f_coeff[0]*sens_f[idx-1, 0] -f_coeff[1]*sens_f[idx-2, 0]
            sens_f[idx, 1] = -y_out[idx-2] -f_coeff[0]*sens_f[idx-1, 1] -f_coeff[1]*sens_f[idx-2, 1]

        # compute gradient w.r.t. f
        grad_f = grad_output.t().matmul(sens_f)

        return grad_b, grad_f, grad_u, grad_y0, grad_u0


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

    import numpy as np
    import matplotlib.pyplot as plt
    from torch.autograd import gradcheck
    from torch.autograd.gradcheck import get_numerical_jacobian, get_analytical_jacobian

    G = SecondOrderDynamical.apply

    # In[Setup problem]
    n_batch = 1
    n_b = 2
    n_f = 2
    N = 200
    u_in = torch.rand((N, n_batch), dtype=torch.double, requires_grad=False)
    y_0 = torch.zeros(n_f, dtype=torch.double)
    u_0 = torch.zeros(n_b, dtype=torch.double)
    # coefficients of a 2nd order oscillator
    b_coeff = torch.tensor([0.0706464146944544, 0], dtype=torch.double, requires_grad=True)  # b_1, b_2
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
    # In[Plot derivatives]
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(numerical[0][0, :], 'b', label='$\\tilde{b}_1$')
    ax[0].plot(numerical[0][1, :], 'k', label='$\\tilde{b}_2$')
    ax[0].grid(True)
    ax[0].legend()

    ax[1].plot(numerical[1][0, :], 'b', label='$\\tilde{f}_1$')
    ax[1].plot(numerical[1][1, :], 'k', label='$\\tilde{f}_2$')
    ax[1].grid(True)
    ax[1].legend()


    fig, ax = plt.subplots(2, 1)
    ax[0].plot(numerical[0][0, :], 'b', label='$\\tilde{b}_1$')
    ax[0].plot(analytical[0][0, :], 'b*', label='$\\tilde{b}_1$')

    ax[0].plot(numerical[0][1, :], 'k', label='$\\tilde{b}_2$')
    ax[0].plot(analytical[0][1, :], 'k*', label='$\\tilde{b}_1$')

    ax[0].grid(True)
    ax[0].legend()

    ax[1].plot(numerical[1][0, :], 'b', label='$\\tilde{f}_1$')
    ax[1].plot(analytical[1][0, :], 'b*', label='$\\tilde{f}_1$')

    ax[1].plot(numerical[1][1, :], 'k', label='$\\tilde{f}_2$')
    ax[1].plot(analytical[1][1, :], 'k*', label='$\\tilde{f}_2$')

    ax[1].grid(True)
    ax[1].legend()

    # In[builtin gradient check]
    #test = gradcheck(G, input, eps=1e-6, atol=1e-4, raise_exception=True)


