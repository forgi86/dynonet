import torch
import numpy as np
import scipy as sp
import scipy.signal
import time
from util.filtering import lfiltic_vec
from torch.nn.parameter import Parameter


class LinearDynamicalSystemFunction(torch.autograd.Function):
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

        >>> G = LinearDynamicalSystem.apply
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
    def forward(ctx, b_coeff, f_coeff, u_in, y_init, u_init):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """

        # detach tensors so we can cast to numpy
        b_coeff, f_coeff, u_in, y_init, u_init = b_coeff.detach(), f_coeff.detach(), u_in.detach(), y_init.detach(), u_init.detach()
        f_np = np.concatenate(([1.0], f_coeff.numpy()))
        b_np = b_coeff.numpy()
        zi = lfiltic_vec(b_np, f_np, y_init.numpy(), u_init.numpy())  # initial conditions for simulation

        y_out, _ = sp.signal.lfilter(b_np, f_np, u_in, axis=0, zi=zi.T)
        y_out = torch.as_tensor(y_out, dtype=u_in.dtype)

        ctx.save_for_backward(b_coeff, f_coeff, u_in, y_init, u_init, y_out)
        return y_out

    @staticmethod
    def backward(ctx, grad_output):

        debug = False
        if debug:
            import pydevd  # required to debug the backward pass. Why?!?
            pydevd.settrace(suspend=False, trace_only_current_thread=True)

        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        b_coeff, f_coeff, u_in, y_0, u_0, y_out = ctx.saved_tensors
        grad_b = grad_f = grad_u = grad_y0 = grad_u0 = None

        dtype_np = u_in.numpy().dtype

        N = u_in.shape[0]
        batch_size = u_in.shape[1]
        n_b = b_coeff.shape[0]  # number of coefficient of polynomial B(q)
        n_f = f_coeff.shape[0]  # number of coefficient of polynomial F(q)

        f_np = np.concatenate(([1.0], f_coeff.numpy())).astype(dtype_np)
        b_np = b_coeff.numpy()

        d0_np = np.array([1.0], dtype=dtype_np)
        d1_np = np.array([0.0, 1.0], dtype=dtype_np)
        #d2_np = np.array([0.0, 0.0, 1.0], dtype=u_in.numpy().dtype)

        product_imp = True

        if ctx.needs_input_grad[0]: # b_coeff
            # compute forward sensitivities w.r.t. the b_i parameters
            if product_imp:
                sens_b = np.zeros_like(u_in, shape=(N, batch_size, n_b))
                sens_b[:, :, 0] = sp.signal.lfilter(d0_np, f_np, u_in, axis=0)
                for idx_coeff in range(1, n_b):
                    sens_b[idx_coeff:, :, idx_coeff] = sens_b[:-idx_coeff, :, 0]
                sens_b = torch.as_tensor(sens_b)

                # compute vector-jacobian product for b
                grad_b = grad_output.view(N*batch_size, 1).t().matmul(sens_b.view(N*batch_size, n_b))
            else:
                sens_b0 = sp.signal.lfilter(d0_np, f_np, u_in, axis=0)
                grad_output_ext = np.r_[grad_output, np.zeros((n_b - 1, 1))]
                grad_b = scipy.signal.correlate(grad_output_ext, sens_b0, 'valid').astype(dtype_np).transpose()
                grad_b = torch.as_tensor(grad_b)

        if ctx.needs_input_grad[1]: # f_coeff

            if product_imp:
                # compute forward sensitivities w.r.t. the f_i parameters
                sens_f = np.zeros_like(u_in, shape=(N, batch_size, n_f))
                sens_f[:, :, 0] = sp.signal.lfilter(d1_np, f_np, -y_out, axis=0)
                for idx_coeff in range(1, n_f):
                    #sens_f[:, :, 1] = 0.0
                    sens_f[idx_coeff:, :,  idx_coeff] = sens_f[:-idx_coeff, :,  0]
                sens_f = torch.as_tensor(sens_f)

                # compute vector-jacobian product for f
                grad_f = grad_output.view(N*batch_size, 1).t().matmul(sens_f.view(N*batch_size, n_f))
            else:
                sens_f1 = sp.signal.lfilter(d1_np, f_np, -y_out, axis=0)
                grad_output_ext = np.r_[grad_output, np.zeros((n_f - 1, 1))]
                grad_f = scipy.signal.correlate(grad_output_ext, sens_f1, 'valid').astype(dtype_np).transpose()
                grad_f = torch.as_tensor(grad_f)

        if ctx.needs_input_grad[2]: # u_in

            # compute jacobian w.r.t. u
            grad_output_flip = grad_output.numpy()[::-1, :]
            grad_u = scipy.signal.lfilter(b_np, f_np, grad_output_flip, axis=0)
            grad_u = np.array(grad_u[::-1, :]).astype(dtype_np)
            grad_u = torch.as_tensor(grad_u)

            #grad_u = torch.as_tensor(np.zeros_like(u_in, shape=(N, batch_size)))
            #u_imp = scipy.signal.unit_impulse(N, idx=None, dtype=u_in.numpy().dtype)
            #g_imp = torch.as_tensor(sp.signal.lfilter(b_np, f_np, u_imp).astype(u_in.numpy().dtype))
            #g_imp = g_imp.view(1, -1)
            #for idx_time in range(N):
            #    grad_u[idx_time, :] = g_imp[:, 0:N-idx_time].matmul(grad_output[idx_time:, :])  #grad_output_t[idx_time:, :].matmul(g_imp[:, 0:N-idx_time])

            #for batch_idx in range(batch_size):
        #np.correlate(grad_output[0, :], g_imp[0, :], 'same')
        #np.correlate(g_imp[0, :], grad_output[:, 0], 'same')

        # if ctx.needs_input_grad[3]: # y_init
        #     init_np = np.zeros((1, n_f)).astype(dtype_np)
        #     num_np = np.array([0.0])
        #     zi = lfiltic_vec([[0.0]], f_np, init_np)
        #     sens_y0 = np.zeros_like(u_in, shape=(N, batch_size, n_f))
        #     sens_y0[:, :, 0], _ = sp.signal.lfilter([0.0], f_np, u_in, axis=0, zi=zi.T)
        #     for idx_coeff in range(1, n_f):
        #         sens_y0[idx_coeff:, :, idx_coeff] = sens_y0[:-idx_coeff, :, 0]
        #     sens_y0 = torch.as_tensor(sens_y0)
        #
        #     grad_y0 = grad_output.view(N * batch_size, 1).t().matmul(sens_y0.view(N * batch_size, n_f))

        return grad_b, grad_f, grad_u, grad_y0, grad_u0


class LinearDynamicalSystem(torch.nn.Module):
    def __init__(self, b_coeff, f_coeff):
        super(LinearDynamicalSystem, self).__init__()
        self.b_coeff = Parameter(torch.tensor(b_coeff))
        self.f_coeff = Parameter(torch.tensor(f_coeff))


    def forward(self, u_in, y_init, u_init):
        return LinearDynamicalSystemFunction.apply(self.b_coeff, self.f_coeff, u_in, y_init, u_init)


class SecondOrderOscillator(torch.nn.Module):
    def __init__(self, b_coeff, rho, psi):
        super(SecondOrderOscillator, self).__init__()
        self.b_coeff = Parameter(torch.tensor(b_coeff))
        self.rho = Parameter(torch.tensor(rho))
        self.psi = Parameter(torch.tensor(psi))

    def forward(self, u_in, y_init, u_init):

        r = torch.sigmoid(self.rho)
        theta = np.pi * torch.sigmoid(self.psi)
        f_1 = -2 * r * torch.cos(theta)
        f_2 = r ** 2
        f_coeff = torch.stack((f_1, f_2))
        return LinearDynamicalSystemFunction.apply(self.b_coeff, f_coeff, u_in, y_init, u_init)


if __name__ == '__main__':

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

    # In[Setup problem]
    n_batch = 1
    n_b = 2
    n_f = 2
    N = 10
    u_in = torch.rand((N, n_batch), dtype=torch.double, requires_grad=True)
    y_init = torch.zeros((n_batch, n_f), dtype=torch.double, requires_grad=False)
    u_init = torch.zeros((n_batch, n_b), dtype=torch.double, requires_grad=False)

    # coefficients of a 2nd order oscillator
    b_coeff = torch.tensor([0.0706464146944544, 0, 0], dtype=torch.double, requires_grad=True)  # b_1, b_2
    f_coeff = torch.tensor([-1.87212998940304, 0.942776404097492], dtype=torch.double, requires_grad=True)  # f_1, f_2
    G = LinearDynamicalSystemFunction.apply

    inputs = (b_coeff, f_coeff, u_in, y_init, u_init)

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
    ax[0].plot(analytical[0][1, :], 'k*', label='$\\tilde{b}_2$')

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


