import torch
import numpy as np
import scipy as sp
import scipy.signal
import time
from util.filtering import lfiltic_vec, lfilter_mimo, lfilter_mimo_components
from torch.nn.parameter import Parameter


class LinearMimoFunction(torch.autograd.Function):
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

        >>> G = LinearMimoFunction.apply
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
    def forward(ctx, b_coeff, a_coeff, u_in, y_0, u_0):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """

        # detach tensors so we can cast to numpy
        b_coeff, a_coeff, u_in, y_0, u_0 = b_coeff.detach(), a_coeff.detach(), u_in.detach(), y_0.detach(), u_0.detach()

        # useful parameters
        out_ch = b_coeff.shape[0]
        in_ch = b_coeff.shape[1]
        n_a = a_coeff.shape[2]

        # construct the A(q) polynomial with coefficient a_0=1
        a_poly = np.empty_like(a_coeff, shape=(out_ch, in_ch, n_a + 1))
        a_poly[:, :, 0] = 1
        a_poly[:, :, 1:] = a_coeff[:, :, :]
        b_poly = np.array(b_coeff)

        #zi = lfiltic_vec(b_poly, a_poly, y_init.numpy(), u_init.numpy())  # initial conditions for simulation

        y_out_comp = lfilter_mimo_components(b_poly, a_poly, u_in)
        y_out = np.sum(y_out_comp, axis=2)
        y_out = torch.as_tensor(y_out, dtype=u_in.dtype)
        y_out_comp = torch.as_tensor(y_out_comp)

        ctx.save_for_backward(b_coeff, a_coeff, u_in, y_0, u_0, y_out_comp)
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
        b_coeff, a_coeff, u_in, y_0, u_0, y_out_comp = ctx.saved_tensors
        grad_b = grad_a = grad_u = grad_y0 = grad_u0 = None
        dtype_np = u_in.numpy().dtype

        out_ch, in_ch, n_b = b_coeff.shape
        _, _, n_a = a_coeff.shape
        batch_size, _, seq_len = u_in.shape


        a_poly = np.empty_like(a_coeff, shape=(out_ch, in_ch, n_a + 1))
        a_poly[:, :, 0] = 1
        a_poly[:, :, 1:] = a_coeff[:, :, :]
        b_poly = np.array(b_coeff)  # not required?

        d0_np = np.array([1.0], dtype=dtype_np) #np.ones_like(u_in, shape=(out_ch, in_ch, 1))
        d1_np = np.array([0.0, 1.0], dtype=dtype_np)

        if ctx.needs_input_grad[0]:  # b_coeff
            # compute forward sensitivities w.r.t. the b_i parameters
            sens_b = np.zeros_like(u_in, shape=(batch_size, out_ch, in_ch, n_b, seq_len))

            for out_idx in range(out_ch):  # it is like a lfilter_mimo_components, can be optimized
                for in_idx in range(in_ch):
                    sens_b[:, out_idx, in_idx, 0, :] = sp.signal.lfilter(d0_np, a_poly[out_idx, in_idx, :], u_in[:, in_idx, :])
            for idx_coeff in range(1, n_b):
                sens_b[:, :, :, idx_coeff, idx_coeff:] = sens_b[:, :, :, 0, :-idx_coeff]
            sens_b = torch.as_tensor(sens_b)
            #grad_b = torch.einsum('boidt,bqt->oid', sens_b, grad_output)
            grad_b = torch.einsum('bot,boidt->oid', grad_output, sens_b)

        if ctx.needs_input_grad[1]:  # a_coeff
            # compute forward sensitivities w.r.t. the f_i parameters
            sens_a = np.zeros_like(u_in, shape=(batch_size, out_ch, in_ch, n_a, seq_len))
            for out_idx in range(out_ch): # it is like a lfilter_mimo_components, can be optimized
                for in_idx in range(in_ch):
                    sens_a[:, out_idx, in_idx, 0, :] = sp.signal.lfilter(d1_np, a_poly[out_idx, in_idx, :], -y_out_comp[:, out_idx, in_idx, :], axis=-1)

            for idx_coeff in range(1, n_a):
                sens_a[:, :, :, idx_coeff, idx_coeff:] = sens_a[:, :, :, 0, :-idx_coeff]
            sens_a = torch.as_tensor(sens_a)
            # compute vector-jacobian product for f
            grad_a =  torch.einsum('bot,boidt->oid', grad_output, sens_a)


        if ctx.needs_input_grad[2]: # u_in
            # compute jacobian w.r.t. u
            grad_output_flip = grad_output.numpy()[:, :, ::-1]

            grad_u = np.zeros_like(u_in)  # B, I, T
            for in_idx in range(in_ch):
                for out_idx in range(out_ch):
                    grad_u[:, in_idx, :] += scipy.signal.lfilter(b_poly[out_idx, in_idx, :], a_poly[out_idx, in_idx, :], grad_output_flip[:, out_idx, :], axis=-1)
            grad_u = np.array(grad_u[:, :, ::-1]).astype(dtype_np)

            grad_u = torch.as_tensor(grad_u)

        return grad_b, grad_a, grad_u, grad_y0, grad_u0


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
    in_ch = 1
    out_ch = 4
    n_b = 2
    n_a = 1
    batch_size = 8
    seq_len = 16

    # In[Create system]

    b_coeff = torch.tensor(np.random.randn(*(out_ch, in_ch, n_b)), requires_grad=True)
    a_coeff = torch.tensor(np.random.rand(*(out_ch, in_ch, n_a)), requires_grad=True)
    G = LinearMimoFunction.apply
    y_0 = torch.tensor(0*np.random.randn(*(out_ch, in_ch, n_a)))
    u_0 = torch.tensor(0*np.random.randn(*(out_ch, in_ch, n_b)))
    u_in = torch.tensor(1*np.random.randn(*(batch_size, in_ch, seq_len)), requires_grad=True)
    inputs = (b_coeff, a_coeff, u_in, y_0, u_0)

    # In[Forward pass]
    y_out = G(*inputs)

    # In[Finite difference derivatives computation]
    def G_fun(input):
        return _as_tuple(G(*input))[0]
    numerical = get_numerical_jacobian(G_fun, inputs)

    # In[Autodiff derivatives computation]
    analytical, reentrant, correct_grad_sizes = get_analytical_jacobian(inputs, y_out)
    #torch.max(numerical[0]- analytical[0])
    test = gradcheck(G, inputs, eps=1e-6, atol=1e-4, raise_exception=True)
