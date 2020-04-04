import torch
import numpy as np  # temporary
from torch.nn.parameter import Parameter
from torchid.functional.linearmimo import LinearMimoFunction


class LinearMimo(torch.nn.Module):
    r"""Applies a Dynamical Linear MIMO system to an input signal.

    Args:
        b_coeff (np.array): Learnable coefficients of the transfer function numerator
        a_coeff (np.array): Learnable coefficients of the transfer function denominator
        n_k (int): Number of input delays

    Shape:
        - Input: :math:`(B, T)`
        - Output: :math:`(B, T)`

    Attributes:
        b_coeff (Tensor): the learnable coefficients of the transfer function numerator
        a_coeff (Tensor): the learnable coefficients of the transfer function denominator

    Examples::

        >>> batch_size = 1
        >>> n_b = 2
        >>> n_f = 2
        >>> seq_len = 100
        >>> u_in = torch.ones((batch_size, seq_len))
        >>> y_0 = torch.zeros((batch_size, n_f))
        >>> u_0 = torch.zeros((batch_size, n_b))
        >>> b_coeff = np.array([0, 0.0706464146944544])  # b_0, b_1
        >>> a_coeff = np.array([-1.87212998940304, 0.942776404097492])  # f_1, f_2
        >>> G = LinearSiso(b_coeff, a_coeff)
        >>> y_out = G(u_in, y_0, u_0)
    """

    def __init__(self, in_channels, out_channels, n_b, n_a, n_k=0):
        super(LinearMimo, self).__init__()
        self.b_coeff = Parameter(torch.zeros(out_channels, in_channels, n_b))
        self.a_coeff = Parameter(torch.zeros(out_channels, in_channels, n_a))
        self.n_k = n_k

        with torch.no_grad():
            self.a_coeff[:] = torch.randn(self.a_coeff.shape) * 0.1
            self.b_coeff[:] = torch.randn(self.b_coeff.shape) * 0.1

    def forward(self, u_in, y_0=None, u_0=None):
        if self.n_k != 0:
            u_d = u_in.roll(self.n_k, dims=-2)  # roll on the time axis
            u_d[..., 0:self.n_k, :] = 0.0  # input sequence with delay
        else:
            u_d = u_in
        return LinearMimoFunction.apply(self.b_coeff, self.a_coeff, u_d, y_0, u_0)


class LinearSiso(LinearMimo):
    r"""Applies a Dynamical Linear MIMO system to an input signal.

    Args:
        b_coeff (np.array): Learnable coefficients of the transfer function numerator
        a_coeff (np.array): Learnable coefficients of the transfer function denominator
        n_k (int): Number of input delays

    Shape:
        - Input: :math:`(B, T)`
        - Output: :math:`(B, T)`

    Attributes:
        b_coeff (Tensor): the learnable coefficients of the transfer function numerator
        a_coeff (Tensor): the learnable coefficients of the transfer function denominator

    Examples::

        >>> batch_size = 1
        >>> n_b = 2
        >>> n_f = 2
        >>> seq_len = 100
        >>> u_in = torch.ones((batch_size, seq_len))
        >>> y_0 = torch.zeros((batch_size, n_f))
        >>> u_0 = torch.zeros((batch_size, n_b))
        >>> b_coeff = np.array([0, 0.0706464146944544])  # b_0, b_1
        >>> a_coeff = np.array([-1.87212998940304, 0.942776404097492])  # f_1, f_2
        >>> G = LinearSiso(b_coeff, a_coeff)
        >>> y_out = G(u_in, y_0, u_0)
    """

    def __init__(self, n_b, n_a, n_k=0):
        super(LinearSiso, self).__init__(1, 1, n_b=n_b, n_a=n_a, n_k=0)