import torch
import numpy as np  # temporary
from torch.nn.parameter import Parameter
from torchid.functional.linearsiso import LinearSisoFunction
from torchid.functional.old.linearmimo_channels_first import LinearMimoFunction


class LinearSiso(torch.nn.Module):
    r"""Applies a Dynamical Linear SISO system to an input signal.

    Args:
        b_coeff (np.array): Learnable coefficients of the transfer function numerator
        a_coeff (np.array): Learnable coefficients of the transfer function denominator

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
    def __init__(self, b_coeff, a_coeff):
        super(LinearSiso, self).__init__()
        self.b_coeff = Parameter(torch.tensor(b_coeff))
        self.a_coeff = Parameter(torch.tensor(a_coeff))

    def forward(self, u_in, y_init, u_init):
        return LinearSisoFunction.apply(self.b_coeff, self.a_coeff, u_in, y_init, u_init)


class SecondOrderOscillator(torch.nn.Module):
    def __init__(self, b_coeff, rho, psi):
        super(SecondOrderOscillator, self).__init__()
        self.b_coeff = Parameter(torch.tensor(b_coeff))
        self.rho = Parameter(torch.tensor(rho))
        self.psi = Parameter(torch.tensor(psi))

    def forward(self, u_in, y_init, u_init):

        r = torch.sigmoid(self.rho)
        theta = np.pi * torch.sigmoid(self.psi)
        a_1 = -2 * r * torch.cos(theta)
        a_2 = r ** 2
        a_coeff = torch.stack((a_1, a_2))
        return LinearSisoFunction.apply(self.b_coeff, a_coeff, u_in, y_init, u_init)


class LinearMimo(torch.nn.Module):
    r"""Applies a Dynamical Linear MIMO system to an input signal.

    Args:
        b_coeff (np.array): Learnable coefficients of the transfer function numerator
        a_coeff (np.array): Learnable coefficients of the transfer function denominator

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
    def __init__(self, in_channels, out_channels, n_b, n_a):
        super(LinearMimo, self).__init__()
        self.b_coeff = Parameter(torch.zeros(out_channels, in_channels, n_b))
        self.a_coeff = Parameter(torch.zeros(out_channels, in_channels, n_a))

    def forward(self, u_in, y_0, u_0):
        return LinearMimoFunction.apply(self.b_coeff, self.a_coeff, u_in, y_0, u_0)


