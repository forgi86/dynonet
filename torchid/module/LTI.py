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

# SISO is implemented as a sub-case of MIMO
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

#    def __getattr__(self, item):
#        if item == 'b_coeff':
#            return self.b_coeff[0, 0, :]

#        if item == 'a_coeff':
#            return self.b_coeff[0, 0, :]

#        return self.__getattribute__(item)

class LinearMimoFir(torch.nn.Module):
    def __init__(self, in_channels, out_channels, n_b, channels_last=True):
        super(LinearMimoFir, self).__init__()
        self.G = torch.nn.Conv1d(in_channels, out_channels, kernel_size=n_b, bias=False, padding=n_b-1)
        self.b_coeff = self.G.weight
        self.n_b = n_b
        self.channels_last = channels_last

    def forward(self, u_in):
        if self.channels_last:
            u_torch = u_in.transpose(-2, -1)

        y_out = self.G(u_torch)
        y_out = y_out[..., 0:-self.n_b+1]

        if self.channels_last:
            y_out = y_out.transpose(-2, -1)
        return y_out


class LinearSecondOrderMimo(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearSecondOrderMimo, self).__init__()
        self.b_coeff = Parameter(torch.zeros(out_channels, in_channels, 2))
        self.rho = Parameter(torch.zeros(out_channels, in_channels, 1))
        self.psi = Parameter(torch.zeros((out_channels, in_channels, 1)))
        with torch.no_grad():
            self.rho[:] = torch.randn(self.rho.shape) * 0.1
            self.psi[:] = torch.randn(self.rho.shape) * 0.1
            self.b_coeff[:] = torch.randn(self.b_coeff.shape) * 0.01

    def forward(self, u_in, y_0=None, u_0=None):
        r = torch.sigmoid(self.rho)
        theta = np.pi * torch.sigmoid(self.psi)
        a_1 = -2 * r * torch.cos(theta)
        a_2 = r ** 2
        a_coeff = torch.cat((a_1, a_2), dim=-1)
        return LinearMimoFunction.apply(self.b_coeff, a_coeff, u_in, y_0, u_0)


class LinearSecondOrderSiso(torch.nn.Module):
    def __init__(self):
        super(LinearSecondOrderSiso, self).__init__(1, 1, 2, 2) # in_channels, out_channels, n_b, n_a




class LinearSisoFir(LinearMimoFir):
    def __init__(self, n_b, channels_last=True):
        super(LinearSisoFir, self).__init__(1, 1, n_b, channels_last=channels_last)