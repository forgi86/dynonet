import torch
import numpy as np  # temporary
from torch.nn.parameter import Parameter
from torchid.functional.lti import MimoLinearDynamicOperatorFun


class MimoLinearDynamicOperator(torch.nn.Module):
    r"""Applies a Dynamical Linear MIMO system to an input signal.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        n_b (int): Number of learnable coefficients of the transfer function numerator
        n_a (int): Number of learnable coefficients of the transfer function denominator
        n_k (int, optional): Number of input delays in the numerator. Default: 0

    Shape:
        - Input: (batch_size, seq_len, in_channels)
        - Output: (batch_size, seq_len, out_channels)

    Attributes:
        b_coeff (Tensor): The learnable coefficients of the transfer function numerator
        a_coeff (Tensor): The learnable coefficients of the transfer function denominator

    Examples::

        >>> in_channels, out_channels = 2, 4
        >>> n_b, n_a, n_k = 2, 2, 1
        >>> G = MimoLinearDynamicOperator(in_channels, out_channels, n_b, n_a, n_k)
        >>> batch_size, seq_len = 32, 100
        >>> u_in = torch.ones((batch_size, seq_len, in_channels))
        >>> y_out = G(u_in, y_0, u_0) # shape: (batch_size, seq_len, out_channels)
    """

    def __init__(self, in_channels, out_channels, n_b, n_a, n_k=0):
        super(MimoLinearDynamicOperator, self).__init__()
        self.b_coeff = Parameter(torch.zeros(out_channels, in_channels, n_b))
        self.a_coeff = Parameter(torch.zeros(out_channels, in_channels, n_a))
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.n_a = n_a
        self.n_b = n_b
        self.n_k = n_k

        with torch.no_grad():
            init_range = 0.01
            self.a_coeff[:] = (torch.rand(self.a_coeff.shape) - 0.5) * 2 * init_range
            self.b_coeff[:] = (torch.rand(self.b_coeff.shape) - 0.5) * 2 * init_range

    def forward(self, u_in, y_0=None, u_0=None):
        if self.n_k != 0:
            #u_d = u_in.roll(self.n_k, dims=-2)  # roll on the time axis
            #u_d[..., 0:self.n_k, :] = 0.0  # input sequence with delay
            u_d = torch.empty_like(u_in)
            u_d[..., self.n_k:, :] = u_in[:, :-self.n_k, :]
            u_d[..., 0:self.n_k, :] = 0.0
        else:
            u_d = u_in
        return MimoLinearDynamicOperatorFun.apply(self.b_coeff, self.a_coeff, u_d, y_0, u_0)

    def get_filtdata(self):
        r"""Returns the numerator and denominator coefficients of the transfer function :math:`q^{-1}`-polynomials.

        The polynomials are function of the variable :math:`q^{-1}`.
        The polynomial coefficients b and a have length m and n, respectively and are sorted in descending power order.

        For a certain input channel :math:`i` and output channel :math:`o`, the  corresponding transfer
        function :math:`G_{i\rightarrow o}(z)` is:

        .. math::
            G_{i\rightarrow o}(z) = q^{-n_k}\frac{b[o, i, 0] + b[o, i, 1]q^{-1} + \dots + b[o, i, n]q^{-m+1}}
            {a[o, i, 0] + a[o, i, 1]q^{-1} + \dots + a[o, i, n]q^{-n+1}}

        Returns:
            np.array(in_channels, out_channels, m), np.array(in_channels, out_channels, n):
                numerator :math:`\beta` and denominator :math:`\alpha` polynomial coefficients of the transfer function.


        Example::

            >>> num, den = G.get_tfdata()
            >>> G_tf = control.TransferFunction(G2_num, G2_den, ts=1.0)
        """
        return self.__get_filtdata__()

    def get_tfdata(self):
        r"""Returns the numerator and denominator coefficients of the transfer function :math:`z`-polynomials.

        The polynomials are function of the variable Z-transform variable :math:`z`.
        The polynomial coefficients :math::`\beta` and :math:`\alpha` have equal length p and are sorted in descending power order.

        For a certain input channel :math:`i` and output channel :math:`o`, the  corresponding transfer
        function :math:`G_{i\rightarrow o}(z)` is:

        .. math::
            G_{i\rightarrow o}(z) = \frac{\beta[o, i, 0]z^{n-1} + \beta[o, i, 1]z^{n-1} + \dots + \beta[o, i, p]}{\alpha[o, i, 0]z^{n-1} + \alpha[o, i, 1]z^{n-2} + \dots + \alpha[o, i, p]}

        Returns:
            np.array(in_channels, out_channels, p), np.array(in_channels, out_channels, p):
                numerator :math:`\beta` and denominator :math:`\alpha` polynomial coefficients of the transfer function.


        Example::

            >>> num, den = G.get_tfdata()
            >>> G_tf = control.TransferFunction(G2_num, G2_den, ts=1.0)
        """
        return self.__get_tfdata__()

    def __get_filtdata__(self):
        # returns the coefficients of the polynomials b and a as function of q^{-1}
        b_coeff_np, a_coeff_np = self.__get_ba_coeff__()
        b_seq = np.zeros_like(b_coeff_np, shape=(self.out_channels, self.in_channels, self.n_b + self.n_k))  #b_coeff_np
        b_seq[:, :, self.n_k:] = b_coeff_np[:, :, :]
        a_seq = np.empty_like(a_coeff_np, shape=(self.out_channels, self.in_channels, self.n_a + 1))
        a_seq[:, :, 0] = 1
        a_seq[:, :, 1:] = a_coeff_np[:, :, :]
        return b_seq, a_seq

    def __get_tfdata__(self):
        b_seq, a_seq = self.__get_filtdata__()
        M = self.n_b + self.n_k  # number of numerator coefficients of the q^{-1} polynomial
        N = self.n_a + 1  # number of denominator coefficients of the q^{-1} polynomial
        if M > N:
            num = b_seq
            den = np.c_[a_seq, np.zeros((self.out_channels, self.in_channels, M - N))]
        elif N > M:
            num = np.c_[b_seq, np.zeros((self.out_channels, self.in_channels, N - M))]
            den = a_seq
        else:  # N == M
            num = b_seq
            den = a_seq

        return num, den

    def __get_ba_coeff__(self):
        return self.b_coeff.detach().numpy(), self.a_coeff.detach().numpy()


# SISO is implemented as a sub-case of MIMO
class SisoLinearDynamicOperator(MimoLinearDynamicOperator):
    r"""Applies a Dynamical Linear SISO system to an input signal.

    Args:
        n_b (int): Number of learnable coefficients of the transfer function numerator
        n_a (int): Number of learnable coefficients of the transfer function denominator
        n_k (int, optional): Number of input delays in the numerator. Default: 0

    Shape:
        - Input: (batch_size, seq_len, 1)
        - Output: (batch_size, seq_len, 1)

    Attributes:
        b_coeff (Tensor): the learnable coefficients of the transfer function numerator
        a_coeff (Tensor): the learnable coefficients of the transfer function denominator

    Examples::

        >>> n_b, n_a = 2, 2
        >>> G = SisoLinearDynamicOperator(b_coeff, a_coeff)
        >>> batch_size, seq_len = 32, 100
        >>> u_in = torch.ones((batch_size, seq_len))
        >>> y_out = G(u_in, y_0, u_0) # shape: (batch_size, seq_len, 1)
    """

    def __init__(self, n_b, n_a, n_k=0):
        super(SisoLinearDynamicOperator, self).__init__(1, 1, n_b=n_b, n_a=n_a, n_k=n_k)

    def get_filtdata(self):
        b_seq, a_seq = super(SisoLinearDynamicOperator, self).__get_filtdata__()  # MIMO numden
        return b_seq[0, 0, :], a_seq[0, 0, :]

    def get_tfdata(self):
        num, den = super(SisoLinearDynamicOperator, self).__get_tfdata__()  # MIMO numden
        return num[0, 0, :], den[0, 0, :]


class MimoFirLinearDynamicOperator(torch.nn.Module):
    r"""Applies a Linear MIMO FIR filter to an input signal.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        n_b (int): Number of learnable FIR coefficients

    Shape:
        - Input: (batch_size, seq_len, in_channels)
        - Output: (batch_size, seq_len, out_channels)

    Attributes:
        G (torch.nn.Conv1d): The underlying Conv1D object used to implement the convolution

    Examples::

        >>> in_channels, out_channels = 2, 4
        >>> n_b = 128
        >>> G = MimoLinearDynamicOperator(in_channels, out_channels, n_b)
        >>> batch_size, seq_len = 32, 100
        >>> u_in = torch.ones((batch_size, seq_len, in_channels))
        >>> y_out = G(u_in, y_0, u_0) # shape: (batch_size, seq_len, out_channels)
    """

    def __init__(self, in_channels, out_channels, n_b, channels_last=True):
        super(MimoFirLinearDynamicOperator, self).__init__()
        self.G = torch.nn.Conv1d(in_channels, out_channels, kernel_size=n_b, bias=False, padding=n_b-1)
        #self.b_coeff = self.G.weight
        self.n_a = 0
        self.n_b = n_b
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.channels_last = channels_last

    def forward(self, u_in):
        # PyTorch 1.4 does not have a channels_last option for Conv1d, which is more convenient
        # in our block-oriented modeling framework.
        # Then, let us transpose the last and second last channels manually before and after applying torch.nn.Conv1d
        if self.channels_last:
            u_torch = u_in.transpose(-2, -1)

        y_out = self.G(u_torch)
        y_out = y_out[..., 0:-self.n_b+1]

        if self.channels_last:
            y_out = y_out.transpose(-2, -1)
        return y_out

    def get_filtdata(self):
        return self.__get_filtdata__()

    def get_tfdata(self):
        return self.__get_tfdata__()

    def __get_filtdata__(self):
        b_coeff, a_coeff = self.__get_ba_coeff__()
        b_seq = b_coeff
        a_seq = np.empty_like(a_coeff, shape=(self.out_channels, self.in_channels, self.n_a + 1))
        a_seq[:, :, 0] = 1
        a_seq[:, :, 1:] = a_coeff[:, :, :]
        return b_seq, a_seq

    def __get_tfdata__(self):
        b_seq, a_seq = self.__get_filtdata__()
        M = self.n_b  # numerator coefficients
        N = self.n_a + 1  # denominator coefficients
        if M > N:
            num = b_seq
            den = np.c_[a_seq, np.zeros((self.out_channels, self.in_channels, M - N))]
        elif N > M:
            num = np.c_[self.b_poly, np.zeros((self.out_channels, self.in_channels, N - M))]
            den = a_seq
        else:  # N == M
            num = b_seq
            den = a_seq

        return num, den

    def __get_ba_coeff__(self):
        b_coeff_np = self.G.weight.detach().numpy()
        b_coeff_np = b_coeff_np[:, :, ::-1]
        a_coeff_np = np.zeros_like(b_coeff_np, shape=(self.out_channels, self.in_channels, 0))
        return b_coeff_np, a_coeff_np


class SisoFirLinearDynamicOperator(MimoFirLinearDynamicOperator):
    r"""Applies a Linear SISO FIR filter to an input signal.

    Args:
        n_b (int): Number of learnable FIR coefficients

    Shape:
        - Input: (batch_size, seq_len, 1)
        - Output: (batch_size, seq_len, 1)

    Attributes:
        G (torch.nn.Conv1d): The underlying Conv1D object used to implement the convolution

    Examples::

        >>> n_b = 128
        >>> G = SisoFirLinearDynamicOperator(n_b)
        >>> batch_size, seq_len = 32, 100
        >>> u_in = torch.ones((batch_size, seq_len, 1))
        >>> y_out = G(u_in, y_0, u_0) # shape: (batch_size, seq_len, 1)
    """
    def __init__(self, n_b, channels_last=True):
        super(SisoFirLinearDynamicOperator, self).__init__(1, 1, n_b, channels_last=channels_last)

    def get_filtdata(self):
        b_seq, a_seq = super(SisoFirLinearDynamicOperator, self).__get_filtdata__() # call to MIMO ba
        return b_seq[0, 0, :], a_seq[0, 0, :]

    def get_tfdata(self):
        num, den = super(SisoFirLinearDynamicOperator, self).__get_tfdata__() # call to MIMO numden
        return num[0, 0, :], den[0, 0, :]


class MimoSecondOrderDynamicOperator(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MimoSecondOrderDynamicOperator, self).__init__()
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
        return MimoLinearDynamicOperatorFun.apply(self.b_coeff, a_coeff, u_in, y_0, u_0)


class SisoSecondOrderDynamicOperator(MimoSecondOrderDynamicOperator):
    def __init__(self):
        super(SisoSecondOrderDynamicOperator, self).__init__(1, 1) # in_channels, out_channels, n_b, n_a
