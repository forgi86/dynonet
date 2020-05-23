import torch
import torch.nn as nn


class MimoStaticNonLinearity(nn.Module):
    r"""Applies a Static MIMO non-linearity.
    The non-linearity is implemented as a feed-forward neural network.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        n_hidden (int, optional): Number of nodes in the hidden layer. Default: 20
        activation (str): Activation function. Either 'tanh', 'relu', or 'sigmoid'. Default: 'tanh'

    Shape:
        - Input: (..., in_channels)
        - Output: (..., out_channels)

    Examples::

        >>> in_channels, out_channels = 2, 4
        >>> F = MimoStaticNonLinearity(in_channels, out_channels)
        >>> batch_size, seq_len = 32, 100
        >>> u_in = torch.ones((batch_size, seq_len, in_channels))
        >>> y_out = F(u_in, y_0, u_0) # shape: (batch_size, seq_len, out_channels)
    """

    def __init__(self, in_channels, out_channels, n_hidden=20, activation='tanh'):
        super(MimoStaticNonLinearity, self).__init__()

        activation_dict = {'tanh': nn.Tanh, 'relu': nn.ReLU, 'sigmoid': nn.Sigmoid}

        self.net = nn.Sequential(
            nn.Linear(in_channels, n_hidden),
            activation_dict[activation](),  #nn.Tanh(),
            nn.Linear(n_hidden, out_channels)
        )

    def forward(self, u_lin):
        y_nl = self.net(u_lin)
        return y_nl


class SisoStaticNonLinearity(MimoStaticNonLinearity):
    r"""Applies a Static SISO non-linearity.
    The non-linearity is implemented as a feed-forward neural network.

    Args:
        n_hidden (int, optional): Number of nodes in the hidden layer. Default: 20
        activation (str): Activation function. Either 'tanh', 'relu', or 'sigmoid'. Default: 'tanh'
        s
    Shape:
        - Input: (..., in_channels)
        - Output: (..., out_channels)

    Examples::

        >>> F = SisoStaticNonLinearity(n_hidden=20)
        >>> batch_size, seq_len = 32, 100
        >>> u_in = torch.ones((batch_size, seq_len, in_channels))
        >>> y_out = F(u_in, y_0, u_0) # shape: (batch_size, seq_len, out_channels)
    """
    def __init__(self, n_hidden=20, activation='tanh'):
        super(SisoStaticNonLinearity, self).__init__(in_channels=1, out_channels=1, n_hidden=n_hidden, activation=activation)


class MimoChannelWiseNonLinearity(nn.Module):
    r"""Applies a Channel-wise non-linearity.
    The non-linearity is implemented as a set of feed-forward neural networks (each one operating on a different channel).

    Args:
        channels (int): Number of both input and output channels
        n_hidden (int, optional): Number of nodes in the hidden layer of each network. Default: 10

    Shape:
        - Input: (..., channels)
        - Output: (..., channels)

    Examples::

        >>> channels = 4
        >>> F = MimoChannelWiseNonLinearity(channels)
        >>> batch_size, seq_len = 32, 100
        >>> u_in = torch.ones((batch_size, seq_len, channels))
        >>> y_out = F(u_in, y_0, u_0) # shape: (batch_size, seq_len, channels)

    """

    def __init__(self, channels, n_hidden=10):
        super(MimoChannelWiseNonLinearity, self).__init__()

        self.net = nn.ModuleList()
        for channel_idx in range(channels):
            channel_net = nn.Sequential(
                nn.Linear(1, n_hidden),  # 2 states, 1 input
                nn.ReLU(),
                nn.Linear(n_hidden, 1)
            )
            self.net.append(channel_net)

    def forward(self, u_lin):

        y_nl = []
        for channel_idx, u_channel in enumerate(u_lin.split(1, dim=-1)):  # split over the last dimension (input channel)
            y_nl_channel = self.net[channel_idx](u_channel)  # Process blocks individually
            y_nl.append(y_nl_channel)

        y_nl = torch.cat(y_nl, -1)  # concatenate all output channels
        return y_nl


if __name__ == '__main__':

    channels = 4
    nn1 = MimoChannelWiseNonLinearity(channels)
    in_data = torch.randn(100, 10, channels)
    xx = net_out = nn1(in_data)