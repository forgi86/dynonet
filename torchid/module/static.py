import torch
import torch.nn as nn


class StaticMimoNonLin(nn.Module):

    def __init__(self, in_channels, out_channels, n_hidden=20):
        super(StaticMimoNonLin, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_channels, n_hidden),  # 2 states, 1 input
            nn.ReLU(),
            nn.Linear(n_hidden, out_channels)
        )

    def forward(self, u_lin):

        y_nl = self.net(u_lin)  # Process blocks individually
        return y_nl


class StaticSisoNonLin(StaticMimoNonLin):
    def __init__(self, n_hidden=20):
        super(StaticSisoNonLin, self).__init__(in_channels=1, out_channels=1, n_hidden=n_hidden)


class StaticChannelWiseNonLin(nn.Module):

    def __init__(self, channels, n_hidden=10):
        super(StaticChannelWiseNonLin, self).__init__()

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
    nn1 = StaticChannelWiseNonLin(channels)
    in_data = torch.randn(100, 10, channels)
    xx = net_out = nn1(in_data)