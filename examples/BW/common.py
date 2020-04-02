import torch
import torch.nn as nn


class StaticNonLin(nn.Module):

    def __init__(self):
        super(StaticNonLin, self).__init__()

        self.net_1 = nn.Sequential(
            nn.Linear(1, 20),  # 2 states, 1 input
            nn.ReLU(),
            nn.Linear(20, 1)
        )

        self.net_2 = nn.Sequential(
            nn.Linear(1, 20),  # 2 states, 1 input
            nn.ReLU(),
            nn.Linear(20, 1)
        )

    def forward(self, u_lin):

        y_nl_1 = self.net_1(u_lin[..., [0]])  # Process blocks individually
        y_nl_2 = self.net_2(u_lin[..., [1]])  # Process blocks individually
        y_nl = torch.cat((y_nl_1, y_nl_2), dim=-1)

        return y_nl


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


class StaticChannelWiseNonLin(nn.Module):

    def __init__(self, channels, n_hidden=10):
        super(StaticChannelWiseNonLin, self).__init__()

        self.net = []
        for channel_idx in range(channels):
            channel_net = nn.Sequential(
                nn.Linear(1, n_hidden),  # 2 states, 1 input
                nn.ReLU(),
                nn.Linear(n_hidden, 1)
            )
            self.net.append(channel_net)

    def forward(self, u_lin):

        y_nl = []
        for channel_idx, u_channel in enumerate(u_lin.split(1, dim=-1)):
            y_nl_channel = self.net[channel_idx](u_channel)  # Process blocks individually
            y_nl.append(y_nl_channel)

        y_nl = torch.stack(y_nl, -1)
        return y_nl


if __name__ == '__main__':

    channels = 4
    nn1 = StaticChannelWiseNonLin(channels)
    in_data = torch.randn(100, 10, channels)
    net_out = nn1(in_data)
