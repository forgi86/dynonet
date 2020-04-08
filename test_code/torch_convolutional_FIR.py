import torch
torch.autograd

if __name__ == '__main__':

    # 1D Convolutional layer convention: B, F, T
    in_channels = 1
    out_channels = 1
    kernel_size = 128
    m = torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, bias=False, padding=kernel_size-1)#kernel_size)

    batch_size = 1
    seq_len = 5000
    u = torch.randn(batch_size, in_channels, seq_len)
    y = m(u)

    y1 = y[..., 0:-kernel_size+1]
    y2 = y1.transpose(-2, -1)

