import torch

if __name__ == '__main__':

    # 1D Convolutional layer convention: B, F, T
    in_channels = 16
    out_channels = 33
    m = torch.nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2)

    batch_size = 20
    seq_len = 50
    input = torch.randn(batch_size, in_channels, seq_len)
    output = m(input)


    # RNN layer convention: B, T, F
    input_size = 10
    hidden_size = 20
    num_layers = 2
    rnn = torch.nn.GRU(input_size, hidden_size, num_layers)

    seq_len = 5
    batch_size = 3
    input_size = 10
    input = torch.randn(seq_len, batch_size, input_size)
    h0 = torch.randn(num_layers, batch_size, hidden_size)
    output, hn = rnn(input, h0)  # seq_len, batch_size, hidden_size
