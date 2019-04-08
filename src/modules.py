import numpy as np
import torch
import torch.nn as nn

class Prenet(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.5, hidden_dims=[256,128]):
        super(Prenet, self).__init__()
        self.hidden_layer_1 = nn.Linear(input_dim, hidden_dims[0])
        self.hidden_layer_2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_x):
        h1 = self.dropout(self.relu(self.hidden_layer_1(input_x)))
        output = self.dropout(self.relu(self.hidden_layer_2(h1)))
        return output

class BatchNormConv1d(nn.Module):
    def __init__(self,
                 input_dim, output_dim,
                 kernel_size, stride, padding,
                 activation=None):
        super(BatchNormConv1d, self).__init__()
        self.conv1d = nn.Conv1d(
            input_dim, output_dim,
            kernel_size=kernel_size,
            stride=stride, padding=padding, bias=False
        )
        # Following tensorflow's default parameters
        self.bn = nn.BatchNorm1d(output_dim, momentum=0.99, eps=1e-3)
        self.activation = activation

    def forward(self, input_x):
        h1 = self.conv1d(input_x)
        if self.activation is not None:
            h1 = self.activation(h1)
        output = self.bn(h1)
        return output

class Highway(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Highway, self).__init__()
        self.H = nn.Linear(input_dim, output_dim)
        self.H.bias.data.zero_()
        self.T = nn.Linear(input_dim, output_dim)
        self.T.bias.data.fill_(-1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_x):
        H = self.relu(self.H(input_x))
        T = self.sigmoid(self.T(input_x))
        output = H * T + input_x * (1.0 - T)
        return output

class Conv1d_banks(nn.Module):
    def __init__(self, input_dim, K=16, conv1d_bank_hidden_dim=128):
        super(Conv1d_banks, self).__init__()
        self.conv1d_banks = nn.ModuleList()
        self.relu = nn.ReLU()
        for k in range(1, K+1):
            self.conv1d_banks.append(
                BatchNormConv1d(
                    input_dim, conv1d_bank_hidden_dim,
                    kernel_size=k, stride=1,
                    padding=k//2, activation=self.relu
                )
            )

    def forward(self, input_x, input_len):
        # input_x: [batch, input_dim, len]
        output = torch.cat(
            [conv1d(input_x)[:, :, :input_len] for conv1d in self.conv1d_banks],
            dim=1
        )
        return output

class CBHG(nn.Module):
    """
    CBHG module:
        - 1-d Convolution Banks
        - Highway networks + Residual connections
        - Bi-GRU
        ref: https://arxiv.org/pdf/1703.10135.pdf
    """
    def __init__(self, input_dim, K=16,
                 conv1d_bank_hidden_dim=128,
                 conv1d_projections_hidden_dim=128,
                 gru_dim=128):
        super(CBHG, self).__init__()
        self.input_dim = input_dim
        self.relu = nn.ReLU()
        self.K = K
        self.conv1d_banks = Conv1d_banks(input_dim, self.K, conv1d_bank_hidden_dim)
        self.max_pool1d = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)
        self.conv1d_projections_1 = BatchNormConv1d(
            K*input_dim, conv1d_projections_hidden_dim,
            kernel_size=3, stride=1, padding=1, activation=self.relu
        )
        self.conv1d_projections_2 = BatchNormConv1d(
            conv1d_projections_hidden_dim, conv1d_projections_hidden_dim,
            kernel_size=3, stride=1, padding=1, activation=None
        )
        self.pre_highway = nn.Linear(conv1d_projections_hidden_dim, input_dim, bias=False)
        self.highways = nn.ModuleList([Highway(input_dim, input_dim) for _ in range(4)])
        self.gru = nn.GRU(input_dim, gru_dim, 1, batch_first=True, bidirectional=True)

    def forward(self, input_x, input_lengths=None):
        # input_x: [batch, len, input_dim] -> [batch, input_dim, len]
        input_x_t = input_x.transpose(1, 2) if input_x.shape[-1] == self.input_dim else input_x
        input_len = input_x_t.shape[-1]

        # h1: [batch, input_dim*K, len]
        h1 = self.conv1d_banks(input_x_t, input_len)
        assert h1.shape[1] == self.input_dim * self. K

        # h2: [batch, input_dim*K//2, len]
        h2 = self.max_pool1d(h1)[:, :, :input_len]

        # h3: [batch, input_dim*K//2, len] -> [batch, len, input_dim]
        h3 = self.conv1d_projections_2(self.conv1d_projections_1(h2))
        h3 = h3.transpose(1, 2)
        if h3.shape[-1] != self.input_dim:
            h3 = self.pre_highway(h3)

        # Residual connection
        h4 = h3 + input_x
        for highway in self.highways:
            h4 = highway(h4)

        if input_lengths is not None:
            h4 = nn.utils.rnn.pack_padded_sequence(h4, input_lengths, batch_first=True)

        # output: [batch, len, gru_dim*2]
        output, _ = self.gru(h4)

        if input_lengths is not None:
            output, _ = nn.utils.rnn.pad_packed_sequence(
                output, batch_first=True)

        return output

class GLU(nn.Module):
    """
    Gated Linear Unit:
        A GLU is a data-driven activation function
        ref: https://arxiv.org/pdf/1711.11293.pdf
        Input: [batch, input_dim, len]
        Output: [batch, input_dim, len]
    """
    def __init__(self, input_dim):
        super(GLU, self).__init__()
        self.input_dim = input_dim
        self.W = nn.Linear(input_dim, input_dim)
        self.V = nn.Linear(input_dim, input_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_x):
        input_x = input_x.transpose(1, 2) if input_x.shape[-1] != self.input_dim else input_x
        output = self.W(input_x) * self.sigmoid(self.V(input_x))
        return output.transpose(1, 2)

class CIG_block(nn.Module):
    """
    CIG_block:
        - 1-d Conv
        - Instance Normalization
        - GLU
        ref: https://arxiv.org/pdf/1711.11293.pdf
    """
    def __init__(self, input_dim, output_dim, kernel_size, stride):
        super(CIG_block, self).__init__()
        self.conv1d = nn.Conv1d(input_dim, output_dim, kernel_size, stride)
        self.instance_norm = nn.InstanceNorm1d(output_dim)
        self.GLU = GLU(output_dim)

    def forward(self, input_x):
        h1 = self.conv1d(input_x)
        # h1: [batch, channel, len]
        h2 = self.instance_norm(h1)
        output = self.GLU(h2)
        return output
