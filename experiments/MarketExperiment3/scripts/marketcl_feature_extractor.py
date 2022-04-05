from dataloaders import get_marketcl_dataloader
import torch
from torch import nn
from torch import optim


class FeatExtractorLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, lr):
        super().__init__()
        ### INSERT YOUR CODE HERE
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear_1 = nn.Linear(hidden_size, output_size)
        self.linear_2 = nn.Linear(hidden_size, output_size)
        self.linear_3 = nn.Linear(hidden_size, output_size)
        self.linear_4 = nn.Linear(hidden_size, output_size)
        self.logsoft = nn.LogSoftmax(dim=-1)
        self.optimizer = optim.SGD(self.parameters(), lr=lr)

    def forward(self, x, lengths):
        if self.input_size == 1: x = x.unsqueeze(-1)
        x = x.float()
        packed_seq = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(packed_seq,)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x_final = x[:,-1,:]
        return x_final

