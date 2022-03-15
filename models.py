import torch
import torch.nn as nn


class KB(nn.Module):

    # Expected input_dim is (num_c, h, w)
    def __init__(self, input_dim, output_dim, hidden_dims):
        super(KB, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims

        self.convs = []

        # TODO: In code, kernel size(and stride) is also flexible, but nothing of this sort was mentioned in the paper
        self.convs.append(nn.Conv2d(in_channels=input_dim.shape[0], out_channels=hidden_dims[0], kernel_size=3, stride=1, padding=1))
        for h in hidden_dims[1:]:
            self.convs.append(nn.Conv2d(in_channels=self.convs[-1].out_channels, out_channels=h, kernel_size=3, stride=1, padding=1))
        
        self.pooling = nn.MaxPool2d()
        self.activation = nn.ReLU()

    def __forward__(self, x, masks = None):
        # masks is a list/tensor of individual masks of hidden dimension size
        # Note: Last two dimensions can be one as in paper, or equal to hidden size

        for i, conv in enumerate(self.convs):
            x = self.pooling(self.activation(conv(x)))
            x = x*masks[i] if masks is not None else x
        return x