import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Encoder(nn.Module):
    """A fully-connected encoder for conditioning information."""
    def __init__(self, context_dim, encoder_units, n_encoder_layers,
        encoder_dropout):
        super().__init__()
        self.context_dim = context_dim
        self.encoder_units = encoder_units
        self.n_encoder_layers = n_encoder_layers
        self.encoder_dropout = encoder_dropout

        self.layers = nn.ModuleList()
        self.layers.append(self.encoder_layer(context_dim, encoder_units))
        self.layers.extend(
            [self.encoder_layer(encoder_units, encoder_units)
            for _ in range(n_encoder_layers - 1)])

    def encoder_layer(self, in_units, out_units):
        layer = nn.Sequential(
            nn.Linear(in_units, out_units),
            nn.PReLU(),
            nn.BatchNorm1d(out_units),
            nn.Dropout(self.encoder_dropout))
        return layer

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class CNNEncoder(nn.Module):
    """A 1-D CNN encoder for conditioning information."""
    def __init__(self, context_dim, encoder_units, n_encoder_layers,
        encoder_dropout, subsample):
        super().__init__()
        self.context_dim = context_dim
        self.encoder_units = encoder_units
        self.n_encoder_layers = n_encoder_layers
        self.encoder_dropout = encoder_dropout
        self.subsample = subsample

        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(self.encoder_layer(1, 32, 9, 5))
        self.conv_layers.extend(
            [self.encoder_layer(32, 32, 3, 1)
            for _ in range(n_encoder_layers - 1)])
        output_size = self.compute_output_size(context_dim, 1, 1, kernel_size, 5)
        self.fc = nn.Sequential(
            nn.Linear(32*output_size, encoder_units),
            nn.PReLU(),
            nn.BatchNorm1d(encoder_units),
            nn.Dropout(self.encoder_dropout))

    def compute_output_size(self, l_in, padding, dilation, kernel_size, stride):
        arg = (l_in + 2*padding - dilation*(kernel_size - 1) - 1) / stride + 1
        return int(np.floor(arg))

    def encoder_layer(self, in_channels, out_channels, kernel_size, stride):
        layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels,
                kernel_size, stride, padding=1),
            nn.PReLU(),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(self.encoder_dropout))
        return layer

    def forward(self, x):
        bsz = x.size(0)
        x = x.unsqueeze(1)
        for layer in self.conv_layers:
            x = layer(x)
        x = x.view(bsz, -1)
        x = self.fc(x)
        return x