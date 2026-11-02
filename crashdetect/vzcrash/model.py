"""Compact temporal classifiers; inspired by the literature, not reproductions."""
import math

import torch
from torch import nn


def make_model(channels=7, family="cnn_transformer"):
    if family == "inception_time":
        from .inception import InceptionNet
        return InceptionNet(channels)
    return ClipNet(channels, family)


class ClipNet(nn.Module):
    def __init__(self, channels=7, family="cnn_transformer"):
        super().__init__()
        if family not in ("cnn_transformer", "cnn_gru"):
            raise ValueError("Unknown model family")
        self.family = family
        self.encoder = nn.Sequential(
            nn.Conv1d(channels, 32, 9, padding=4), nn.GELU(), nn.MaxPool1d(4, ceil_mode=True),
            nn.Conv1d(32, 64, 7, padding=3), nn.GELU(), nn.MaxPool1d(4, ceil_mode=True),
            nn.Conv1d(64, 96, 5, padding=2), nn.GELU(), nn.MaxPool1d(2, ceil_mode=True))
        if family == "cnn_transformer":
            layer = nn.TransformerEncoderLayer(96, 4, 192, .1, batch_first=True, activation="gelu")
            self.temporal = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        else:
            self.temporal = nn.GRU(96, 96, num_layers=2, dropout=.1, batch_first=True)
        self.head = nn.Sequential(nn.Linear(192, 64), nn.GELU(), nn.Dropout(.1), nn.Linear(64, 1))

    def forward(self, x, lengths=None):
        if lengths is None:
            lengths = torch.full((len(x),), x.shape[1], dtype=torch.long, device=x.device)
        hidden = x.transpose(1, 2)
        stage_lengths = lengths
        for layer in self.encoder:
            mask = torch.arange(hidden.shape[2], device=x.device)[None, :] >= stage_lengths[:, None]
            if isinstance(layer, nn.MaxPool1d):
                hidden = layer(hidden.masked_fill(mask[:, None, :], -torch.inf))
                stage_lengths = (stage_lengths + layer.kernel_size - 1) // layer.kernel_size
                mask = torch.arange(hidden.shape[2], device=x.device)[None, :] >= stage_lengths[:, None]
                hidden = hidden.masked_fill(mask[:, None, :], 0)
            else:
                hidden = layer(hidden.masked_fill(mask[:, None, :], 0))
        tokens = hidden.transpose(1, 2)
        token_lengths = (lengths + 31) // 32
        padding = torch.arange(tokens.shape[1], device=x.device)[None, :] >= token_lengths[:, None]
        if self.family == "cnn_transformer":
            position = torch.arange(tokens.shape[1], device=x.device, dtype=x.dtype)[:, None]
            freq = torch.exp(torch.arange(0, 96, 2, device=x.device, dtype=x.dtype) * (-math.log(10000) / 96))
            encoding = torch.zeros(tokens.shape[1], 96, device=x.device, dtype=x.dtype)
            encoding[:, 0::2], encoding[:, 1::2] = torch.sin(position * freq), torch.cos(position * freq)
            tokens = self.temporal(tokens + encoding, src_key_padding_mask=padding)
        else:
            packed = nn.utils.rnn.pack_padded_sequence(tokens, token_lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed, _ = self.temporal(packed)
            tokens, _ = nn.utils.rnn.pad_packed_sequence(packed, batch_first=True, total_length=tokens.shape[1])
        mean = tokens.masked_fill(padding[:, :, None], 0).sum(1) / token_lengths[:, None]
        maximum = tokens.masked_fill(padding[:, :, None], -torch.inf).amax(1)
        return self.head(torch.cat([mean, maximum], dim=1)).squeeze(-1)
