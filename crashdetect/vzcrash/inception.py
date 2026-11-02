"""Single InceptionTime-style CNN, adapted for variable-length sensor clips.

Six multi-scale modules and two residual shortcuts, following Fawaz et al.
https://arxiv.org/abs/1909.04939 . This is one network, not the five-network
ensemble. Odd kernels (39/19/9) give symmetric padding; masked batch statistics
and masked global averaging exclude the variable-length batch padding.
"""
import torch
from torch import nn


class MaskedBatchNorm(nn.Module):
    def __init__(self, channels, momentum=.1, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.register_buffer("running_mean", torch.zeros(channels))
        self.register_buffer("running_var", torch.ones(channels))
        self.momentum, self.eps = momentum, eps

    def forward(self, x, valid):
        # Accumulate in float32 even when convolutions use mixed precision.
        values = x.float().masked_fill(~valid, 0)
        if self.training:
            count = valid.sum().clamp_min(2)
            mean = values.sum((0, 2)) / count
            centered = (values - mean[None, :, None]).masked_fill(~valid, 0)
            var = centered.square().sum((0, 2)) / count
            with torch.no_grad():
                self.running_mean.lerp_(mean.detach(), self.momentum)
                self.running_var.lerp_((var * count / (count - 1)).detach(), self.momentum)
        else:
            mean, var = self.running_mean, self.running_var
        result = (values - mean[None, :, None]) * torch.rsqrt(var[None, :, None] + self.eps)
        result = result * self.weight[None, :, None] + self.bias[None, :, None]
        return result.masked_fill(~valid, 0).to(x.dtype)


class InceptionModule(nn.Module):
    def __init__(self, channels, filters=32):
        super().__init__()
        self.bottleneck = nn.Conv1d(channels, 32, 1, bias=False)
        self.branches = nn.ModuleList([
            nn.Conv1d(32, filters, k, padding=k // 2, bias=False) for k in (39, 19, 9)
        ])
        self.pool = nn.MaxPool1d(3, stride=1, padding=1)
        self.pool_projection = nn.Conv1d(channels, filters, 1, bias=False)
        self.norm = MaskedBatchNorm(filters * 4)

    def forward(self, x, valid):
        x = x.masked_fill(~valid, 0)
        reduced = self.bottleneck(x).masked_fill(~valid, 0)
        pooled = self.pool(x.masked_fill(~valid, -torch.inf)).masked_fill(~valid, 0)
        branches = [layer(reduced) for layer in self.branches]
        branches.append(self.pool_projection(pooled))
        return self.norm(torch.cat(branches, dim=1), valid).relu()


class InceptionNet(nn.Module):
    def __init__(self, channels=7):
        super().__init__()
        self.blocks = nn.ModuleList([InceptionModule(channels if i == 0 else 128) for i in range(6)])
        self.shortcuts = nn.ModuleList([nn.Conv1d(channels, 128, 1, bias=False),
                                        nn.Conv1d(128, 128, 1, bias=False)])
        self.shortcut_norms = nn.ModuleList([MaskedBatchNorm(128) for _ in range(2)])
        self.head = nn.Linear(128, 1)

    def forward(self, x, lengths=None):
        if lengths is None:
            lengths = torch.full((len(x),), x.shape[1], dtype=torch.long, device=x.device)
        valid = (torch.arange(x.shape[1], device=x.device)[None, :] < lengths[:, None])[:, None, :]
        hidden = x.transpose(1, 2).masked_fill(~valid, 0)
        residual = hidden
        for i, block in enumerate(self.blocks):
            hidden = block(hidden, valid)
            if i % 3 == 2:
                shortcut = self.shortcut_norms[i // 3](self.shortcuts[i // 3](residual), valid)
                hidden = (hidden + shortcut).relu().masked_fill(~valid, 0)
                residual = hidden
        pooled = hidden.float().sum(2) / lengths[:, None]
        return self.head(pooled).squeeze(-1)
