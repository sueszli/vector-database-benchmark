""" This implementation is adapted from https://github.com/wenet-e2e/wespeaker.
"""
import torch
import torch.nn as nn

class TAP(nn.Module):
    """
    Temporal average pooling, only first-order mean is considered
    """

    def __init__(self, **kwargs):
        if False:
            while True:
                i = 10
        super(TAP, self).__init__()

    def forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        pooling_mean = x.mean(dim=-1)
        pooling_mean = pooling_mean.flatten(start_dim=1)
        return pooling_mean

class TSDP(nn.Module):
    """
    Temporal standard deviation pooling, only second-order std is considered
    """

    def __init__(self, **kwargs):
        if False:
            print('Hello World!')
        super(TSDP, self).__init__()

    def forward(self, x):
        if False:
            return 10
        pooling_std = torch.sqrt(torch.var(x, dim=-1) + 1e-08)
        pooling_std = pooling_std.flatten(start_dim=1)
        return pooling_std

class TSTP(nn.Module):
    """
    Temporal statistics pooling, concatenate mean and std, which is used in
    x-vector
    Comment: simple concatenation can not make full use of both statistics
    """

    def __init__(self, **kwargs):
        if False:
            return 10
        super(TSTP, self).__init__()

    def forward(self, x):
        if False:
            return 10
        pooling_mean = x.mean(dim=-1)
        pooling_std = torch.sqrt(torch.var(x, dim=-1) + 1e-08)
        pooling_mean = pooling_mean.flatten(start_dim=1)
        pooling_std = pooling_std.flatten(start_dim=1)
        stats = torch.cat((pooling_mean, pooling_std), 1)
        return stats

class ASTP(nn.Module):
    """ Attentive statistics pooling: Channel- and context-dependent
        statistics pooling, first used in ECAPA_TDNN.
    """

    def __init__(self, in_dim, bottleneck_dim=128, global_context_att=False):
        if False:
            return 10
        super(ASTP, self).__init__()
        self.global_context_att = global_context_att
        if global_context_att:
            self.linear1 = nn.Conv1d(in_dim * 3, bottleneck_dim, kernel_size=1)
        else:
            self.linear1 = nn.Conv1d(in_dim, bottleneck_dim, kernel_size=1)
        self.linear2 = nn.Conv1d(bottleneck_dim, in_dim, kernel_size=1)

    def forward(self, x):
        if False:
            return 10
        '\n        x: a 3-dimensional tensor in tdnn-based architecture (B,F,T)\n            or a 4-dimensional tensor in resnet architecture (B,C,F,T)\n            0-dim: batch-dimension, last-dim: time-dimension (frame-dimension)\n        '
        if len(x.shape) == 4:
            x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
        assert len(x.shape) == 3
        if self.global_context_att:
            context_mean = torch.mean(x, dim=-1, keepdim=True).expand_as(x)
            context_std = torch.sqrt(torch.var(x, dim=-1, keepdim=True) + 1e-10).expand_as(x)
            x_in = torch.cat((x, context_mean, context_std), dim=1)
        else:
            x_in = x
        alpha = torch.tanh(self.linear1(x_in))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        var = torch.sum(alpha * x ** 2, dim=2) - mean ** 2
        std = torch.sqrt(var.clamp(min=1e-10))
        return torch.cat([mean, std], dim=1)