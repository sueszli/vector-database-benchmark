import torch
import torch.nn as nn
from fairseq.modules import Fp32GroupNorm

class KmeansVectorQuantizer(nn.Module):

    def __init__(self, dim, num_vars, groups, combine_groups, vq_dim, time_first, gamma=0.25):
        if False:
            print('Hello World!')
        'Vector quantization using straight pass-through estimator (i.e. kmeans)\n\n        Args:\n            dim: input dimension (channels)\n            num_vars: number of quantized vectors per group\n            groups: number of groups for vector quantization\n            combine_groups: whether to use the vectors for all groups\n            vq_dim: dimensionality of the resulting quantized vector\n            time_first: if true, expect input in BxTxC format, otherwise in BxCxT\n            gamma: commitment loss coefficient\n        '
        super().__init__()
        self.groups = groups
        self.combine_groups = combine_groups
        self.input_dim = dim
        self.num_vars = num_vars
        self.vq_dim = vq_dim
        self.time_first = time_first
        assert vq_dim % groups == 0, f'dim {vq_dim} must be divisible by groups {groups} for concatenation'
        self.var_dim = vq_dim // groups
        num_groups = groups if not combine_groups else 1
        self.embedding = nn.Parameter(0.01 * torch.randn(num_vars, num_groups, self.var_dim))
        self.projection = nn.Sequential(nn.Conv1d(dim, dim, kernel_size=1, groups=groups, bias=False), Fp32GroupNorm(groups, dim))
        self.gamma = gamma
        self.mse_mean = nn.MSELoss(reduction='mean')

    def _pass_grad(self, x, y):
        if False:
            i = 10
            return i + 15
        'Manually set gradient for backward pass.\n        for y = f(x), ensure that during the backward pass,\n        dL/dy = dL/dx regardless of f(x).\n        Returns:\n            y, with the gradient forced to be dL/dy = dL/dx.\n        '
        return y.detach() + (x - x.detach())

    @property
    def expand_embedding(self):
        if False:
            return 10
        if self.combine_groups:
            return self.embedding.expand(self.num_vars, self.groups, self.var_dim)
        return self.embedding

    def forward_idx(self, x):
        if False:
            i = 10
            return i + 15
        res = self.forward(x, produce_targets=True)
        return (res['x'], res['targets'])

    def forward(self, x, produce_targets=False):
        if False:
            return 10
        result = {'num_vars': self.num_vars}
        if self.time_first:
            x = x.transpose(1, 2)
        (bsz, fsz, tsz) = x.shape
        ze = self.projection(x)
        ze_ = ze.view(bsz, self.groups, self.var_dim, tsz).permute(0, 3, 1, 2)
        d = (ze_.unsqueeze(0) - self.expand_embedding.unsqueeze(1).unsqueeze(1)).view(self.num_vars, bsz, tsz, self.groups, -1).norm(dim=-1, p=2)
        idx = d.argmin(dim=0)
        zq = torch.stack([self.expand_embedding[idx[..., group], group] for group in range(self.groups)], dim=-2).view(bsz, tsz, self.groups * self.var_dim).permute(0, 2, 1)
        assert ze.shape == zq.shape, (ze.shape, zq.shape)
        x = self._pass_grad(ze, zq)
        with torch.no_grad():
            hard_x = idx.new_zeros(bsz * tsz * self.groups, self.num_vars).scatter_(-1, idx.view(-1, 1), 1.0).view(bsz * tsz, self.groups, -1)
            hard_probs = torch.mean(hard_x.float(), dim=0)
            result['code_perplexity'] = torch.exp(-torch.sum(hard_probs * torch.log(hard_probs + 1e-07), dim=-1)).sum()
        if produce_targets:
            result['targets'] = idx
        if self.time_first:
            x = x.transpose(1, 2)
        result['x'] = x
        ze = ze.float()
        zq = zq.float()
        latent_loss = self.mse_mean(zq, ze.detach())
        commitment_loss = self.mse_mean(ze, zq.detach())
        result['kmeans_loss'] = latent_loss + self.gamma * commitment_loss
        return result