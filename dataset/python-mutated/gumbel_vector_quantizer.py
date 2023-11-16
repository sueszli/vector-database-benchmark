import torch
import torch.nn as nn
import torch.nn.functional as F

class GumbelVectorQuantizer(nn.Module):

    def __init__(self, dim, num_vars, temp, groups, combine_groups, vq_dim, time_first, activation=nn.GELU(), weight_proj_depth=1, weight_proj_factor=1, hard=True, std=0):
        if False:
            print('Hello World!')
        'Vector quantization using gumbel softmax\n\n        Args:\n            dim: input dimension (channels)\n            num_vars: number of quantized vectors per group\n            temp: temperature for training. this should be a tuple of 3 elements: (start, stop, decay factor)\n            groups: number of groups for vector quantization\n            combine_groups: whether to use the vectors for all groups\n            vq_dim: dimensionality of the resulting quantized vector\n            time_first: if true, expect input in BxTxC format, otherwise in BxCxT\n            activation: what activation to use (should be a module). this is only used if weight_proj_depth is > 1\n            weight_proj_depth: number of layers (with activation in between) to project input before computing logits\n            weight_proj_factor: this is used only if weight_proj_depth is > 1. scales the inner dimensionality of\n                                projections by this factor\n        '
        super().__init__()
        self.groups = groups
        self.combine_groups = combine_groups
        self.input_dim = dim
        self.num_vars = num_vars
        self.time_first = time_first
        self.hard = hard
        assert vq_dim % groups == 0, f'dim {vq_dim} must be divisible by groups {groups} for concatenation'
        var_dim = vq_dim // groups
        num_groups = groups if not combine_groups else 1
        self.vars = nn.Parameter(torch.FloatTensor(1, num_groups * num_vars, var_dim))
        if std == 0:
            nn.init.uniform_(self.vars)
        else:
            nn.init.normal_(self.vars, mean=0, std=std)
        if weight_proj_depth > 1:

            def block(input_dim, output_dim):
                if False:
                    i = 10
                    return i + 15
                return nn.Sequential(nn.Linear(input_dim, output_dim), activation)
            inner_dim = self.input_dim * weight_proj_factor
            self.weight_proj = nn.Sequential(*[block(self.input_dim if i == 0 else inner_dim, inner_dim) for i in range(weight_proj_depth - 1)], nn.Linear(inner_dim, groups * num_vars))
        else:
            self.weight_proj = nn.Linear(self.input_dim, groups * num_vars)
            nn.init.normal_(self.weight_proj.weight, mean=0, std=1)
            nn.init.zeros_(self.weight_proj.bias)
        if isinstance(temp, str):
            import ast
            temp = ast.literal_eval(temp)
        assert len(temp) == 3, f'{temp}, {len(temp)}'
        (self.max_temp, self.min_temp, self.temp_decay) = temp
        self.curr_temp = self.max_temp
        self.codebook_indices = None

    def set_num_updates(self, num_updates):
        if False:
            while True:
                i = 10
        self.curr_temp = max(self.max_temp * self.temp_decay ** num_updates, self.min_temp)

    def get_codebook_indices(self):
        if False:
            while True:
                i = 10
        if self.codebook_indices is None:
            from itertools import product
            p = [range(self.num_vars)] * self.groups
            inds = list(product(*p))
            self.codebook_indices = torch.tensor(inds, dtype=torch.long, device=self.vars.device).flatten()
            if not self.combine_groups:
                self.codebook_indices = self.codebook_indices.view(self.num_vars ** self.groups, -1)
                for b in range(1, self.groups):
                    self.codebook_indices[:, b] += self.num_vars * b
                self.codebook_indices = self.codebook_indices.flatten()
        return self.codebook_indices

    def codebook(self):
        if False:
            return 10
        indices = self.get_codebook_indices()
        return self.vars.squeeze(0).index_select(0, indices).view(self.num_vars ** self.groups, -1)

    def sample_from_codebook(self, b, n):
        if False:
            while True:
                i = 10
        indices = self.get_codebook_indices()
        indices = indices.view(-1, self.groups)
        cb_size = indices.size(0)
        assert n < cb_size, f'sample size {n} is greater than size of codebook {cb_size}'
        sample_idx = torch.randint(low=0, high=cb_size, size=(b * n,))
        indices = indices[sample_idx]
        z = self.vars.squeeze(0).index_select(0, indices.flatten()).view(b, n, -1)
        return z

    def to_codebook_index(self, indices):
        if False:
            while True:
                i = 10
        res = indices.new_full(indices.shape[:-1], 0)
        for i in range(self.groups):
            exponent = self.groups - i - 1
            res += indices[..., i] * self.num_vars ** exponent
        return res

    def forward_idx(self, x):
        if False:
            print('Hello World!')
        res = self.forward(x, produce_targets=True)
        return (res['x'], res['targets'])

    def forward(self, x, produce_targets=False):
        if False:
            print('Hello World!')
        result = {'num_vars': self.num_vars * self.groups}
        if not self.time_first:
            x = x.transpose(1, 2)
        (bsz, tsz, fsz) = x.shape
        x = x.reshape(-1, fsz)
        x = self.weight_proj(x)
        x = x.view(bsz * tsz * self.groups, -1)
        with torch.no_grad():
            (_, k) = x.max(-1)
            hard_x = x.new_zeros(*x.shape).scatter_(-1, k.view(-1, 1), 1.0).view(bsz * tsz, self.groups, -1)
            hard_probs = torch.mean(hard_x.float(), dim=0)
            result['code_perplexity'] = torch.exp(-torch.sum(hard_probs * torch.log(hard_probs + 1e-07), dim=-1)).sum()
        avg_probs = torch.softmax(x.view(bsz * tsz, self.groups, -1).float(), dim=-1).mean(dim=0)
        result['prob_perplexity'] = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-07), dim=-1)).sum()
        result['temp'] = self.curr_temp
        if self.training:
            x = F.gumbel_softmax(x.float(), tau=self.curr_temp, hard=self.hard).type_as(x)
        else:
            x = hard_x
        x = x.view(bsz * tsz, -1)
        vars = self.vars
        if self.combine_groups:
            vars = vars.repeat(1, self.groups, 1)
        if produce_targets:
            result['targets'] = x.view(bsz * tsz * self.groups, -1).argmax(dim=-1).view(bsz, tsz, self.groups).detach()
        x = x.unsqueeze(-1) * vars
        x = x.view(bsz * tsz, self.groups, self.num_vars, -1)
        x = x.sum(-2)
        x = x.view(bsz, tsz, -1)
        if not self.time_first:
            x = x.transpose(1, 2)
        result['x'] = x
        return result