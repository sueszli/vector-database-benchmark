"""
Implementation of ``POPART`` algorithm for reward rescale.
<link https://arxiv.org/abs/1602.07714 link>

POPART is an adaptive normalization algorithm to normalized the targets used in the learning updates.
The two main components in POPART are:
**ART**: to update scale and shift such that the return is appropriately normalized
**POP**: to preserve the outputs of the unnormalized function when we change the scale and shift.

"""
from typing import Optional, Union
import math
import torch
import torch.nn as nn

class PopArt(nn.Module):
    """
    **Overview**:
        A linear layer with popart normalization.
        For more popart implementation info, you can refer to the paper <link https://arxiv.org/abs/1809.04474 link>
    """

    def __init__(self, input_features: Union[int, None]=None, output_features: Union[int, None]=None, beta: float=0.5) -> None:
        if False:
            while True:
                i = 10
        super(PopArt, self).__init__()
        self.beta = beta
        self.input_features = input_features
        self.output_features = output_features
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        self.bias = nn.Parameter(torch.Tensor(output_features))
        self.register_buffer('mu', torch.zeros(output_features, requires_grad=False))
        self.register_buffer('sigma', torch.ones(output_features, requires_grad=False))
        self.register_buffer('v', torch.ones(output_features, requires_grad=False))
        self.reset_parameters()

    def reset_parameters(self):
        if False:
            return 10
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            (fan_in, _) = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if False:
            return 10
        '\n        **Overview**:\n            The computation of the linear layer, which outputs both the output and the normalized output of the layer.\n        '
        normalized_output = x.mm(self.weight.t())
        normalized_output += self.bias.unsqueeze(0).expand_as(normalized_output)
        with torch.no_grad():
            output = normalized_output * self.sigma + self.mu
        return {'pred': normalized_output.squeeze(1), 'unnormalized_pred': output.squeeze(1)}

    def update_parameters(self, value):
        if False:
            print('Hello World!')
        '\n        **Overview**:\n            The parameters update, which outputs both the output and the normalized output of the layer.\n        '
        self.mu = self.mu.to(value.device)
        self.sigma = self.sigma.to(value.device)
        self.v = self.v.to(value.device)
        old_mu = self.mu
        old_std = self.sigma
        batch_mean = torch.mean(value, 0)
        batch_v = torch.mean(torch.pow(value, 2), 0)
        batch_mean[torch.isnan(batch_mean)] = self.mu[torch.isnan(batch_mean)]
        batch_v[torch.isnan(batch_v)] = self.v[torch.isnan(batch_v)]
        batch_mean = (1 - self.beta) * self.mu + self.beta * batch_mean
        batch_v = (1 - self.beta) * self.v + self.beta * batch_v
        batch_std = torch.sqrt(batch_v - batch_mean ** 2)
        batch_std = torch.clamp(batch_std, min=0.0001, max=1000000.0)
        batch_std[torch.isnan(batch_std)] = self.sigma[torch.isnan(batch_std)]
        self.mu = batch_mean
        self.v = batch_v
        self.sigma = batch_std
        self.weight.data = (self.weight.data.t() * old_std / self.sigma).t()
        self.bias.data = (old_std * self.bias.data + old_mu - self.mu) / self.sigma
        return {'new_mean': batch_mean, 'new_std': batch_std}