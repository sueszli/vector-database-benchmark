"""
Layer norm done in fp32 (for fp16 training)
"""
import torch.nn as nn
import torch.nn.functional as F

class Fp32GroupNorm(nn.GroupNorm):

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super().__init__(*args, **kwargs)

    def forward(self, input):
        if False:
            while True:
                i = 10
        output = F.group_norm(input.float(), self.num_groups, self.weight.float() if self.weight is not None else None, self.bias.float() if self.bias is not None else None, self.eps)
        return output.type_as(input)