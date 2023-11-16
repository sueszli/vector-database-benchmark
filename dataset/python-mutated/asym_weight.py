import torch
import torch.nn as nn
import torch.nn.functional as F
from bigdl.nano.utils.common import invalidInputError

class AsymWeightLoss(nn.Module):
    """
    AsymWeightLoss is an asymmetric loss.
    """

    def __init__(self, underestimation_penalty=1, L1=False):
        if False:
            i = 10
            return i + 15
        '\n        :param underestimation_penalty: when underestimation_penalty is set to 1, the loss is MSE,\n               if set larger than 1, this loss panelize underestimate more and vice versa.\n        :param L1: if use L1 loss rather than MSE(L2) loss.\n        '
        super().__init__()
        invalidInputError(underestimation_penalty > 0, 'underestimation_penalty should be larger than 0')
        self.L1 = L1
        self.underestimation_penalty = underestimation_penalty

    def forward(self, y_hat, y):
        if False:
            i = 10
            return i + 15
        if self.L1:
            loss = F.relu(y_hat - y) + F.relu(y - y_hat) * self.underestimation_penalty
        else:
            loss = torch.pow(F.relu(y_hat - y), 2) + torch.pow(F.relu(y - y_hat), 2) * self.underestimation_penalty
        return torch.mean(loss)