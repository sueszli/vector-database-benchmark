import torch
import torch.nn as nn
from bigdl.nano.utils.common import invalidInputError

class LinexLoss(nn.Module):
    """
    LinexLoss is an asymmetric loss talked about in
    https://www.scirp.org/journal/paperinformation.aspx?paperid=97986

    """

    def __init__(self, a=1):
        if False:
            return 10
        '\n        :param a: when a is set to ~0, this loss is similar to MSE.\n               when a is set to > 0, this loss panelize underestimate more.\n               when a is set to < 0, this loss panelize overestimate more.\n        '
        super().__init__()
        invalidInputError(a != 0, 'a should not be set to 0')
        self.a = a
        self.b = 2 / a ** 2

    def forward(self, y_hat, y):
        if False:
            return 10
        delta = y - y_hat
        loss = self.b * (torch.exp(self.a * delta) - self.a * delta - 1)
        return torch.mean(loss)