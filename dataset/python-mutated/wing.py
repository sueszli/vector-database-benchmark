from functools import partial
import math
import torch
from torch import nn

def wing_loss(outputs: torch.Tensor, targets: torch.Tensor, width: int=5, curvature: float=0.5, reduction: str='mean') -> torch.Tensor:
    if False:
        while True:
            i = 10
    'The Wing loss.\n\n    It has been proposed in `Wing Loss for Robust Facial Landmark Localisation\n    with Convolutional Neural Networks`_.\n\n    Args:\n        @TODO: Docs. Contribution is welcome.\n\n    Adapted from:\n    https://github.com/BloodAxe/pytorch-toolbelt (MIT License)\n\n    .. _Wing Loss for Robust Facial Landmark Localisation with Convolutional\n        Neural Networks: https://arxiv.org/abs/1711.06753\n    '
    diff_abs = (targets - outputs).abs()
    loss = diff_abs.clone()
    idx_smaller = diff_abs < width
    idx_bigger = diff_abs >= width
    loss[idx_smaller] = width * torch.log(1 + diff_abs[idx_smaller] / curvature)
    c = width - width * math.log(1 + width / curvature)
    loss[idx_bigger] = loss[idx_bigger] - c
    if reduction == 'sum':
        loss = loss.sum()
    if reduction == 'mean':
        loss = loss.mean()
    return loss

class WingLoss(nn.Module):
    """Creates a criterion that optimizes a Wing loss.

    It has been proposed in `Wing Loss for Robust Facial Landmark Localisation
    with Convolutional Neural Networks`_.

    Adapted from:
    https://github.com/BloodAxe/pytorch-toolbelt

    .. _Wing Loss for Robust Facial Landmark Localisation with Convolutional
        Neural Networks: https://arxiv.org/abs/1711.06753
    """

    def __init__(self, width: int=5, curvature: float=0.5, reduction: str='mean'):
        if False:
            for i in range(10):
                print('nop')
        '\n        Args:\n            @TODO: Docs. Contribution is welcome.\n        '
        super().__init__()
        self.loss_fn = partial(wing_loss, width=width, curvature=curvature, reduction=reduction)

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if False:
            i = 10
            return i + 15
        '\n        Args:\n            @TODO: Docs. Contribution is welcome.\n        '
        loss = self.loss_fn(outputs, targets)
        return loss
__all__ = ['WingLoss']