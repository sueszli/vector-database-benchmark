from typing import List
from functools import partial
import torch
from torch import nn
from catalyst.metrics.functional import dice

class DiceLoss(nn.Module):
    """The Dice loss.
    DiceLoss = 1 - dice score
    dice score = 2 * intersection / (intersection + union)) =     = 2 * tp / (2 * tp + fp + fn)
    """

    def __init__(self, class_dim: int=1, mode: str='macro', weights: List[float]=None, eps: float=1e-07):
        if False:
            return 10
        '\n        Args:\n            class_dim: indicates class dimention (K) for\n                ``outputs`` and ``targets`` tensors (default = 1)\n            mode: class summation strategy. Must be one of [\'micro\', \'macro\',\n                \'weighted\']. If mode=\'micro\', classes are ignored, and metric\n                are calculated generally. If mode=\'macro\', metric are\n                calculated per-class and than are averaged over all classes.\n                If mode=\'weighted\', metric are calculated per-class and than\n                summed over all classes with weights.\n            weights: class weights(for mode="weighted")\n            eps: epsilon to avoid zero division\n        '
        super().__init__()
        assert mode in ['micro', 'macro', 'weighted']
        self.loss_fn = partial(dice, eps=eps, class_dim=class_dim, threshold=None, mode=mode, weights=weights)

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if False:
            return 10
        'Calculates loss between ``logits`` and ``target`` tensors.'
        dice_score = self.loss_fn(outputs, targets)
        return 1 - dice_score
__all__ = ['DiceLoss']