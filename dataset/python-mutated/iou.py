from typing import List
from functools import partial
import torch
from torch import nn
from catalyst.metrics.functional import iou

class IoULoss(nn.Module):
    """The intersection over union (Jaccard) loss.
    IOULoss = 1 - iou score
    iou score = intersection / union = tp / (tp + fp + fn)
    """

    def __init__(self, class_dim: int=1, mode: str='macro', weights: List[float]=None, eps: float=1e-07):
        if False:
            i = 10
            return i + 15
        '\n        Args:\n            class_dim: indicates class dimention (K) for\n                ``outputs`` and ``targets`` tensors (default = 1)\n            mode: class summation strategy. Must be one of [\'micro\', \'macro\',\n                \'weighted\']. If mode=\'micro\', classes are ignored, and metric\n                are calculated generally. If mode=\'macro\', metric are\n                calculated per-class and than are averaged over all classes.\n                If mode=\'weighted\', metric are calculated per-class and than\n                summed over all classes with weights.\n            weights: class weights(for mode="weighted")\n            eps: epsilon to avoid zero division\n        '
        super().__init__()
        assert mode in ['micro', 'macro', 'weighted']
        self.loss_fn = partial(iou, eps=eps, class_dim=class_dim, threshold=None, mode=mode, weights=weights)

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if False:
            i = 10
            return i + 15
        'Calculates loss between ``logits`` and ``target`` tensors.'
        iou_score = self.loss_fn(outputs, targets)
        return 1 - iou_score
__all__ = ['IoULoss']