import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Optional

class LabelSmoothCELoss(nn.Module):
    """
    Overview:
        Label smooth cross entropy loss.
    Interfaces:
        forward
    """

    def __init__(self, ratio: float) -> None:
        if False:
            print('Hello World!')
        super().__init__()
        self.ratio = ratio

    def forward(self, logits: torch.Tensor, labels: torch.LongTensor) -> torch.Tensor:
        if False:
            i = 10
            return i + 15
        '\n        Overview:\n            Calculate label smooth cross entropy loss.\n        Arguments:\n            - logits (:obj:`torch.Tensor`): Predicted logits.\n            - labels (:obj:`torch.LongTensor`): Ground truth.\n        Returns:\n            - loss (:obj:`torch.Tensor`): Calculated loss.\n        '
        (B, N) = logits.shape
        val = float(self.ratio) / (N - 1)
        one_hot = torch.full_like(logits, val)
        one_hot.scatter_(1, labels.unsqueeze(1), 1 - val)
        logits = F.log_softmax(logits, dim=1)
        return -torch.sum(logits * one_hot.detach()) / B

class SoftFocalLoss(nn.Module):
    """
    Overview:
        Soft focal loss.
    Interfaces:
        forward
    """

    def __init__(self, gamma: int=2, weight: Any=None, size_average: bool=True, reduce: Optional[bool]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.gamma = gamma
        self.nll_loss = torch.nn.NLLLoss2d(weight, size_average, reduce=reduce)

    def forward(self, inputs: torch.Tensor, targets: torch.LongTensor) -> torch.Tensor:
        if False:
            return 10
        '\n        Overview:\n            Calculate soft focal loss.\n        Arguments:\n            - logits (:obj:`torch.Tensor`): Predicted logits.\n            - labels (:obj:`torch.LongTensor`): Ground truth.\n        Returns:\n            - loss (:obj:`torch.Tensor`): Calculated loss.\n        '
        return self.nll_loss((1 - F.softmax(inputs, 1)) ** self.gamma * F.log_softmax(inputs, 1), targets)

def build_ce_criterion(cfg: dict) -> nn.Module:
    if False:
        i = 10
        return i + 15
    '\n    Overview:\n        Get a cross enntropy loss instance according to given config.\n    Arguments:\n        - cfg (:obj:`dict`)\n    Returns:\n        - loss (:obj:`nn.Module`): loss function instance\n    '
    if cfg.type == 'cross_entropy':
        return nn.CrossEntropyLoss()
    elif cfg.type == 'label_smooth_ce':
        return LabelSmoothCELoss(cfg.kwargs.smooth_ratio)
    elif cfg.type == 'soft_focal_loss':
        return SoftFocalLoss()
    else:
        raise ValueError('invalid criterion type:{}'.format(cfg.type))