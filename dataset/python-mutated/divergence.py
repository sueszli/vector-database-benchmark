"""Losses based on the divergence between probability distributions."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from kornia.core import Tensor

def _kl_div_2d(p: Tensor, q: Tensor) -> Tensor:
    if False:
        while True:
            i = 10
    (batch, chans, height, width) = p.shape
    unsummed_kl = F.kl_div(q.reshape(batch * chans, height * width).log(), p.reshape(batch * chans, height * width), reduction='none')
    kl_values = unsummed_kl.sum(-1).view(batch, chans)
    return kl_values

def _js_div_2d(p: Tensor, q: Tensor) -> Tensor:
    if False:
        while True:
            i = 10
    m = 0.5 * (p + q)
    return 0.5 * _kl_div_2d(p, m) + 0.5 * _kl_div_2d(q, m)

def _reduce_loss(losses: Tensor, reduction: str) -> Tensor:
    if False:
        while True:
            i = 10
    if reduction == 'none':
        return losses
    return torch.mean(losses) if reduction == 'mean' else torch.sum(losses)

def js_div_loss_2d(pred: Tensor, target: Tensor, reduction: str='mean') -> Tensor:
    if False:
        for i in range(10):
            print('nop')
    "Calculate the Jensen-Shannon divergence loss between heatmaps.\n\n    Args:\n        pred: the input tensor with shape :math:`(B, N, H, W)`.\n        target: the target tensor with shape :math:`(B, N, H, W)`.\n        reduction: Specifies the reduction to apply to the\n          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction\n          will be applied, ``'mean'``: the sum of the output will be divided by\n          the number of elements in the output, ``'sum'``: the output will be\n          summed.\n\n    Examples:\n        >>> pred = torch.full((1, 1, 2, 4), 0.125)\n        >>> loss = js_div_loss_2d(pred, pred)\n        >>> loss.item()\n        0.0\n    "
    return _reduce_loss(_js_div_2d(target, pred), reduction)

def kl_div_loss_2d(pred: Tensor, target: Tensor, reduction: str='mean') -> Tensor:
    if False:
        print('Hello World!')
    "Calculate the Kullback-Leibler divergence loss between heatmaps.\n\n    Args:\n        pred: the input tensor with shape :math:`(B, N, H, W)`.\n        target: the target tensor with shape :math:`(B, N, H, W)`.\n        reduction: Specifies the reduction to apply to the\n          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction\n          will be applied, ``'mean'``: the sum of the output will be divided by\n          the number of elements in the output, ``'sum'``: the output will be\n          summed.\n\n    Examples:\n        >>> pred = torch.full((1, 1, 2, 4), 0.125)\n        >>> loss = kl_div_loss_2d(pred, pred)\n        >>> loss.item()\n        0.0\n    "
    return _reduce_loss(_kl_div_2d(target, pred), reduction)