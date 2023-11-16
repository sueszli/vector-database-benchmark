import torch
from torch import nn as nn
from mmdet.models.losses.utils import weighted_loss
from ..builder import LOSSES

@weighted_loss
def uncertain_smooth_l1_loss(pred, target, sigma, alpha=1.0, beta=1.0):
    if False:
        for i in range(10):
            print('nop')
    'Smooth L1 loss with uncertainty.\n\n    Args:\n        pred (torch.Tensor): The prediction.\n        target (torch.Tensor): The learning target of the prediction.\n        sigma (torch.Tensor): The sigma for uncertainty.\n        alpha (float, optional): The coefficient of log(sigma).\n            Defaults to 1.0.\n        beta (float, optional): The threshold in the piecewise function.\n            Defaults to 1.0.\n\n    Returns:\n        torch.Tensor: Calculated loss\n    '
    assert beta > 0
    assert target.numel() > 0
    assert pred.size() == target.size() == sigma.size(), f'The size of pred {pred.size()}, target {target.size()}, and sigma {sigma.size()} are inconsistent.'
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)
    loss = torch.exp(-sigma) * loss + alpha * sigma
    return loss

@weighted_loss
def uncertain_l1_loss(pred, target, sigma, alpha=1.0):
    if False:
        return 10
    'L1 loss with uncertainty.\n\n    Args:\n        pred (torch.Tensor): The prediction.\n        target (torch.Tensor): The learning target of the prediction.\n        sigma (torch.Tensor): The sigma for uncertainty.\n        alpha (float, optional): The coefficient of log(sigma).\n            Defaults to 1.0.\n\n    Returns:\n        torch.Tensor: Calculated loss\n    '
    assert target.numel() > 0
    assert pred.size() == target.size() == sigma.size(), f'The size of pred {pred.size()}, target {target.size()}, and sigma {sigma.size()} are inconsistent.'
    loss = torch.abs(pred - target)
    loss = torch.exp(-sigma) * loss + alpha * sigma
    return loss

@LOSSES.register_module()
class UncertainSmoothL1Loss(nn.Module):
    """Smooth L1 loss with uncertainty.

    Please refer to `PGD <https://arxiv.org/abs/2107.14160>`_ and
    `Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry
    and Semantics <https://arxiv.org/abs/1705.07115>`_ for more details.

    Args:
        alpha (float, optional): The coefficient of log(sigma).
            Defaults to 1.0.
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.
        reduction (str, optional): The method to reduce the loss.
            Options are 'none', 'mean' and 'sum'. Defaults to 'mean'.
        loss_weight (float, optional): The weight of loss. Defaults to 1.0
    """

    def __init__(self, alpha=1.0, beta=1.0, reduction='mean', loss_weight=1.0):
        if False:
            print('Hello World!')
        super(UncertainSmoothL1Loss, self).__init__()
        assert reduction in ['none', 'sum', 'mean']
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, sigma, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        if False:
            print('Hello World!')
        'Forward function.\n\n        Args:\n            pred (torch.Tensor): The prediction.\n            target (torch.Tensor): The learning target of the prediction.\n            sigma (torch.Tensor): The sigma for uncertainty.\n            weight (torch.Tensor, optional): The weight of loss for each\n                prediction. Defaults to None.\n            avg_factor (int, optional): Average factor that is used to average\n                the loss. Defaults to None.\n            reduction_override (str, optional): The reduction method used to\n                override the original reduction method of the loss.\n                Defaults to None.\n        '
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        loss_bbox = self.loss_weight * uncertain_smooth_l1_loss(pred, target, weight, sigma=sigma, alpha=self.alpha, beta=self.beta, reduction=reduction, avg_factor=avg_factor, **kwargs)
        return loss_bbox

@LOSSES.register_module()
class UncertainL1Loss(nn.Module):
    """L1 loss with uncertainty.

    Args:
        alpha (float, optional): The coefficient of log(sigma).
            Defaults to 1.0.
        reduction (str, optional): The method to reduce the loss.
            Options are 'none', 'mean' and 'sum'. Defaults to 'mean'.
        loss_weight (float, optional): The weight of loss. Defaults to 1.0.
    """

    def __init__(self, alpha=1.0, reduction='mean', loss_weight=1.0):
        if False:
            return 10
        super(UncertainL1Loss, self).__init__()
        assert reduction in ['none', 'sum', 'mean']
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, sigma, weight=None, avg_factor=None, reduction_override=None):
        if False:
            i = 10
            return i + 15
        'Forward function.\n\n        Args:\n            pred (torch.Tensor): The prediction.\n            target (torch.Tensor): The learning target of the prediction.\n            sigma (torch.Tensor): The sigma for uncertainty.\n            weight (torch.Tensor, optional): The weight of loss for each\n                prediction. Defaults to None.\n            avg_factor (int, optional): Average factor that is used to average\n                the loss. Defaults to None.\n            reduction_override (str, optional): The reduction method used to\n                override the original reduction method of the loss.\n                Defaults to None.\n        '
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        loss_bbox = self.loss_weight * uncertain_l1_loss(pred, target, weight, sigma=sigma, alpha=self.alpha, reduction=reduction, avg_factor=avg_factor)
        return loss_bbox