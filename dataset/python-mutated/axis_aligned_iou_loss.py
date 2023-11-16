import torch
from torch import nn as nn
from mmdet.models.losses.utils import weighted_loss
from ...core.bbox import AxisAlignedBboxOverlaps3D
from ..builder import LOSSES

@weighted_loss
def axis_aligned_iou_loss(pred, target):
    if False:
        return 10
    'Calculate the IoU loss (1-IoU) of two sets of axis aligned bounding\n    boxes. Note that predictions and targets are one-to-one corresponded.\n\n    Args:\n        pred (torch.Tensor): Bbox predictions with shape [..., 6]\n            (x1, y1, z1, x2, y2, z2).\n        target (torch.Tensor): Bbox targets (gt) with shape [..., 6]\n            (x1, y1, z1, x2, y2, z2).\n\n    Returns:\n        torch.Tensor: IoU loss between predictions and targets.\n    '
    axis_aligned_iou = AxisAlignedBboxOverlaps3D()(pred, target, is_aligned=True)
    iou_loss = 1 - axis_aligned_iou
    return iou_loss

@LOSSES.register_module()
class AxisAlignedIoULoss(nn.Module):
    """Calculate the IoU loss (1-IoU) of axis aligned bounding boxes.

    Args:
        reduction (str): Method to reduce losses.
            The valid reduction method are none, sum or mean.
        loss_weight (float, optional): Weight of loss. Defaults to 1.0.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        if False:
            while True:
                i = 10
        super(AxisAlignedIoULoss, self).__init__()
        assert reduction in ['none', 'sum', 'mean']
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        if False:
            while True:
                i = 10
        "Forward function of loss calculation.\n\n        Args:\n            pred (torch.Tensor): Bbox predictions with shape [..., 6]\n                (x1, y1, z1, x2, y2, z2).\n            target (torch.Tensor): Bbox targets (gt) with shape [..., 6]\n                (x1, y1, z1, x2, y2, z2).\n            weight (torch.Tensor | float, optional): Weight of loss.\n                Defaults to None.\n            avg_factor (int, optional): Average factor that is used to average\n                the loss. Defaults to None.\n            reduction_override (str, optional): Method to reduce losses.\n                The valid reduction method are 'none', 'sum' or 'mean'.\n                Defaults to None.\n\n        Returns:\n            torch.Tensor: IoU loss between predictions and targets.\n        "
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        if weight is not None and (not torch.any(weight > 0)) and (reduction != 'none'):
            return (pred * weight).sum()
        return axis_aligned_iou_loss(pred, target, weight=weight, avg_factor=avg_factor, reduction=reduction) * self.loss_weight