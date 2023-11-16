import torch
from torch import nn as nn
from torch.nn.functional import l1_loss, mse_loss, smooth_l1_loss
from ..builder import LOSSES

def chamfer_distance(src, dst, src_weight=1.0, dst_weight=1.0, criterion_mode='l2', reduction='mean'):
    if False:
        print('Hello World!')
    "Calculate Chamfer Distance of two sets.\n\n    Args:\n        src (torch.Tensor): Source set with shape [B, N, C] to\n            calculate Chamfer Distance.\n        dst (torch.Tensor): Destination set with shape [B, M, C] to\n            calculate Chamfer Distance.\n        src_weight (torch.Tensor or float): Weight of source loss.\n        dst_weight (torch.Tensor or float): Weight of destination loss.\n        criterion_mode (str): Criterion mode to calculate distance.\n            The valid modes are smooth_l1, l1 or l2.\n        reduction (str): Method to reduce losses.\n            The valid reduction method are 'none', 'sum' or 'mean'.\n\n    Returns:\n        tuple: Source and Destination loss with the corresponding indices.\n\n            - loss_src (torch.Tensor): The min distance\n                from source to destination.\n            - loss_dst (torch.Tensor): The min distance\n                from destination to source.\n            - indices1 (torch.Tensor): Index the min distance point\n                for each point in source to destination.\n            - indices2 (torch.Tensor): Index the min distance point\n                for each point in destination to source.\n    "
    if criterion_mode == 'smooth_l1':
        criterion = smooth_l1_loss
    elif criterion_mode == 'l1':
        criterion = l1_loss
    elif criterion_mode == 'l2':
        criterion = mse_loss
    else:
        raise NotImplementedError
    src_expand = src.unsqueeze(2).repeat(1, 1, dst.shape[1], 1)
    dst_expand = dst.unsqueeze(1).repeat(1, src.shape[1], 1, 1)
    distance = criterion(src_expand, dst_expand, reduction='none').sum(-1)
    (src2dst_distance, indices1) = torch.min(distance, dim=2)
    (dst2src_distance, indices2) = torch.min(distance, dim=1)
    loss_src = src2dst_distance * src_weight
    loss_dst = dst2src_distance * dst_weight
    if reduction == 'sum':
        loss_src = torch.sum(loss_src)
        loss_dst = torch.sum(loss_dst)
    elif reduction == 'mean':
        loss_src = torch.mean(loss_src)
        loss_dst = torch.mean(loss_dst)
    elif reduction == 'none':
        pass
    else:
        raise NotImplementedError
    return (loss_src, loss_dst, indices1, indices2)

@LOSSES.register_module()
class ChamferDistance(nn.Module):
    """Calculate Chamfer Distance of two sets.

    Args:
        mode (str): Criterion mode to calculate distance.
            The valid modes are smooth_l1, l1 or l2.
        reduction (str): Method to reduce losses.
            The valid reduction method are none, sum or mean.
        loss_src_weight (float): Weight of loss_source.
        loss_dst_weight (float): Weight of loss_target.
    """

    def __init__(self, mode='l2', reduction='mean', loss_src_weight=1.0, loss_dst_weight=1.0):
        if False:
            while True:
                i = 10
        super(ChamferDistance, self).__init__()
        assert mode in ['smooth_l1', 'l1', 'l2']
        assert reduction in ['none', 'sum', 'mean']
        self.mode = mode
        self.reduction = reduction
        self.loss_src_weight = loss_src_weight
        self.loss_dst_weight = loss_dst_weight

    def forward(self, source, target, src_weight=1.0, dst_weight=1.0, reduction_override=None, return_indices=False, **kwargs):
        if False:
            print('Hello World!')
        "Forward function of loss calculation.\n\n        Args:\n            source (torch.Tensor): Source set with shape [B, N, C] to\n                calculate Chamfer Distance.\n            target (torch.Tensor): Destination set with shape [B, M, C] to\n                calculate Chamfer Distance.\n            src_weight (torch.Tensor | float, optional):\n                Weight of source loss. Defaults to 1.0.\n            dst_weight (torch.Tensor | float, optional):\n                Weight of destination loss. Defaults to 1.0.\n            reduction_override (str, optional): Method to reduce losses.\n                The valid reduction method are 'none', 'sum' or 'mean'.\n                Defaults to None.\n            return_indices (bool, optional): Whether to return indices.\n                Defaults to False.\n\n        Returns:\n            tuple[torch.Tensor]: If ``return_indices=True``, return losses of\n                source and target with their corresponding indices in the\n                order of ``(loss_source, loss_target, indices1, indices2)``.\n                If ``return_indices=False``, return\n                ``(loss_source, loss_target)``.\n        "
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        (loss_source, loss_target, indices1, indices2) = chamfer_distance(source, target, src_weight, dst_weight, self.mode, reduction)
        loss_source *= self.loss_src_weight
        loss_target *= self.loss_dst_weight
        if return_indices:
            return (loss_source, loss_target, indices1, indices2)
        else:
            return (loss_source, loss_target)