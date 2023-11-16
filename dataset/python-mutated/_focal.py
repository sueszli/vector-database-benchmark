import torch
import torch.nn.functional as F

def sigmoid_focal_loss(outputs: torch.Tensor, targets: torch.Tensor, gamma: float=2.0, alpha: float=0.25, reduction: str='mean'):
    if False:
        while True:
            i = 10
    '\n    Compute binary focal loss between target and output logits.\n\n    Args:\n        outputs: tensor of arbitrary shape\n        targets: tensor of the same shape as input\n        gamma: gamma for focal loss\n        alpha: alpha for focal loss\n        reduction (string, optional):\n            specifies the reduction to apply to the output:\n            ``"none"`` | ``"mean"`` | ``"sum"`` | ``"batchwise_mean"``.\n            ``"none"``: no reduction will be applied,\n            ``"mean"``: the sum of the output will be divided by the number of\n            elements in the output,\n            ``"sum"``: the output will be summed.\n\n    Returns:\n        computed loss\n\n    Source: https://github.com/BloodAxe/pytorch-toolbelt\n    '
    targets = targets.type(outputs.type())
    logpt = -F.binary_cross_entropy_with_logits(outputs, targets, reduction='none')
    pt = torch.exp(logpt)
    loss = -(1 - pt).pow(gamma) * logpt
    if alpha is not None:
        loss = loss * (alpha * targets + (1 - alpha) * (1 - targets))
    if reduction == 'mean':
        loss = loss.mean()
    if reduction == 'sum':
        loss = loss.sum()
    if reduction == 'batchwise_mean':
        loss = loss.sum(0)
    return loss

def reduced_focal_loss(outputs: torch.Tensor, targets: torch.Tensor, threshold: float=0.5, gamma: float=2.0, reduction='mean') -> torch.Tensor:
    if False:
        return 10
    'Compute reduced focal loss between target and output logits.\n\n    It has been proposed in `Reduced Focal Loss\\: 1st Place Solution to xView\n    object detection in Satellite Imagery`_ paper.\n\n    .. note::\n        ``size_average`` and ``reduce`` params are in the process of being\n        deprecated, and in the meantime, specifying either of those two args\n        will override ``reduction``.\n\n    Source: https://github.com/BloodAxe/pytorch-toolbelt\n\n    .. _Reduced Focal Loss\\: 1st Place Solution to xView object detection\n        in Satellite Imagery: https://arxiv.org/abs/1903.01347\n\n    Args:\n        outputs: tensor of arbitrary shape\n        targets: tensor of the same shape as input\n        threshold: threshold for focal reduction\n        gamma: gamma for focal reduction\n        reduction: specifies the reduction to apply to the output:\n            ``"none"`` | ``"mean"`` | ``"sum"`` | ``"batchwise_mean"``.\n            ``"none"``: no reduction will be applied,\n            ``"mean"``: the sum of the output will be divided by the number of\n            elements in the output,\n            ``"sum"``: the output will be summed.\n            ``"batchwise_mean"`` computes mean loss per sample in batch.\n            Default: "mean"\n\n    Returns:  # noqa: DAR201\n        torch.Tensor: computed loss\n    '
    targets = targets.type(outputs.type())
    logpt = -F.binary_cross_entropy_with_logits(outputs, targets, reduction='none')
    pt = torch.exp(logpt)
    focal_reduction = ((1.0 - pt) / threshold).pow(gamma)
    focal_reduction[pt < threshold] = 1
    loss = -focal_reduction * logpt
    if reduction == 'mean':
        loss = loss.mean()
    if reduction == 'sum':
        loss = loss.sum()
    if reduction == 'batchwise_mean':
        loss = loss.sum(0)
    return loss
__all__ = ['sigmoid_focal_loss', 'reduced_focal_loss']