import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from modelscope.models.cv.tinynas_detection.damo.utils.boxes import bbox_overlaps

def reduce_loss(loss, reduction):
    if False:
        print('Hello World!')
    'Reduce loss as specified.\n    Args:\n        loss (Tensor): Elementwise loss tensor.\n        reduction (str): Options are "none", "mean" and "sum".\n    Return:\n        Tensor: Reduced loss tensor.\n    '
    reduction_enum = F._Reduction.get_enum(reduction)
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()

def weighted_loss(loss_func):
    if False:
        i = 10
        return i + 15
    "Create a weighted version of a given loss function.\n\n    To use this decorator, the loss function must have the signature like\n    `loss_func(pred, target, **kwargs)`. The function only needs to compute\n    element-wise loss without any reduction. This decorator will add weight\n    and reduction arguments to the function. The decorated function will have\n    the signature like `loss_func(pred, target, weight=None, reduction='mean',\n    avg_factor=None, **kwargs)`.\n\n    Example:\n\n    >>> import torch\n    >>> @weighted_loss\n    >>> def l1_loss(pred, target):\n    >>>     return (pred - target).abs()\n\n    >>> pred = torch.Tensor([0, 2, 3])\n    >>> target = torch.Tensor([1, 1, 1])\n    >>> weight = torch.Tensor([1, 0, 1])\n\n    >>> l1_loss(pred, target)\n    tensor(1.3333)\n    >>> l1_loss(pred, target, weight)\n    tensor(1.)\n    >>> l1_loss(pred, target, reduction='none')\n    tensor([1., 1., 2.])\n    >>> l1_loss(pred, target, weight, avg_factor=2)\n    tensor(1.5000)\n    "

    @functools.wraps(loss_func)
    def wrapper(pred, target, weight=None, reduction='mean', avg_factor=None, **kwargs):
        if False:
            while True:
                i = 10
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss
    return wrapper

def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    if False:
        for i in range(10):
            print('nop')
    'Apply element-wise weight and reduce loss.\n    Args:\n        loss (Tensor): Element-wise loss.\n        weight (Tensor): Element-wise weights.\n        reduction (str): Same as built-in losses of PyTorch.\n        avg_factor (float): Avarage factor when computing the mean of losses.\n    Returns:\n        Tensor: Processed loss values.\n    '
    if weight is not None:
        loss = loss * weight
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    elif reduction == 'mean':
        loss = loss.sum() / avg_factor
    elif reduction != 'none':
        raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss

@weighted_loss
def giou_loss(pred, target, eps=1e-07):
    if False:
        return 10
    '`Generalized Intersection over Union: A Metric and A Loss for Bounding\n    Box Regression <https://arxiv.org/abs/1902.09630>`_.\n    Args:\n        pred (torch.Tensor): Predicted bboxes of format (x1, y1, x2, y2),\n            shape (n, 4).\n        target (torch.Tensor): Corresponding gt bboxes, shape (n, 4).\n        eps (float): Eps to avoid log(0).\n    Return:\n        Tensor: Loss tensor.\n    '
    gious = bbox_overlaps(pred, target, mode='giou', is_aligned=True, eps=eps)
    loss = 1 - gious
    return loss

class GIoULoss(nn.Module):

    def __init__(self, eps=1e-06, reduction='mean', loss_weight=1.0):
        if False:
            i = 10
            return i + 15
        super(GIoULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        if False:
            while True:
                i = 10
        if weight is not None and (not torch.any(weight > 0)):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        if weight is not None and weight.dim() > 1:
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * giou_loss(pred, target, weight, eps=self.eps, reduction=reduction, avg_factor=avg_factor, **kwargs)
        return loss

@weighted_loss
def distribution_focal_loss(pred, label):
    if False:
        while True:
            i = 10
    'Distribution Focal Loss (DFL) is from `Generalized Focal Loss: Learning\n    Qualified and Distributed Bounding Boxes for Dense Object Detection\n    <https://arxiv.org/abs/2006.04388>`_.\n    Args:\n        pred (torch.Tensor): Predicted general distribution of bounding boxes\n            (before softmax) with shape (N, n+1), n is the max value of the\n            integral set `{0, ..., n}` in paper.\n        label (torch.Tensor): Target distance label for bounding boxes with\n            shape (N,).\n    Returns:\n        torch.Tensor: Loss tensor with shape (N,).\n    '
    dis_left = label.long()
    dis_right = dis_left + 1
    weight_left = dis_right.float() - label
    weight_right = label - dis_left.float()
    loss = F.cross_entropy(pred, dis_left, reduction='none') * weight_left + F.cross_entropy(pred, dis_right, reduction='none') * weight_right
    return loss

class DistributionFocalLoss(nn.Module):
    """Distribution Focal Loss (DFL) is a variant of `Generalized Focal Loss:
    Learning Qualified and Distributed Bounding Boxes for Dense Object
    Detection <https://arxiv.org/abs/2006.04388>`_.
    Args:
        reduction (str): Options are `'none'`, `'mean'` and `'sum'`.
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        if False:
            i = 10
            return i + 15
        super(DistributionFocalLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        if False:
            i = 10
            return i + 15
        'Forward function.\n        Args:\n            pred (torch.Tensor): Predicted general distribution of bounding\n                boxes (before softmax) with shape (N, n+1), n is the max value\n                of the integral set `{0, ..., n}` in paper.\n            target (torch.Tensor): Target distance label for bounding boxes\n                with shape (N,).\n            weight (torch.Tensor, optional): The weight of loss for each\n                prediction. Defaults to None.\n            avg_factor (int, optional): Average factor that is used to average\n                the loss. Defaults to None.\n            reduction_override (str, optional): The reduction method used to\n                override the original reduction method of the loss.\n                Defaults to None.\n        '
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        loss_cls = self.loss_weight * distribution_focal_loss(pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss_cls

@weighted_loss
def quality_focal_loss(pred, target, beta=2.0, use_sigmoid=True):
    if False:
        return 10
    'Quality Focal Loss (QFL) is from `Generalized Focal Loss: Learning\n    Qualified and Distributed Bounding Boxes for Dense Object Detection\n    <https://arxiv.org/abs/2006.04388>`_.\n    Args:\n        pred (torch.Tensor): Predicted joint representation of classification\n            and quality (IoU) estimation with shape (N, C), C is the number of\n            classes.\n        target (tuple([torch.Tensor])): Target category label with shape (N,)\n            and target quality label with shape (N,).\n        beta (float): The beta parameter for calculating the modulating factor.\n            Defaults to 2.0.\n    Returns:\n        torch.Tensor: Loss tensor with shape (N,).\n    '
    assert len(target) == 2, 'target for QFL must be a tuple of two elements,\n        including category label and quality label, respectively'
    (label, score) = target
    if use_sigmoid:
        func = F.binary_cross_entropy_with_logits
    else:
        func = F.binary_cross_entropy
    pred_sigmoid = pred.sigmoid() if use_sigmoid else pred
    scale_factor = pred_sigmoid
    zerolabel = scale_factor.new_zeros(pred.shape)
    loss = func(pred, zerolabel, reduction='none') * scale_factor.pow(beta)
    bg_class_ind = pred.size(1)
    pos = ((label >= 0) & (label < bg_class_ind)).nonzero(as_tuple=False).squeeze(1)
    pos_label = label[pos].long()
    scale_factor = score[pos] - pred_sigmoid[pos, pos_label]
    loss[pos, pos_label] = func(pred[pos, pos_label], score[pos], reduction='none') * scale_factor.abs().pow(beta)
    loss = loss.sum(dim=1, keepdim=False)
    return loss

class QualityFocalLoss(nn.Module):
    """Quality Focal Loss (QFL) is a variant of `Generalized Focal Loss:
    Learning Qualified and Distributed Bounding Boxes for Dense Object
    Detection <https://arxiv.org/abs/2006.04388>`_.
    Args:
        use_sigmoid (bool): Whether sigmoid operation is conducted in QFL.
            Defaults to True.
        beta (float): The beta parameter for calculating the modulating factor.
            Defaults to 2.0.
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self, use_sigmoid=True, beta=2.0, reduction='mean', loss_weight=1.0):
        if False:
            return 10
        super(QualityFocalLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        if False:
            i = 10
            return i + 15
        'Forward function.\n        Args:\n            pred (torch.Tensor): Predicted joint representation of\n                classification and quality (IoU) estimation with shape (N, C),\n                C is the number of classes.\n            target (tuple([torch.Tensor])): Target category label with shape\n                (N,) and target quality label with shape (N,).\n            weight (torch.Tensor, optional): The weight of loss for each\n                prediction. Defaults to None.\n            avg_factor (int, optional): Average factor that is used to average\n                the loss. Defaults to None.\n            reduction_override (str, optional): The reduction method used to\n                override the original reduction method of the loss.\n                Defaults to None.\n        '
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        loss_cls = self.loss_weight * quality_focal_loss(pred, target, weight, beta=self.beta, use_sigmoid=self.use_sigmoid, reduction=reduction, avg_factor=avg_factor)
        return loss_cls