import torch
from torch import nn as nn
from mmdet3d.ops import PAConv, PAConvCUDA
from mmdet.models.losses.utils import weight_reduce_loss
from ..builder import LOSSES

def weight_correlation(conv):
    if False:
        while True:
            i = 10
    "Calculate correlations between kernel weights in Conv's weight bank as\n    regularization loss. The cosine similarity is used as metrics.\n\n    Args:\n        conv (nn.Module): A Conv modules to be regularized.\n            Currently we only support `PAConv` and `PAConvCUDA`.\n\n    Returns:\n        torch.Tensor: Correlations between each kernel weights in weight bank.\n    "
    assert isinstance(conv, (PAConv, PAConvCUDA)), f'unsupported module type {type(conv)}'
    kernels = conv.weight_bank
    in_channels = conv.in_channels
    out_channels = conv.out_channels
    num_kernels = conv.num_kernels
    flatten_kernels = kernels.view(in_channels, num_kernels, out_channels).permute(1, 0, 2).reshape(num_kernels, -1)
    inner_product = torch.matmul(flatten_kernels, flatten_kernels.T)
    kernel_norms = torch.sum(flatten_kernels ** 2, dim=-1, keepdim=True) ** 0.5
    kernel_norms = torch.matmul(kernel_norms, kernel_norms.T)
    cosine_sims = inner_product / kernel_norms
    corr = torch.sum(torch.triu(cosine_sims, diagonal=1) ** 2)
    return corr

def paconv_regularization_loss(modules, reduction):
    if False:
        for i in range(10):
            print('nop')
    'Computes correlation loss of PAConv weight kernels as regularization.\n\n    Args:\n        modules (List[nn.Module] | :obj:`generator`):\n            A list or a python generator of torch.nn.Modules.\n        reduction (str): Method to reduce losses among PAConv modules.\n            The valid reduction method are none, sum or mean.\n\n    Returns:\n        torch.Tensor: Correlation loss of kernel weights.\n    '
    corr_loss = []
    for module in modules:
        if isinstance(module, (PAConv, PAConvCUDA)):
            corr_loss.append(weight_correlation(module))
    corr_loss = torch.stack(corr_loss)
    corr_loss = weight_reduce_loss(corr_loss, reduction=reduction)
    return corr_loss

@LOSSES.register_module()
class PAConvRegularizationLoss(nn.Module):
    """Calculate correlation loss of kernel weights in PAConv's weight bank.

    This is used as a regularization term in PAConv model training.

    Args:
        reduction (str): Method to reduce losses. The reduction is performed
            among all PAConv modules instead of prediction tensors.
            The valid reduction method are none, sum or mean.
        loss_weight (float, optional): Weight of loss. Defaults to 1.0.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        if False:
            for i in range(10):
                print('nop')
        super(PAConvRegularizationLoss, self).__init__()
        assert reduction in ['none', 'sum', 'mean']
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, modules, reduction_override=None, **kwargs):
        if False:
            return 10
        "Forward function of loss calculation.\n\n        Args:\n            modules (List[nn.Module] | :obj:`generator`):\n                A list or a python generator of torch.nn.Modules.\n            reduction_override (str, optional): Method to reduce losses.\n                The valid reduction method are 'none', 'sum' or 'mean'.\n                Defaults to None.\n\n        Returns:\n            torch.Tensor: Correlation loss of kernel weights.\n        "
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        return self.loss_weight * paconv_regularization_loss(modules, reduction=reduction)