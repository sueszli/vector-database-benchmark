from __future__ import annotations
from typing import Callable
import torch
from torch import nn
from kornia.core import Module, Tensor, as_tensor, stack, tensor, where, zeros_like

class _HausdorffERLossBase(Module):
    """Base class for binary Hausdorff loss based on morphological erosion.

    This is an Hausdorff Distance (HD) Loss that based on morphological erosion,which provided
    a differentiable approximation of Hausdorff distance as stated in :cite:`karimi2019reducing`.
    The code is refactored on top of `here <https://github.com/PatRyg99/HausdorffLoss/
        blob/master/hausdorff_loss.py>`__.

    Args:
        alpha: controls the erosion rate in each iteration.
        k: the number of iterations of erosion.
        reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
            'none': no reduction will be applied, 'mean': the weighted mean of the output is taken,
            'sum': the output will be summed.

    Returns:
        Estimated Hausdorff Loss.
    """
    conv: Callable[..., Tensor]
    max_pool: Callable[..., Tensor]

    def __init__(self, alpha: float=2.0, k: int=10, reduction: str='mean') -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.alpha = alpha
        self.k = k
        self.reduction = reduction
        self.register_buffer('kernel', self.get_kernel())

    def get_kernel(self) -> Tensor:
        if False:
            print('Hello World!')
        'Get kernel for image morphology convolution.'
        raise NotImplementedError

    def perform_erosion(self, pred: Tensor, target: Tensor) -> Tensor:
        if False:
            i = 10
            return i + 15
        bound = (pred - target) ** 2
        kernel = as_tensor(self.kernel, device=pred.device, dtype=pred.dtype)
        eroded = zeros_like(bound, device=pred.device, dtype=pred.dtype)
        mask = torch.ones_like(bound, device=pred.device, dtype=torch.bool)
        padding = (kernel.size(-1) - 1) // 2
        for k in range(self.k):
            dilation = self.conv(bound, weight=kernel, padding=padding, groups=1)
            erosion = dilation - 0.5
            erosion[erosion < 0] = 0
            erosion_max = self.max_pool(erosion)
            erosion_min = -self.max_pool(-erosion)
            _to_norm = erosion_max - erosion_min != 0
            to_norm = _to_norm.squeeze()
            if to_norm.any():
                _erosion_to_fill = (erosion - erosion_min) / (erosion_max - erosion_min)
                erosion = where(mask * _to_norm, _erosion_to_fill, erosion)
            eroded = eroded + erosion * (k + 1) ** self.alpha
            bound = erosion
        return eroded

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        if False:
            return 10
        'Compute Hausdorff loss.\n\n        Args:\n            pred: predicted tensor with a shape of :math:`(B, C, H, W)` or :math:`(B, C, D, H, W)`.\n                Each channel is as binary as: 1 -> fg, 0 -> bg.\n            target: target tensor with a shape of :math:`(B, 1, H, W)` or :math:`(B, C, D, H, W)`.\n\n        Returns:\n            Estimated Hausdorff Loss.\n        '
        if not (pred.shape[2:] == target.shape[2:] and pred.size(0) == target.size(0) and (target.size(1) == 1)):
            raise ValueError(f'Prediction and target need to be of same size, and target should not be one-hot.Got {pred.shape} and {target.shape}.')
        if pred.size(1) < target.max().item():
            raise ValueError('Invalid target value.')
        out = stack([self.perform_erosion(pred[:, i:i + 1], where(target == i, tensor(1, device=target.device, dtype=target.dtype), tensor(0, device=target.device, dtype=target.dtype))) for i in range(pred.size(1))])
        if self.reduction == 'mean':
            out = out.mean()
        elif self.reduction == 'sum':
            out = out.sum()
        elif self.reduction == 'none':
            pass
        else:
            raise NotImplementedError(f'reduction `{self.reduction}` has not been implemented yet.')
        return out

class HausdorffERLoss(_HausdorffERLossBase):
    """Binary Hausdorff loss based on morphological erosion.

    Hausdorff Distance loss measures the maximum distance of a predicted segmentation boundary to
    the nearest ground-truth edge pixel. For two segmentation point sets X and Y ,
    the one-sided HD from X to Y is defined as:

    .. math::

        hd(X,Y) = \\max_{x \\in X} \\min_{y \\in Y}||x - y||_2

    Furthermore, the bidirectional HD is:

    .. math::

        HD(X,Y) = max(hd(X, Y), hd(Y, X))

    This is an Hausdorff Distance (HD) Loss that based on morphological erosion, which provided
    a differentiable approximation of Hausdorff distance as stated in :cite:`karimi2019reducing`.
    The code is refactored on top of `here <https://github.com/PatRyg99/HausdorffLoss/
    blob/master/hausdorff_loss.py>`__.

    Args:
        alpha: controls the erosion rate in each iteration.
        k: the number of iterations of erosion.
        reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
            'none': no reduction will be applied, 'mean': the weighted mean of the output is taken,
            'sum': the output will be summed.

    Examples:
        >>> hdloss = HausdorffERLoss()
        >>> input = torch.randn(5, 3, 20, 20)
        >>> target = (torch.rand(5, 1, 20, 20) * 2).long()
        >>> res = hdloss(input, target)
    """
    conv = torch.conv2d
    max_pool = nn.AdaptiveMaxPool2d(1)

    def get_kernel(self) -> Tensor:
        if False:
            print('Hello World!')
        'Get kernel for image morphology convolution.'
        cross = tensor([[[0, 1, 0], [1, 1, 1], [0, 1, 0]]])
        kernel = cross * 0.2
        return kernel[None]

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        if False:
            return 10
        'Compute Hausdorff loss.\n\n        Args:\n            pred: predicted tensor with a shape of :math:`(B, C, H, W)`.\n                Each channel is as binary as: 1 -> fg, 0 -> bg.\n            target: target tensor with a shape of :math:`(B, 1, H, W)`.\n\n        Returns:\n            Estimated Hausdorff Loss.\n        '
        if pred.dim() != 4:
            raise ValueError(f'Only 2D images supported. Got {pred.dim()}.')
        if not (target.max() < pred.size(1) and target.min() >= 0 and (target.dtype == torch.long)):
            raise ValueError(f'Expect long type target value in range (0, {pred.size(1)}). ({target.min()}, {target.max()})')
        return super().forward(pred, target)

class HausdorffERLoss3D(_HausdorffERLossBase):
    """Binary 3D Hausdorff loss based on morphological erosion.

    Hausdorff Distance loss measures the maximum distance of a predicted segmentation boundary to
    the nearest ground-truth edge pixel. For two segmentation point sets X and Y ,
    the one-sided HD from X to Y is defined as:

    .. math::

        hd(X,Y) = \\max_{x \\in X} \\min_{y \\in Y}||x - y||_2

    Furthermore, the bidirectional HD is:

    .. math::

        HD(X,Y) = max(hd(X, Y), hd(Y, X))

    This is a 3D Hausdorff Distance (HD) Loss that based on morphological erosion, which provided
    a differentiable approximation of Hausdorff distance as stated in :cite:`karimi2019reducing`.
    The code is refactored on top of `here <https://github.com/PatRyg99/HausdorffLoss/
    blob/master/hausdorff_loss.py>`__.

    Args:
        alpha: controls the erosion rate in each iteration.
        k: the number of iterations of erosion.
        reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
            'none': no reduction will be applied, 'mean': the weighted mean of the output is taken,
            'sum': the output will be summed.

    Examples:
        >>> hdloss = HausdorffERLoss3D()
        >>> input = torch.randn(5, 3, 20, 20, 20)
        >>> target = (torch.rand(5, 1, 20, 20, 20) * 2).long()
        >>> res = hdloss(input, target)
    """
    conv = torch.conv3d
    max_pool = nn.AdaptiveMaxPool3d(1)

    def get_kernel(self) -> Tensor:
        if False:
            print('Hello World!')
        'Get kernel for image morphology convolution.'
        cross = tensor([[[0, 1, 0], [1, 1, 1], [0, 1, 0]]])
        bound = tensor([[[0, 0, 0], [0, 1, 0], [0, 0, 0]]])
        kernel = stack([bound, cross, bound], 1) * (1 / 7)
        return kernel[None]

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        if False:
            i = 10
            return i + 15
        'Compute 3D Hausdorff loss.\n\n        Args:\n            pred: predicted tensor with a shape of :math:`(B, C, D, H, W)`.\n                Each channel is as binary as: 1 -> fg, 0 -> bg.\n            target: target tensor with a shape of :math:`(B, 1, D, H, W)`.\n\n        Returns:\n            Estimated Hausdorff Loss.\n        '
        if pred.dim() != 5:
            raise ValueError(f'Only 3D images supported. Got {pred.dim()}.')
        return super().forward(pred, target)