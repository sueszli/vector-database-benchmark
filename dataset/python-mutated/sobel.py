from __future__ import annotations
import torch
import torch.nn.functional as F
from kornia.core import Module, Tensor, pad
from kornia.core.check import KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SHAPE
from .kernels import get_spatial_gradient_kernel2d, get_spatial_gradient_kernel3d, normalize_kernel2d

def spatial_gradient(input: Tensor, mode: str='sobel', order: int=1, normalized: bool=True) -> Tensor:
    if False:
        while True:
            i = 10
    'Compute the first order image derivative in both x and y using a Sobel operator.\n\n    .. image:: _static/img/spatial_gradient.png\n\n    Args:\n        input: input image tensor with shape :math:`(B, C, H, W)`.\n        mode: derivatives modality, can be: `sobel` or `diff`.\n        order: the order of the derivatives.\n        normalized: whether the output is normalized.\n\n    Return:\n        the derivatives of the input feature map. with shape :math:`(B, C, 2, H, W)`.\n\n    .. note::\n       See a working example `here <https://kornia.github.io/tutorials/nbs/filtering_edges.html>`__.\n\n    Examples:\n        >>> input = torch.rand(1, 3, 4, 4)\n        >>> output = spatial_gradient(input)  # 1x3x2x4x4\n        >>> output.shape\n        torch.Size([1, 3, 2, 4, 4])\n    '
    KORNIA_CHECK_IS_TENSOR(input)
    KORNIA_CHECK_SHAPE(input, ['B', 'C', 'H', 'W'])
    kernel = get_spatial_gradient_kernel2d(mode, order, device=input.device, dtype=input.dtype)
    if normalized:
        kernel = normalize_kernel2d(kernel)
    (b, c, h, w) = input.shape
    tmp_kernel = kernel[:, None, ...]
    spatial_pad = [kernel.size(1) // 2, kernel.size(1) // 2, kernel.size(2) // 2, kernel.size(2) // 2]
    out_channels: int = 3 if order == 2 else 2
    padded_inp: Tensor = pad(input.reshape(b * c, 1, h, w), spatial_pad, 'replicate')
    out = F.conv2d(padded_inp, tmp_kernel, groups=1, padding=0, stride=1)
    return out.reshape(b, c, out_channels, h, w)

def spatial_gradient3d(input: Tensor, mode: str='diff', order: int=1) -> Tensor:
    if False:
        for i in range(10):
            print('nop')
    'Compute the first and second order volume derivative in x, y and d using a diff operator.\n\n    Args:\n        input: input features tensor with shape :math:`(B, C, D, H, W)`.\n        mode: derivatives modality, can be: `sobel` or `diff`.\n        order: the order of the derivatives.\n\n    Return:\n        the spatial gradients of the input feature map with shape math:`(B, C, 3, D, H, W)`\n        or :math:`(B, C, 6, D, H, W)`.\n\n    Examples:\n        >>> input = torch.rand(1, 4, 2, 4, 4)\n        >>> output = spatial_gradient3d(input)\n        >>> output.shape\n        torch.Size([1, 4, 3, 2, 4, 4])\n    '
    KORNIA_CHECK_IS_TENSOR(input)
    KORNIA_CHECK_SHAPE(input, ['B', 'C', 'D', 'H', 'W'])
    (b, c, d, h, w) = input.shape
    dev = input.device
    dtype = input.dtype
    if mode == 'diff' and order == 1:
        x: Tensor = pad(input, 6 * [1], 'replicate')
        center = slice(1, -1)
        left = slice(0, -2)
        right = slice(2, None)
        out = torch.empty(b, c, 3, d, h, w, device=dev, dtype=dtype)
        out[..., 0, :, :, :] = x[..., center, center, right] - x[..., center, center, left]
        out[..., 1, :, :, :] = x[..., center, right, center] - x[..., center, left, center]
        out[..., 2, :, :, :] = x[..., right, center, center] - x[..., left, center, center]
        out = 0.5 * out
    else:
        kernel = get_spatial_gradient_kernel3d(mode, order, device=dev, dtype=dtype)
        tmp_kernel = kernel.repeat(c, 1, 1, 1, 1)
        kernel_flip = tmp_kernel.flip(-3)
        spatial_pad = [kernel.size(2) // 2, kernel.size(2) // 2, kernel.size(3) // 2, kernel.size(3) // 2, kernel.size(4) // 2, kernel.size(4) // 2]
        out_ch: int = 6 if order == 2 else 3
        out = F.conv3d(pad(input, spatial_pad, 'replicate'), kernel_flip, padding=0, groups=c).view(b, c, out_ch, d, h, w)
    return out

def sobel(input: Tensor, normalized: bool=True, eps: float=1e-06) -> Tensor:
    if False:
        for i in range(10):
            print('nop')
    'Compute the Sobel operator and returns the magnitude per channel.\n\n    .. image:: _static/img/sobel.png\n\n    Args:\n        input: the input image with shape :math:`(B,C,H,W)`.\n        normalized: if True, L1 norm of the kernel is set to 1.\n        eps: regularization number to avoid NaN during backprop.\n\n    Return:\n        the sobel edge gradient magnitudes map with shape :math:`(B,C,H,W)`.\n\n    .. note::\n       See a working example `here <https://kornia.github.io/tutorials/nbs/filtering_edges.html>`__.\n\n    Example:\n        >>> input = torch.rand(1, 3, 4, 4)\n        >>> output = sobel(input)  # 1x3x4x4\n        >>> output.shape\n        torch.Size([1, 3, 4, 4])\n    '
    KORNIA_CHECK_IS_TENSOR(input)
    KORNIA_CHECK_SHAPE(input, ['B', 'C', 'H', 'W'])
    edges: Tensor = spatial_gradient(input, normalized=normalized)
    gx: Tensor = edges[:, :, 0]
    gy: Tensor = edges[:, :, 1]
    magnitude: Tensor = torch.sqrt(gx * gx + gy * gy + eps)
    return magnitude

class SpatialGradient(Module):
    """Compute the first order image derivative in both x and y using a Sobel operator.

    Args:
        mode: derivatives modality, can be: `sobel` or `diff`.
        order: the order of the derivatives.
        normalized: whether the output is normalized.

    Return:
        the sobel edges of the input feature map.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, 2, H, W)`

    Examples:
        >>> input = torch.rand(1, 3, 4, 4)
        >>> output = SpatialGradient()(input)  # 1x3x2x4x4
    """

    def __init__(self, mode: str='sobel', order: int=1, normalized: bool=True) -> None:
        if False:
            return 10
        super().__init__()
        self.normalized: bool = normalized
        self.order: int = order
        self.mode: str = mode

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        return f'{self.__class__.__name__}(order={self.order}, normalized={self.normalized}, mode={self.mode})'

    def forward(self, input: Tensor) -> Tensor:
        if False:
            i = 10
            return i + 15
        return spatial_gradient(input, self.mode, self.order, self.normalized)

class SpatialGradient3d(Module):
    """Compute the first and second order volume derivative in x, y and d using a diff operator.

    Args:
        mode: derivatives modality, can be: `sobel` or `diff`.
        order: the order of the derivatives.

    Return:
        the spatial gradients of the input feature map.

    Shape:
        - Input: :math:`(B, C, D, H, W)`. D, H, W are spatial dimensions, gradient is calculated w.r.t to them.
        - Output: :math:`(B, C, 3, D, H, W)` or :math:`(B, C, 6, D, H, W)`

    Examples:
        >>> input = torch.rand(1, 4, 2, 4, 4)
        >>> output = SpatialGradient3d()(input)
        >>> output.shape
        torch.Size([1, 4, 3, 2, 4, 4])
    """

    def __init__(self, mode: str='diff', order: int=1) -> None:
        if False:
            return 10
        super().__init__()
        self.order: int = order
        self.mode: str = mode
        self.kernel = get_spatial_gradient_kernel3d(mode, order)

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        return f'{self.__class__.__name__}(order={self.order}, mode={self.mode})'

    def forward(self, input: Tensor) -> Tensor:
        if False:
            return 10
        return spatial_gradient3d(input, self.mode, self.order)

class Sobel(Module):
    """Compute the Sobel operator and returns the magnitude per channel.

    Args:
        normalized: if True, L1 norm of the kernel is set to 1.
        eps: regularization number to avoid NaN during backprop.

    Return:
        the sobel edge gradient magnitudes map.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Examples:
        >>> input = torch.rand(1, 3, 4, 4)
        >>> output = Sobel()(input)  # 1x3x4x4
    """

    def __init__(self, normalized: bool=True, eps: float=1e-06) -> None:
        if False:
            while True:
                i = 10
        super().__init__()
        self.normalized: bool = normalized
        self.eps: float = eps

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        return f'{self.__class__.__name__}(normalized={self.normalized})'

    def forward(self, input: Tensor) -> Tensor:
        if False:
            print('Hello World!')
        return sobel(input, self.normalized, self.eps)