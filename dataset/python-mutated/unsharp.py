from __future__ import annotations
from kornia.core import Module, Tensor
from .gaussian import gaussian_blur2d

def unsharp_mask(input: Tensor, kernel_size: tuple[int, int] | int, sigma: tuple[float, float] | Tensor, border_type: str='reflect') -> Tensor:
    if False:
        while True:
            i = 10
    "Create an operator that sharpens a tensor by applying operation out = 2 * image - gaussian_blur2d(image).\n\n    .. image:: _static/img/unsharp_mask.png\n\n    Args:\n        input: the input tensor with shape :math:`(B,C,H,W)`.\n        kernel_size: the size of the kernel.\n        sigma: the standard deviation of the kernel.\n        border_type: the padding mode to be applied before convolving.\n          The expected modes are: ``'constant'``, ``'reflect'``,\n          ``'replicate'`` or ``'circular'``.\n\n    Returns:\n        the blurred tensor with shape :math:`(B,C,H,W)`.\n\n    Examples:\n        >>> input = torch.rand(2, 4, 5, 5)\n        >>> output = unsharp_mask(input, (3, 3), (1.5, 1.5))\n        >>> output.shape\n        torch.Size([2, 4, 5, 5])\n    "
    data_blur: Tensor = gaussian_blur2d(input, kernel_size, sigma, border_type)
    data_sharpened: Tensor = input + (input - data_blur)
    return data_sharpened

class UnsharpMask(Module):
    """Create an operator that sharpens image with: out = 2 * image - gaussian_blur2d(image).

    Args:
        kernel_size: the size of the kernel.
        sigma: the standard deviation of the kernel.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``.

    Returns:
        the sharpened tensor with shape :math:`(B,C,H,W)`.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    .. note::
       See a working example `here <https://kornia.github.io/tutorials/nbs/unsharp_mask.html>`__.

    Examples:
        >>> input = torch.rand(2, 4, 5, 5)
        >>> sharpen = UnsharpMask((3, 3), (1.5, 1.5))
        >>> output = sharpen(input)
        >>> output.shape
        torch.Size([2, 4, 5, 5])
    """

    def __init__(self, kernel_size: tuple[int, int] | int, sigma: tuple[float, float] | Tensor, border_type: str='reflect') -> None:
        if False:
            return 10
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.border_type = border_type

    def forward(self, input: Tensor) -> Tensor:
        if False:
            i = 10
            return i + 15
        return unsharp_mask(input, self.kernel_size, self.sigma, self.border_type)