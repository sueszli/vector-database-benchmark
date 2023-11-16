import torch
from torch import Tensor, nn

def _rgb_to_y(r: Tensor, g: Tensor, b: Tensor) -> Tensor:
    if False:
        print('Hello World!')
    y: Tensor = 0.299 * r + 0.587 * g + 0.114 * b
    return y

def rgb_to_ycbcr(image: Tensor) -> Tensor:
    if False:
        i = 10
        return i + 15
    'Convert an RGB image to YCbCr.\n\n    .. image:: _static/img/rgb_to_ycbcr.png\n\n    Args:\n        image: RGB Image to be converted to YCbCr with shape :math:`(*, 3, H, W)`.\n\n    Returns:\n        YCbCr version of the image with shape :math:`(*, 3, H, W)`.\n\n    Examples:\n        >>> input = torch.rand(2, 3, 4, 5)\n        >>> output = rgb_to_ycbcr(input)  # 2x3x4x5\n    '
    if not isinstance(image, Tensor):
        raise TypeError(f'Input type is not a Tensor. Got {type(image)}')
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f'Input size must have a shape of (*, 3, H, W). Got {image.shape}')
    r: Tensor = image[..., 0, :, :]
    g: Tensor = image[..., 1, :, :]
    b: Tensor = image[..., 2, :, :]
    delta: float = 0.5
    y: Tensor = _rgb_to_y(r, g, b)
    cb: Tensor = (b - y) * 0.564 + delta
    cr: Tensor = (r - y) * 0.713 + delta
    return torch.stack([y, cb, cr], -3)

def rgb_to_y(image: Tensor) -> Tensor:
    if False:
        print('Hello World!')
    'Convert an RGB image to Y.\n\n    Args:\n        image: RGB Image to be converted to Y with shape :math:`(*, 3, H, W)`.\n\n    Returns:\n        Y version of the image with shape :math:`(*, 1, H, W)`.\n\n    Examples:\n        >>> input = torch.rand(2, 3, 4, 5)\n        >>> output = rgb_to_y(input)  # 2x1x4x5\n    '
    if not isinstance(image, Tensor):
        raise TypeError(f'Input type is not a Tensor. Got {type(image)}')
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f'Input size must have a shape of (*, 3, H, W). Got {image.shape}')
    r: Tensor = image[..., 0:1, :, :]
    g: Tensor = image[..., 1:2, :, :]
    b: Tensor = image[..., 2:3, :, :]
    y: Tensor = _rgb_to_y(r, g, b)
    return y

def ycbcr_to_rgb(image: Tensor) -> Tensor:
    if False:
        i = 10
        return i + 15
    'Convert an YCbCr image to RGB.\n\n    The image data is assumed to be in the range of (0, 1).\n\n    Args:\n        image: YCbCr Image to be converted to RGB with shape :math:`(*, 3, H, W)`.\n\n    Returns:\n        RGB version of the image with shape :math:`(*, 3, H, W)`.\n\n    Examples:\n        >>> input = torch.rand(2, 3, 4, 5)\n        >>> output = ycbcr_to_rgb(input)  # 2x3x4x5\n    '
    if not isinstance(image, Tensor):
        raise TypeError(f'Input type is not a Tensor. Got {type(image)}')
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f'Input size must have a shape of (*, 3, H, W). Got {image.shape}')
    y: Tensor = image[..., 0, :, :]
    cb: Tensor = image[..., 1, :, :]
    cr: Tensor = image[..., 2, :, :]
    delta: float = 0.5
    cb_shifted: Tensor = cb - delta
    cr_shifted: Tensor = cr - delta
    r: Tensor = y + 1.403 * cr_shifted
    g: Tensor = y - 0.714 * cr_shifted - 0.344 * cb_shifted
    b: Tensor = y + 1.773 * cb_shifted
    return torch.stack([r, g, b], -3).clamp(0, 1)

class RgbToYcbcr(nn.Module):
    """Convert an image from RGB to YCbCr.

    The image data is assumed to be in the range of (0, 1).

    Returns:
        YCbCr version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> ycbcr = RgbToYcbcr()
        >>> output = ycbcr(input)  # 2x3x4x5
    """

    def forward(self, image: Tensor) -> Tensor:
        if False:
            print('Hello World!')
        return rgb_to_ycbcr(image)

class YcbcrToRgb(nn.Module):
    """Convert an image from YCbCr to Rgb.

    The image data is assumed to be in the range of (0, 1).

    Returns:
        RGB version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgb = YcbcrToRgb()
        >>> output = rgb(input)  # 2x3x4x5
    """

    def forward(self, image: Tensor) -> Tensor:
        if False:
            i = 10
            return i + 15
        return ycbcr_to_rgb(image)