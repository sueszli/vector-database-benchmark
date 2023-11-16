import torch
from torch import nn

def rgb_to_xyz(image: torch.Tensor) -> torch.Tensor:
    if False:
        return 10
    'Convert a RGB image to XYZ.\n\n    .. image:: _static/img/rgb_to_xyz.png\n\n    Args:\n        image: RGB Image to be converted to XYZ with shape :math:`(*, 3, H, W)`.\n\n    Returns:\n         XYZ version of the image with shape :math:`(*, 3, H, W)`.\n\n    Example:\n        >>> input = torch.rand(2, 3, 4, 5)\n        >>> output = rgb_to_xyz(input)  # 2x3x4x5\n    '
    if not isinstance(image, torch.Tensor):
        raise TypeError(f'Input type is not a torch.Tensor. Got {type(image)}')
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f'Input size must have a shape of (*, 3, H, W). Got {image.shape}')
    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]
    x: torch.Tensor = 0.412453 * r + 0.35758 * g + 0.180423 * b
    y: torch.Tensor = 0.212671 * r + 0.71516 * g + 0.072169 * b
    z: torch.Tensor = 0.019334 * r + 0.119193 * g + 0.950227 * b
    out: torch.Tensor = torch.stack([x, y, z], -3)
    return out

def xyz_to_rgb(image: torch.Tensor) -> torch.Tensor:
    if False:
        print('Hello World!')
    'Convert a XYZ image to RGB.\n\n    Args:\n        image: XYZ Image to be converted to RGB with shape :math:`(*, 3, H, W)`.\n\n    Returns:\n        RGB version of the image with shape :math:`(*, 3, H, W)`.\n\n    Example:\n        >>> input = torch.rand(2, 3, 4, 5)\n        >>> output = xyz_to_rgb(input)  # 2x3x4x5\n    '
    if not isinstance(image, torch.Tensor):
        raise TypeError(f'Input type is not a torch.Tensor. Got {type(image)}')
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f'Input size must have a shape of (*, 3, H, W). Got {image.shape}')
    x: torch.Tensor = image[..., 0, :, :]
    y: torch.Tensor = image[..., 1, :, :]
    z: torch.Tensor = image[..., 2, :, :]
    r: torch.Tensor = 3.2404813432005266 * x + -1.5371515162713185 * y + -0.4985363261688878 * z
    g: torch.Tensor = -0.9692549499965682 * x + 1.8759900014898907 * y + 0.0415559265582928 * z
    b: torch.Tensor = 0.0556466391351772 * x + -0.2040413383665112 * y + 1.0573110696453443 * z
    out: torch.Tensor = torch.stack([r, g, b], dim=-3)
    return out

class RgbToXyz(nn.Module):
    """Convert an image from RGB to XYZ.

    The image data is assumed to be in the range of (0, 1).

    Returns:
        XYZ version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> xyz = RgbToXyz()
        >>> output = xyz(input)  # 2x3x4x5

    Reference:
        [1] https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html
    """

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if False:
            for i in range(10):
                print('nop')
        return rgb_to_xyz(image)

class XyzToRgb(nn.Module):
    """Converts an image from XYZ to RGB.

    Returns:
        RGB version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgb = XyzToRgb()
        >>> output = rgb(input)  # 2x3x4x5

    Reference:
        [1] https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html
    """

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if False:
            print('Hello World!')
        return xyz_to_rgb(image)