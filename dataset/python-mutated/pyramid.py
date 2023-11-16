from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from kornia.core import Module, Tensor, pad, stack, tensor
from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SHAPE
from kornia.filters import filter2d, gaussian_blur2d
__all__ = ['PyrDown', 'PyrUp', 'ScalePyramid', 'pyrdown', 'pyrup', 'build_pyramid', 'build_laplacian_pyramid', 'upscale_double']

def _get_pyramid_gaussian_kernel() -> Tensor:
    if False:
        print('Hello World!')
    'Utility function that return a pre-computed gaussian kernel.'
    return tensor([[[1.0, 4.0, 6.0, 4.0, 1.0], [4.0, 16.0, 24.0, 16.0, 4.0], [6.0, 24.0, 36.0, 24.0, 6.0], [4.0, 16.0, 24.0, 16.0, 4.0], [1.0, 4.0, 6.0, 4.0, 1.0]]]) / 256.0

class PyrDown(Module):
    """Blur a tensor and downsamples it.

    Args:
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``.
        align_corners: interpolation flag.
        factor: the downsampling factor

    Return:
        the downsampled tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H / 2, W / 2)`

    Examples:
        >>> input = torch.rand(1, 2, 4, 4)
        >>> output = PyrDown()(input)  # 1x2x2x2
    """

    def __init__(self, border_type: str='reflect', align_corners: bool=False, factor: float=2.0) -> None:
        if False:
            print('Hello World!')
        super().__init__()
        self.border_type: str = border_type
        self.align_corners: bool = align_corners
        self.factor: float = factor

    def forward(self, input: Tensor) -> Tensor:
        if False:
            for i in range(10):
                print('nop')
        return pyrdown(input, self.border_type, self.align_corners, self.factor)

class PyrUp(Module):
    """Upsample a tensor and then blurs it.

    Args:
        borde_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``.
        align_corners: interpolation flag.

    Return:
        the upsampled tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H * 2, W * 2)`

    Examples:
        >>> input = torch.rand(1, 2, 4, 4)
        >>> output = PyrUp()(input)  # 1x2x8x8
    """

    def __init__(self, border_type: str='reflect', align_corners: bool=False) -> None:
        if False:
            return 10
        super().__init__()
        self.border_type: str = border_type
        self.align_corners: bool = align_corners

    def forward(self, input: Tensor) -> Tensor:
        if False:
            return 10
        return pyrup(input, self.border_type, self.align_corners)

class ScalePyramid(Module):
    """Create an scale pyramid of image, usually used for local feature detection.

    Images are consequently smoothed with Gaussian blur and downscaled.

    Args:
        n_levels: number of the levels in octave.
        init_sigma: initial blur level.
        min_size: the minimum size of the octave in pixels.
        double_image: add 2x upscaled image as 1st level of pyramid. OpenCV SIFT does this.

    Returns:
        1st output: images
        2nd output: sigmas (coefficients for scale conversion)
        3rd output: pixelDists (coefficients for coordinate conversion)

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output 1st: :math:`[(B, C, NL, H, W), (B, C, NL, H/2, W/2), ...]`
        - Output 2nd: :math:`[(B, NL), (B, NL), (B, NL), ...]`
        - Output 3rd: :math:`[(B, NL), (B, NL), (B, NL), ...]`

    Examples:
        >>> input = torch.rand(2, 4, 100, 100)
        >>> sp, sigmas, pds = ScalePyramid(3, 15)(input)
    """

    def __init__(self, n_levels: int=3, init_sigma: float=1.6, min_size: int=15, double_image: bool=False) -> None:
        if False:
            print('Hello World!')
        super().__init__()
        self.n_levels = n_levels
        self.extra_levels: int = 3
        self.init_sigma = init_sigma
        self.min_size = min_size
        self.border = min_size // 2 - 1
        self.sigma_step = 2 ** (1.0 / float(self.n_levels))
        self.double_image = double_image

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        return f'{self.__class__.__name__}(n_levels={self.n_levels}, init_sigma={self.init_sigma}, min_size={self.min_size}, extra_levels={self.extra_levels}, border={self.border}, sigma_step={self.sigma_step}, double_image={self.double_image})'

    def get_kernel_size(self, sigma: float) -> int:
        if False:
            for i in range(10):
                print('nop')
        ksize = int(2.0 * 4.0 * sigma + 1.0)
        if ksize % 2 == 0:
            ksize += 1
        return ksize

    def get_first_level(self, input: Tensor) -> tuple[Tensor, float, float]:
        if False:
            return 10
        pixel_distance = 1.0
        cur_sigma = 0.5
        if self.double_image:
            x = upscale_double(input)
            pixel_distance = 0.5
            cur_sigma *= 2.0
        else:
            x = input
        if self.init_sigma > cur_sigma:
            sigma = max(math.sqrt(self.init_sigma ** 2 - cur_sigma ** 2), 0.01)
            ksize = self.get_kernel_size(sigma)
            cur_level = gaussian_blur2d(x, (ksize, ksize), (sigma, sigma))
            cur_sigma = self.init_sigma
        else:
            cur_level = x
        return (cur_level, cur_sigma, pixel_distance)

    def forward(self, x: Tensor) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
        if False:
            while True:
                i = 10
        (bs, _, _, _) = x.size()
        (cur_level, cur_sigma, pixel_distance) = self.get_first_level(x)
        sigmas = [cur_sigma * torch.ones(bs, self.n_levels + self.extra_levels).to(x.device).to(x.dtype)]
        pixel_dists = [pixel_distance * torch.ones(bs, self.n_levels + self.extra_levels).to(x.device).to(x.dtype)]
        pyr = [[cur_level]]
        oct_idx = 0
        while True:
            cur_level = pyr[-1][0]
            for level_idx in range(1, self.n_levels + self.extra_levels):
                sigma = cur_sigma * math.sqrt(self.sigma_step ** 2 - 1.0)
                ksize = self.get_kernel_size(sigma)
                ksize = min(ksize, cur_level.size(2), cur_level.size(3))
                if ksize % 2 == 0:
                    ksize += 1
                cur_level = gaussian_blur2d(cur_level, (ksize, ksize), (sigma, sigma))
                cur_sigma *= self.sigma_step
                pyr[-1].append(cur_level)
                sigmas[-1][:, level_idx] = cur_sigma
                pixel_dists[-1][:, level_idx] = pixel_distance
            _pyr = pyr[-1][-self.extra_levels]
            nextOctaveFirstLevel = _pyr[:, :, ::2, ::2]
            pixel_distance *= 2.0
            cur_sigma = self.init_sigma
            if min(nextOctaveFirstLevel.size(2), nextOctaveFirstLevel.size(3)) <= self.min_size:
                break
            pyr.append([nextOctaveFirstLevel])
            sigmas.append(cur_sigma * torch.ones(bs, self.n_levels + self.extra_levels).to(x.device))
            pixel_dists.append(pixel_distance * torch.ones(bs, self.n_levels + self.extra_levels).to(x.device))
            oct_idx += 1
        output_pyr = [stack(i, 2) for i in pyr]
        return (output_pyr, sigmas, pixel_dists)

def pyrdown(input: Tensor, border_type: str='reflect', align_corners: bool=False, factor: float=2.0) -> Tensor:
    if False:
        print('Hello World!')
    "Blur a tensor and downsamples it.\n\n    .. image:: _static/img/pyrdown.png\n\n    Args:\n        input: the tensor to be downsampled.\n        border_type: the padding mode to be applied before convolving.\n          The expected modes are: ``'constant'``, ``'reflect'``,\n          ``'replicate'`` or ``'circular'``.\n        align_corners: interpolation flag.\n        factor: the downsampling factor\n\n    Return:\n        the downsampled tensor.\n\n    Examples:\n        >>> input = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)\n        >>> pyrdown(input, align_corners=True)\n        tensor([[[[ 3.7500,  5.2500],\n                  [ 9.7500, 11.2500]]]])\n    "
    KORNIA_CHECK_SHAPE(input, ['B', 'C', 'H', 'W'])
    kernel: Tensor = _get_pyramid_gaussian_kernel()
    (_, _, height, width) = input.shape
    x_blur: Tensor = filter2d(input, kernel, border_type)
    out: Tensor = F.interpolate(x_blur, size=(int(float(height) / factor), int(float(width) // factor)), mode='bilinear', align_corners=align_corners)
    return out

def pyrup(input: Tensor, border_type: str='reflect', align_corners: bool=False) -> Tensor:
    if False:
        for i in range(10):
            print('nop')
    "Upsample a tensor and then blurs it.\n\n    .. image:: _static/img/pyrup.png\n\n    Args:\n        input: the tensor to be downsampled.\n        border_type: the padding mode to be applied before convolving.\n          The expected modes are: ``'constant'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.\n        align_corners: interpolation flag.\n\n    Return:\n        the downsampled tensor.\n\n    Examples:\n        >>> input = torch.arange(4, dtype=torch.float32).reshape(1, 1, 2, 2)\n        >>> pyrup(input, align_corners=True)\n        tensor([[[[0.7500, 0.8750, 1.1250, 1.2500],\n                  [1.0000, 1.1250, 1.3750, 1.5000],\n                  [1.5000, 1.6250, 1.8750, 2.0000],\n                  [1.7500, 1.8750, 2.1250, 2.2500]]]])\n    "
    KORNIA_CHECK_SHAPE(input, ['B', 'C', 'H', 'W'])
    kernel: Tensor = _get_pyramid_gaussian_kernel()
    (_, _, height, width) = input.shape
    x_up: Tensor = F.interpolate(input, size=(height * 2, width * 2), mode='bilinear', align_corners=align_corners)
    x_blur: Tensor = filter2d(x_up, kernel, border_type)
    return x_blur

def build_pyramid(input: Tensor, max_level: int, border_type: str='reflect', align_corners: bool=False) -> list[Tensor]:
    if False:
        for i in range(10):
            print('nop')
    "Construct the Gaussian pyramid for a tensor image.\n\n    .. image:: _static/img/build_pyramid.png\n\n    The function constructs a vector of images and builds the Gaussian pyramid\n    by recursively applying pyrDown to the previously built pyramid layers.\n\n    Args:\n        input : the tensor to be used to construct the pyramid.\n        max_level: 0-based index of the last (the smallest) pyramid layer.\n          It must be non-negative.\n        border_type: the padding mode to be applied before convolving.\n          The expected modes are: ``'constant'``, ``'reflect'``,\n          ``'replicate'`` or ``'circular'``.\n        align_corners: interpolation flag.\n\n    Shape:\n        - Input: :math:`(B, C, H, W)`\n        - Output :math:`[(B, C, H, W), (B, C, H/2, W/2), ...]`\n    "
    KORNIA_CHECK_SHAPE(input, ['B', 'C', 'H', 'W'])
    KORNIA_CHECK(isinstance(max_level, int) or max_level < 0, f'Invalid max_level, it must be a positive integer. Got: {max_level}')
    pyramid: list[Tensor] = []
    pyramid.append(input)
    for _ in range(max_level - 1):
        img_curr: Tensor = pyramid[-1]
        img_down: Tensor = pyrdown(img_curr, border_type, align_corners)
        pyramid.append(img_down)
    return pyramid

def is_powerof_two(x: int) -> bool:
    if False:
        for i in range(10):
            print('nop')
    return bool(x) and (not x & x - 1)

def find_next_powerof_two(x: int) -> int:
    if False:
        while True:
            i = 10
    n = math.ceil(math.log(x) / math.log(2))
    return 2 ** n

def build_laplacian_pyramid(input: Tensor, max_level: int, border_type: str='reflect', align_corners: bool=False) -> list[Tensor]:
    if False:
        print('Hello World!')
    "Construct the Laplacian pyramid for a tensor image.\n\n    The function constructs a vector of images and builds the Laplacian pyramid\n    by recursively computing the difference after applying\n    pyrUp to the adjacent layer in it's Gaussian pyramid.\n\n    See :cite:`burt1987laplacian` for more details.\n\n    Args:\n        input : the tensor to be used to construct the pyramid with shape :math:`(B, C, H, W)`.\n        max_level: 0-based index of the last (the smallest) pyramid layer.\n          It must be non-negative.\n        border_type: the padding mode to be applied before convolving.\n          The expected modes are: ``'constant'``, ``'reflect'``,\n          ``'replicate'`` or ``'circular'``.\n        align_corners: interpolation flag.\n\n    Return:\n        Output: :math:`[(B, C, H, W), (B, C, H/2, W/2), ...]`\n    "
    KORNIA_CHECK_SHAPE(input, ['B', 'C', 'H', 'W'])
    KORNIA_CHECK(isinstance(max_level, int) or max_level < 0, f'Invalid max_level, it must be a positive integer. Got: {max_level}')
    h = input.size()[2]
    w = input.size()[3]
    require_padding = not (is_powerof_two(w) or is_powerof_two(h))
    if require_padding:
        padding = (0, find_next_powerof_two(w) - w, 0, find_next_powerof_two(h) - h)
        input = pad(input, padding, 'reflect')
    gaussian_pyramid: list[Tensor] = build_pyramid(input, max_level, border_type, align_corners)
    laplacian_pyramid: list[Tensor] = []
    for i in range(max_level - 1):
        img_expand: Tensor = pyrup(gaussian_pyramid[i + 1], border_type, align_corners)
        laplacian: Tensor = gaussian_pyramid[i] - img_expand
        laplacian_pyramid.append(laplacian)
    laplacian_pyramid.append(gaussian_pyramid[-1])
    return laplacian_pyramid

def upscale_double(x: Tensor) -> Tensor:
    if False:
        i = 10
        return i + 15
    'Upscale image by the factor of 2, even indices maps to original indices.\n\n    Odd indices are linearly interpolated from the even ones.\n\n    Args:\n        x: input image.\n\n    Shape:\n        - Input: :math:`(*, H, W)`\n        - Output :math:`(*, H, W)`\n    '
    KORNIA_CHECK_IS_TENSOR(x)
    KORNIA_CHECK_SHAPE(x, ['*', 'H', 'W'])
    double_shape = x.shape[:-2] + (x.shape[-2] * 2, x.shape[-1] * 2)
    upscaled = torch.zeros(double_shape, device=x.device, dtype=x.dtype)
    upscaled[..., ::2, ::2] = x
    upscaled[..., ::2, 1::2][..., :-1] = (upscaled[..., ::2, ::2][..., :-1] + upscaled[..., ::2, 2::2]) / 2
    upscaled[..., ::2, -1] = upscaled[..., ::2, -2]
    upscaled[..., 1::2, :][..., :-1, :] = (upscaled[..., ::2, :][..., :-1, :] + upscaled[..., 2::2, :]) / 2
    upscaled[..., -1, :] = upscaled[..., -2, :]
    return upscaled