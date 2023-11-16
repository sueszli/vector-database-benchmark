from __future__ import annotations
import torch
import torch.nn.functional as F
from kornia.core import Module, Tensor, concatenate, stack, tensor, where, zeros
from kornia.filters.sobel import spatial_gradient3d
from kornia.geometry.conversions import normalize_pixel_coordinates, normalize_pixel_coordinates3d
from kornia.utils import create_meshgrid, create_meshgrid3d
from kornia.utils._compat import torch_version_ge
from kornia.utils.helpers import safe_solve_with_mask
from .dsnt import spatial_expectation2d, spatial_softmax2d
from .nms import nms3d

def _get_window_grid_kernel2d(h: int, w: int, device: torch.device=torch.device('cpu')) -> Tensor:
    if False:
        return 10
    'Helper function, which generates a kernel to with window coordinates, residual to window center.\n\n    Args:\n         h: kernel height.\n         : kernel width.\n         device: device, on which generate.\n\n    Returns:\n        conv_kernel [2x1xhxw]\n    '
    window_grid2d = create_meshgrid(h, w, False, device=device)
    window_grid2d = normalize_pixel_coordinates(window_grid2d, h, w)
    conv_kernel = window_grid2d.permute(3, 0, 1, 2)
    return conv_kernel

def _get_center_kernel2d(h: int, w: int, device: torch.device=torch.device('cpu')) -> Tensor:
    if False:
        print('Hello World!')
    'Helper function, which generates a kernel to return center coordinates, when applied with F.conv2d to 2d\n    coordinates grid.\n\n    Args:\n        h: kernel height.\n        w: kernel width.\n        device: device, on which generate.\n\n    Returns:\n        conv_kernel [2x2xhxw].\n    '
    center_kernel = zeros(2, 2, h, w, device=device)
    if h % 2 != 0:
        h_i1 = h // 2
        h_i2 = h // 2 + 1
    else:
        h_i1 = h // 2 - 1
        h_i2 = h // 2 + 1
    if w % 2 != 0:
        w_i1 = w // 2
        w_i2 = w // 2 + 1
    else:
        w_i1 = w // 2 - 1
        w_i2 = w // 2 + 1
    center_kernel[(0, 1), (0, 1), h_i1:h_i2, w_i1:w_i2] = 1.0 / float((h_i2 - h_i1) * (w_i2 - w_i1))
    return center_kernel

def _get_center_kernel3d(d: int, h: int, w: int, device: torch.device=torch.device('cpu')) -> Tensor:
    if False:
        return 10
    'Helper function, which generates a kernel to return center coordinates, when applied with F.conv2d to 3d\n    coordinates grid.\n\n    Args:\n        d: kernel depth.\n        h: kernel height.\n        w: kernel width.\n        device: device, on which generate.\n\n    Returns:\n        conv_kernel [3x3xdxhxw].\n    '
    center_kernel = zeros(3, 3, d, h, w, device=device)
    if h % 2 != 0:
        h_i1 = h // 2
        h_i2 = h // 2 + 1
    else:
        h_i1 = h // 2 - 1
        h_i2 = h // 2 + 1
    if w % 2 != 0:
        w_i1 = w // 2
        w_i2 = w // 2 + 1
    else:
        w_i1 = w // 2 - 1
        w_i2 = w // 2 + 1
    if d % 2 != 0:
        d_i1 = d // 2
        d_i2 = d // 2 + 1
    else:
        d_i1 = d // 2 - 1
        d_i2 = d // 2 + 1
    center_num = float((h_i2 - h_i1) * (w_i2 - w_i1) * (d_i2 - d_i1))
    center_kernel[(0, 1, 2), (0, 1, 2), d_i1:d_i2, h_i1:h_i2, w_i1:w_i2] = 1.0 / center_num
    return center_kernel

def _get_window_grid_kernel3d(d: int, h: int, w: int, device: torch.device=torch.device('cpu')) -> Tensor:
    if False:
        i = 10
        return i + 15
    'Helper function, which generates a kernel to return coordinates, residual to window center.\n\n    Args:\n        d: kernel depth.\n        h: kernel height.\n        w: kernel width.\n        device: device, on which generate.\n\n    Returns:\n        conv_kernel [3x1xdxhxw]\n    '
    grid2d = create_meshgrid(h, w, True, device=device)
    if d > 1:
        z = torch.linspace(-1, 1, d, device=device).view(d, 1, 1, 1)
    else:
        z = zeros(1, 1, 1, 1, device=device)
    grid3d = concatenate([z.repeat(1, h, w, 1).contiguous(), grid2d.repeat(d, 1, 1, 1)], 3)
    conv_kernel = grid3d.permute(3, 0, 1, 2).unsqueeze(1)
    return conv_kernel

class ConvSoftArgmax2d(Module):
    """Module that calculates soft argmax 2d per window.

    See
    :func: `~kornia.geometry.subpix.conv_soft_argmax2d` for details.
    """

    def __init__(self, kernel_size: tuple[int, int]=(3, 3), stride: tuple[int, int]=(1, 1), padding: tuple[int, int]=(1, 1), temperature: Tensor | float=tensor(1.0), normalized_coordinates: bool=True, eps: float=1e-08, output_value: bool=False) -> None:
        if False:
            print('Hello World!')
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.temperature = temperature
        self.normalized_coordinates = normalized_coordinates
        self.eps = eps
        self.output_value = output_value

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        return f'{self.__class__.__name__}(kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, temperature={self.temperature}, normalized_coordinates={self.normalized_coordinates}, eps={self.eps}, output_value={self.output_value})'

    def forward(self, x: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        if False:
            for i in range(10):
                print('nop')
        return conv_soft_argmax2d(x, self.kernel_size, self.stride, self.padding, self.temperature, self.normalized_coordinates, self.eps, self.output_value)

class ConvSoftArgmax3d(Module):
    """Module that calculates soft argmax 3d per window.

    See
    :func: `~kornia.geometry.subpix.conv_soft_argmax3d` for details.
    """

    def __init__(self, kernel_size: tuple[int, int, int]=(3, 3, 3), stride: tuple[int, int, int]=(1, 1, 1), padding: tuple[int, int, int]=(1, 1, 1), temperature: Tensor | float=tensor(1.0), normalized_coordinates: bool=False, eps: float=1e-08, output_value: bool=True, strict_maxima_bonus: float=0.0) -> None:
        if False:
            print('Hello World!')
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.temperature = temperature
        self.normalized_coordinates = normalized_coordinates
        self.eps = eps
        self.output_value = output_value
        self.strict_maxima_bonus = strict_maxima_bonus

    def __repr__(self) -> str:
        if False:
            return 10
        return f'{self.__class__.__name__}(kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, temperature={self.temperature}, normalized_coordinates={self.normalized_coordinates}, eps={self.eps}, strict_maxima_bonus={self.strict_maxima_bonus}, output_value={self.output_value})'

    def forward(self, x: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        if False:
            for i in range(10):
                print('nop')
        return conv_soft_argmax3d(x, self.kernel_size, self.stride, self.padding, self.temperature, self.normalized_coordinates, self.eps, self.output_value, self.strict_maxima_bonus)

def conv_soft_argmax2d(input: Tensor, kernel_size: tuple[int, int]=(3, 3), stride: tuple[int, int]=(1, 1), padding: tuple[int, int]=(1, 1), temperature: Tensor | float=tensor(1.0), normalized_coordinates: bool=True, eps: float=1e-08, output_value: bool=False) -> Tensor | tuple[Tensor, Tensor]:
    if False:
        i = 10
        return i + 15
    'Compute the convolutional spatial Soft-Argmax 2D over the windows of a given heatmap.\n\n    .. math::\n        ij(X) = \\frac{\\sum{(i,j)} * exp(x / T)  \\in X} {\\sum{exp(x / T)  \\in X}}\n\n    .. math::\n        val(X) = \\frac{\\sum{x * exp(x / T)  \\in X}} {\\sum{exp(x / T)  \\in X}}\n\n    where :math:`T` is temperature.\n\n    Args:\n        input: the given heatmap with shape :math:`(N, C, H_{in}, W_{in})`.\n        kernel_size: the size of the window.\n        stride: the stride of the window.\n        padding: input zero padding.\n        temperature: factor to apply to input.\n        normalized_coordinates: whether to return the coordinates normalized in the range of :math:`[-1, 1]`.\n            Otherwise, it will return the coordinates in the range of the input shape.\n        eps: small value to avoid zero division.\n        output_value: if True, val is output, if False, only ij.\n\n    Returns:\n        Function has two outputs - argmax coordinates and the softmaxpooled heatmap values themselves.\n        On each window, the function computed returns with shapes :math:`(N, C, 2, H_{out},\n        W_{out})`, :math:`(N, C, H_{out}, W_{out})`,\n\n        where\n\n         .. math::\n             H_{out} = \\left\\lfloor\\frac{H_{in}  + 2 \\times \\text{padding}[0] -\n               (\\text{kernel\\_size}[0] - 1) - 1}{\\text{stride}[0]} + 1\\right\\rfloor\n\n         .. math::\n             W_{out} = \\left\\lfloor\\frac{W_{in}  + 2 \\times \\text{padding}[1] -\n               (\\text{kernel\\_size}[1] - 1) - 1}{\\text{stride}[1]} + 1\\right\\rfloor\n\n    Examples:\n        >>> input = torch.randn(20, 16, 50, 32)\n        >>> nms_coords, nms_val = conv_soft_argmax2d(input, (3,3), (2,2), (1,1), output_value=True)\n    '
    if not torch.is_tensor(input):
        raise TypeError(f'Input type is not a Tensor. Got {type(input)}')
    if not len(input.shape) == 4:
        raise ValueError(f'Invalid input shape, we expect BxCxHxW. Got: {input.shape}')
    if temperature <= 0:
        raise ValueError(f'Temperature should be positive float or tensor. Got: {temperature}')
    (b, c, h, w) = input.shape
    (ky, kx) = kernel_size
    device: torch.device = input.device
    dtype: torch.dtype = input.dtype
    input = input.view(b * c, 1, h, w)
    center_kernel: Tensor = _get_center_kernel2d(ky, kx, device).to(dtype)
    window_kernel: Tensor = _get_window_grid_kernel2d(ky, kx, device).to(dtype)
    x_max = F.adaptive_max_pool2d(input, (1, 1))
    x_exp = ((input - x_max.detach()) / temperature).exp()
    pool_coef: float = float(kx * ky)
    den = pool_coef * F.avg_pool2d(x_exp, kernel_size, stride=stride, padding=padding) + eps
    x_softmaxpool = pool_coef * F.avg_pool2d(x_exp * input, kernel_size, stride=stride, padding=padding) / den
    x_softmaxpool = x_softmaxpool.view(b, c, x_softmaxpool.size(2), x_softmaxpool.size(3))
    grid_global: Tensor = create_meshgrid(h, w, False, device).to(dtype).permute(0, 3, 1, 2)
    grid_global_pooled = F.conv2d(grid_global, center_kernel, stride=stride, padding=padding)
    coords_max: Tensor = F.conv2d(x_exp, window_kernel, stride=stride, padding=padding)
    coords_max = coords_max / den.expand_as(coords_max)
    coords_max = coords_max + grid_global_pooled.expand_as(coords_max)
    if normalized_coordinates:
        coords_max = normalize_pixel_coordinates(coords_max.permute(0, 2, 3, 1), h, w)
        coords_max = coords_max.permute(0, 3, 1, 2)
    coords_max = coords_max.view(b, c, 2, coords_max.size(2), coords_max.size(3))
    if output_value:
        return (coords_max, x_softmaxpool)
    return coords_max

def conv_soft_argmax3d(input: Tensor, kernel_size: tuple[int, int, int]=(3, 3, 3), stride: tuple[int, int, int]=(1, 1, 1), padding: tuple[int, int, int]=(1, 1, 1), temperature: Tensor | float=tensor(1.0), normalized_coordinates: bool=False, eps: float=1e-08, output_value: bool=True, strict_maxima_bonus: float=0.0) -> Tensor | tuple[Tensor, Tensor]:
    if False:
        while True:
            i = 10
    'Compute the convolutional spatial Soft-Argmax 3D over the windows of a given heatmap.\n\n    .. math::\n             ijk(X) = \\frac{\\sum{(i,j,k)} * exp(x / T)  \\in X} {\\sum{exp(x / T)  \\in X}}\n\n    .. math::\n             val(X) = \\frac{\\sum{x * exp(x / T)  \\in X}} {\\sum{exp(x / T)  \\in X}}\n\n    where ``T`` is temperature.\n\n    Args:\n        input: the given heatmap with shape :math:`(N, C, D_{in}, H_{in}, W_{in})`.\n        kernel_size:  size of the window.\n        stride: stride of the window.\n        padding: input zero padding.\n        temperature: factor to apply to input.\n        normalized_coordinates: whether to return the coordinates normalized in the range of :math:[-1, 1]`.\n            Otherwise, it will return the coordinates in the range of the input shape.\n        eps: small value to avoid zero division.\n        output_value: if True, val is output, if False, only ij.\n        strict_maxima_bonus: pixels, which are strict maxima will score (1 + strict_maxima_bonus) * value.\n          This is needed for mimic behavior of strict NMS in classic local features\n\n    Returns:\n        Function has two outputs - argmax coordinates and the softmaxpooled heatmap values themselves.\n        On each window, the function computed returns with shapes :math:`(N, C, 3, D_{out}, H_{out}, W_{out})`,\n        :math:`(N, C, D_{out}, H_{out}, W_{out})`,\n\n        where\n\n         .. math::\n             D_{out} = \\left\\lfloor\\frac{D_{in}  + 2 \\times \\text{padding}[0] -\n             (\\text{kernel\\_size}[0] - 1) - 1}{\\text{stride}[0]} + 1\\right\\rfloor\n\n         .. math::\n             H_{out} = \\left\\lfloor\\frac{H_{in}  + 2 \\times \\text{padding}[1] -\n             (\\text{kernel\\_size}[1] - 1) - 1}{\\text{stride}[1]} + 1\\right\\rfloor\n\n         .. math::\n             W_{out} = \\left\\lfloor\\frac{W_{in}  + 2 \\times \\text{padding}[2] -\n             (\\text{kernel\\_size}[2] - 1) - 1}{\\text{stride}[2]} + 1\\right\\rfloor\n\n    Examples:\n        >>> input = torch.randn(20, 16, 3, 50, 32)\n        >>> nms_coords, nms_val = conv_soft_argmax3d(input, (3, 3, 3), (1, 2, 2), (0, 1, 1))\n    '
    if not torch.is_tensor(input):
        raise TypeError(f'Input type is not a Tensor. Got {type(input)}')
    if not len(input.shape) == 5:
        raise ValueError(f'Invalid input shape, we expect BxCxDxHxW. Got: {input.shape}')
    if temperature <= 0:
        raise ValueError(f'Temperature should be positive float or tensor. Got: {temperature}')
    (b, c, d, h, w) = input.shape
    (kz, ky, kx) = kernel_size
    device: torch.device = input.device
    dtype: torch.dtype = input.dtype
    input = input.view(b * c, 1, d, h, w)
    center_kernel: Tensor = _get_center_kernel3d(kz, ky, kx, device).to(dtype)
    window_kernel: Tensor = _get_window_grid_kernel3d(kz, ky, kx, device).to(dtype)
    x_max = F.adaptive_max_pool3d(input, (1, 1, 1))
    x_exp = ((input - x_max.detach()) / temperature).exp()
    pool_coef: float = float(kx * ky * kz)
    den = pool_coef * F.avg_pool3d(x_exp.view_as(input), kernel_size, stride=stride, padding=padding) + eps
    grid_global: Tensor = create_meshgrid3d(d, h, w, False, device=device).to(dtype).permute(0, 4, 1, 2, 3)
    grid_global_pooled = F.conv3d(grid_global, center_kernel, stride=stride, padding=padding)
    coords_max: Tensor = F.conv3d(x_exp, window_kernel, stride=stride, padding=padding)
    coords_max = coords_max / den.expand_as(coords_max)
    coords_max = coords_max + grid_global_pooled.expand_as(coords_max)
    if normalized_coordinates:
        coords_max = normalize_pixel_coordinates3d(coords_max.permute(0, 2, 3, 4, 1), d, h, w)
        coords_max = coords_max.permute(0, 4, 1, 2, 3)
    coords_max = coords_max.view(b, c, 3, coords_max.size(2), coords_max.size(3), coords_max.size(4))
    if not output_value:
        return coords_max
    x_softmaxpool = pool_coef * F.avg_pool3d(x_exp.view(input.size()) * input, kernel_size, stride=stride, padding=padding) / den
    if strict_maxima_bonus > 0:
        in_levels: int = input.size(2)
        out_levels: int = x_softmaxpool.size(2)
        skip_levels: int = (in_levels - out_levels) // 2
        strict_maxima: Tensor = F.avg_pool3d(nms3d(input, kernel_size), 1, stride, 0)
        strict_maxima = strict_maxima[:, :, skip_levels:out_levels - skip_levels]
        x_softmaxpool *= 1.0 + strict_maxima_bonus * strict_maxima
    x_softmaxpool = x_softmaxpool.view(b, c, x_softmaxpool.size(2), x_softmaxpool.size(3), x_softmaxpool.size(4))
    return (coords_max, x_softmaxpool)

def spatial_soft_argmax2d(input: Tensor, temperature: Tensor=tensor(1.0), normalized_coordinates: bool=True) -> Tensor:
    if False:
        while True:
            i = 10
    'Compute the Spatial Soft-Argmax 2D of a given input heatmap.\n\n    Args:\n        input: the given heatmap with shape :math:`(B, N, H, W)`.\n        temperature: factor to apply to input.\n        normalized_coordinates: whether to return the coordinates normalized in the range of :math:`[-1, 1]`.\n            Otherwise, it will return the coordinates in the range of the input shape.\n\n    Returns:\n        the index of the maximum 2d coordinates of the give map :math:`(B, N, 2)`.\n        The output order is x-coord and y-coord.\n\n    Examples:\n        >>> input = torch.tensor([[[\n        ... [0., 0., 0.],\n        ... [0., 10., 0.],\n        ... [0., 0., 0.]]]])\n        >>> spatial_soft_argmax2d(input, normalized_coordinates=False)\n        tensor([[[1.0000, 1.0000]]])\n    '
    input_soft: Tensor = spatial_softmax2d(input, temperature)
    output: Tensor = spatial_expectation2d(input_soft, normalized_coordinates)
    return output

class SpatialSoftArgmax2d(Module):
    """Compute the Spatial Soft-Argmax 2D of a given heatmap.

    See :func:`~kornia.geometry.subpix.spatial_soft_argmax2d` for details.
    """

    def __init__(self, temperature: Tensor=tensor(1.0), normalized_coordinates: bool=True) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.temperature: Tensor = temperature
        self.normalized_coordinates: bool = normalized_coordinates

    def __repr__(self) -> str:
        if False:
            return 10
        return f'{self.__class__.__name__}temperature={self.temperature}, normalized_coordinates={self.normalized_coordinates})'

    def forward(self, input: Tensor) -> Tensor:
        if False:
            return 10
        return spatial_soft_argmax2d(input, self.temperature, self.normalized_coordinates)

def conv_quad_interp3d(input: Tensor, strict_maxima_bonus: float=10.0, eps: float=1e-07) -> tuple[Tensor, Tensor]:
    if False:
        while True:
            i = 10
    'Compute the single iteration of quadratic interpolation of the extremum (max or min).\n\n    Args:\n        input: the given heatmap with shape :math:`(N, C, D_{in}, H_{in}, W_{in})`.\n        strict_maxima_bonus: pixels, which are strict maxima will score (1 + strict_maxima_bonus) * value.\n          This is needed for mimic behavior of strict NMS in classic local features\n        eps: parameter to control the hessian matrix ill-condition number.\n\n    Returns:\n        the location and value per each 3x3x3 window which contains strict extremum, similar to one done is SIFT.\n        :math:`(N, C, 3, D_{out}, H_{out}, W_{out})`, :math:`(N, C, D_{out}, H_{out}, W_{out})`,\n\n        where\n\n         .. math::\n             D_{out} = \\left\\lfloor\\frac{D_{in}  + 2 \\times \\text{padding}[0] -\n             (\\text{kernel\\_size}[0] - 1) - 1}{\\text{stride}[0]} + 1\\right\\rfloor\n\n         .. math::\n             H_{out} = \\left\\lfloor\\frac{H_{in}  + 2 \\times \\text{padding}[1] -\n             (\\text{kernel\\_size}[1] - 1) - 1}{\\text{stride}[1]} + 1\\right\\rfloor\n\n         .. math::\n             W_{out} = \\left\\lfloor\\frac{W_{in}  + 2 \\times \\text{padding}[2] -\n             (\\text{kernel\\_size}[2] - 1) - 1}{\\text{stride}[2]} + 1\\right\\rfloor\n\n    Examples:\n        >>> input = torch.randn(20, 16, 3, 50, 32)\n        >>> nms_coords, nms_val = conv_quad_interp3d(input, 1.0)\n    '
    if not torch.is_tensor(input):
        raise TypeError(f'Input type is not a Tensor. Got {type(input)}')
    if not len(input.shape) == 5:
        raise ValueError(f'Invalid input shape, we expect BxCxDxHxW. Got: {input.shape}')
    (B, CH, D, H, W) = input.shape
    grid_global: Tensor = create_meshgrid3d(D, H, W, False, device=input.device).permute(0, 4, 1, 2, 3)
    grid_global = grid_global.to(input.dtype)
    b: Tensor = spatial_gradient3d(input, order=1, mode='diff')
    b = b.permute(0, 1, 3, 4, 5, 2).reshape(-1, 3, 1)
    A: Tensor = spatial_gradient3d(input, order=2, mode='diff')
    A = A.permute(0, 1, 3, 4, 5, 2).reshape(-1, 6)
    dxx = A[..., 0]
    dyy = A[..., 1]
    dss = A[..., 2]
    dxy = 0.25 * A[..., 3]
    dys = 0.25 * A[..., 4]
    dxs = 0.25 * A[..., 5]
    Hes = stack([dxx, dxy, dxs, dxy, dyy, dys, dxs, dys, dss], -1).view(-1, 3, 3)
    if not torch_version_ge(1, 10):
        Hes += torch.rand(Hes[0].size(), device=Hes.device).abs()[None] * eps
    nms_mask: Tensor = nms3d(input, (3, 3, 3), True)
    x_solved: Tensor = torch.zeros_like(b)
    (x_solved_masked, _, solved_correctly) = safe_solve_with_mask(b[nms_mask.view(-1)], Hes[nms_mask.view(-1)])
    new_nms_mask = nms_mask.masked_scatter(nms_mask, solved_correctly)
    x_solved[where(new_nms_mask.view(-1, 1, 1))[0]] = x_solved_masked[solved_correctly]
    dx: Tensor = -x_solved
    mask1 = dx.abs().max(dim=1, keepdim=True)[0] > 0.7
    dx.masked_fill_(mask1.expand_as(dx), 0)
    dy: Tensor = 0.5 * torch.bmm(b.permute(0, 2, 1), dx)
    y_max = input + dy.view(B, CH, D, H, W)
    if strict_maxima_bonus > 0:
        y_max += strict_maxima_bonus * new_nms_mask.to(input.dtype)
    dx_res: Tensor = dx.flip(1).reshape(B, CH, D, H, W, 3).permute(0, 1, 5, 2, 3, 4)
    dx_res[:, :, (1, 2)] = dx_res[:, :, (2, 1)]
    coords_max: Tensor = grid_global.repeat(B, 1, 1, 1, 1).unsqueeze(1)
    coords_max = coords_max + dx_res
    return (coords_max, y_max)

class ConvQuadInterp3d(Module):
    """Calculate soft argmax 3d per window.

    See
    :func: `~kornia.geometry.subpix.conv_quad_interp3d` for details.
    """

    def __init__(self, strict_maxima_bonus: float=10.0, eps: float=1e-07) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.strict_maxima_bonus = strict_maxima_bonus
        self.eps = eps

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return f'{self.__class__.__name__}(strict_maxima_bonus={self.strict_maxima_bonus})'

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        if False:
            i = 10
            return i + 15
        return conv_quad_interp3d(x, self.strict_maxima_bonus, self.eps)