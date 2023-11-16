"""In this module several equalization methods are exposed: he, ahe, clahe."""
import math
from typing import Tuple
import torch
import torch.nn.functional as F
from kornia.utils.helpers import _torch_histc_cast
from kornia.utils.image import perform_keep_shape_image
from .histogram import histogram

def _compute_tiles(imgs: torch.Tensor, grid_size: Tuple[int, int], even_tile_size: bool=False) -> Tuple[torch.Tensor, torch.Tensor]:
    if False:
        print('Hello World!')
    "Compute tiles on an image according to a grid size.\n\n    Note that padding can be added to the image in order to crop properly the image.\n    So, the grid_size (GH, GW) x tile_size (TH, TW) >= image_size (H, W)\n\n    Args:\n        imgs: batch of 2D images with shape (B, C, H, W) or (C, H, W).\n        grid_size: number of tiles to be cropped in each direction (GH, GW)\n        even_tile_size: Determine if the width and height of the tiles must be even.\n\n    Returns:\n        tensor with tiles (B, GH, GW, C, TH, TW). B = 1 in case of a single image is provided.\n        tensor with the padded batch of 2D imageswith shape (B, C, H', W').\n    "
    batch: torch.Tensor = imgs
    (h, w) = batch.shape[-2:]
    kernel_vert: int = math.ceil(h / grid_size[0])
    kernel_horz: int = math.ceil(w / grid_size[1])
    if even_tile_size:
        kernel_vert += 1 if kernel_vert % 2 else 0
        kernel_horz += 1 if kernel_horz % 2 else 0
    pad_vert = kernel_vert * grid_size[0] - h
    pad_horz = kernel_horz * grid_size[1] - w
    if pad_vert > batch.shape[-2] or pad_horz > batch.shape[-1]:
        raise ValueError('Cannot compute tiles on the image according to the given grid size')
    if pad_vert > 0 or pad_horz > 0:
        batch = F.pad(batch, [0, pad_horz, 0, pad_vert], mode='reflect')
    c: int = batch.shape[-3]
    tiles: torch.Tensor = batch.unfold(1, c, c).unfold(2, kernel_vert, kernel_vert).unfold(3, kernel_horz, kernel_horz).squeeze(1).contiguous()
    if tiles.shape[-5] != grid_size[0]:
        raise AssertionError
    if tiles.shape[-4] != grid_size[1]:
        raise AssertionError
    return (tiles, batch)

def _compute_interpolation_tiles(padded_imgs: torch.Tensor, tile_size: Tuple[int, int]) -> torch.Tensor:
    if False:
        return 10
    'Compute interpolation tiles on a properly padded set of images.\n\n    Note that images must be padded. So, the tile_size (TH, TW) * grid_size (GH, GW) = image_size (H, W)\n\n    Args:\n        padded_imgs: batch of 2D images with shape (B, C, H, W) already padded to extract tiles\n                                    of size (TH, TW).\n        tile_size: shape of the current tiles (TH, TW).\n\n    Returns:\n        tensor with the interpolation tiles (B, 2GH, 2GW, C, TH/2, TW/2).\n    '
    if padded_imgs.dim() != 4:
        raise AssertionError('Images Tensor must be 4D.')
    if padded_imgs.shape[-2] % tile_size[0] != 0:
        raise AssertionError('Images are not correctly padded.')
    if padded_imgs.shape[-1] % tile_size[1] != 0:
        raise AssertionError('Images are not correctly padded.')
    interp_kernel_vert: int = tile_size[0] // 2
    interp_kernel_horz: int = tile_size[1] // 2
    c: int = padded_imgs.shape[-3]
    interp_tiles: torch.Tensor = padded_imgs.unfold(1, c, c).unfold(2, interp_kernel_vert, interp_kernel_vert).unfold(3, interp_kernel_horz, interp_kernel_horz).squeeze(1).contiguous()
    if interp_tiles.shape[-3] != c:
        raise AssertionError
    if interp_tiles.shape[-2] != tile_size[0] / 2:
        raise AssertionError
    if interp_tiles.shape[-1] != tile_size[1] / 2:
        raise AssertionError
    return interp_tiles

def _my_histc(tiles: torch.Tensor, bins: int) -> torch.Tensor:
    if False:
        i = 10
        return i + 15
    return _torch_histc_cast(tiles, bins=bins, min=0, max=1)

def _compute_luts(tiles_x_im: torch.Tensor, num_bins: int=256, clip: float=40.0, diff: bool=False) -> torch.Tensor:
    if False:
        i = 10
        return i + 15
    'Compute luts for a batched set of tiles.\n\n    Same approach as in OpenCV (https://github.com/opencv/opencv/blob/master/modules/imgproc/src/clahe.cpp)\n\n    Args:\n        tiles_x_im: set of tiles per image to apply the lut. (B, GH, GW, C, TH, TW)\n        num_bins: number of bins. default: 256\n        clip: threshold value for contrast limiting. If it is 0 then the clipping is disabled.\n        diff: denote if the differentiable histagram will be used. Default: False\n\n    Returns:\n        Lut for each tile (B, GH, GW, C, 256).\n    '
    if tiles_x_im.dim() != 6:
        raise AssertionError('Tensor must be 6D.')
    (b, gh, gw, c, th, tw) = tiles_x_im.shape
    pixels: int = th * tw
    tiles: torch.Tensor = tiles_x_im.view(-1, pixels)
    if not diff:
        if torch.jit.is_scripting():
            histos = torch.stack([_torch_histc_cast(tile, bins=num_bins, min=0, max=1) for tile in tiles])
        else:
            histos = torch.stack(list(map(_my_histc, tiles, [num_bins] * len(tiles))))
    else:
        bins: torch.Tensor = torch.linspace(0, 1, num_bins, device=tiles.device)
        histos = histogram(tiles, bins, torch.tensor(0.001)).squeeze()
        histos *= pixels
    if clip > 0.0:
        max_val: float = max(clip * pixels // num_bins, 1)
        histos.clamp_(max=max_val)
        clipped: torch.Tensor = pixels - histos.sum(1)
        residual: torch.Tensor = torch.remainder(clipped, num_bins)
        redist: torch.Tensor = (clipped - residual).div(num_bins)
        histos += redist[None].transpose(0, 1)
        v_range: torch.Tensor = torch.arange(num_bins, device=histos.device)
        mat_range: torch.Tensor = v_range.repeat(histos.shape[0], 1)
        histos += mat_range < residual[None].transpose(0, 1)
    lut_scale: float = (num_bins - 1) / pixels
    luts: torch.Tensor = torch.cumsum(histos, 1) * lut_scale
    luts = luts.clamp(0, num_bins - 1)
    if not diff:
        luts = luts.floor()
    luts = luts.view((b, gh, gw, c, num_bins))
    return luts

def _map_luts(interp_tiles: torch.Tensor, luts: torch.Tensor) -> torch.Tensor:
    if False:
        return 10
    'Assign the required luts to each tile.\n\n    Args:\n        interp_tiles: set of interpolation tiles. (B, 2GH, 2GW, C, TH/2, TW/2)\n        luts: luts for each one of the original tiles. (B, GH, GW, C, 256)\n\n    Returns:\n         mapped luts (B, 2GH, 2GW, 4, C, 256)\n    '
    if interp_tiles.dim() != 6:
        raise AssertionError('interp_tiles tensor must be 6D.')
    if luts.dim() != 5:
        raise AssertionError('luts tensor must be 5D.')
    (num_imgs, gh, gw, c, _, _) = interp_tiles.shape
    j_idxs = torch.empty(0, 4, dtype=torch.long)
    if gh > 2:
        j_floor = torch.arange(1, gh - 1).view(gh - 2, 1).div(2, rounding_mode='trunc')
        j_idxs = torch.tensor([[0, 0, 1, 1], [-1, -1, 0, 0]] * ((gh - 2) // 2))
        j_idxs += j_floor
    i_idxs = torch.empty(0, 4, dtype=torch.long)
    if gw > 2:
        i_floor = torch.arange(1, gw - 1).view(gw - 2, 1).div(2, rounding_mode='trunc')
        i_idxs = torch.tensor([[0, 1, 0, 1], [-1, 0, -1, 0]] * ((gw - 2) // 2))
        i_idxs += i_floor
    luts_x_interp_tiles: torch.Tensor = torch.full((num_imgs, gh, gw, 4, c, luts.shape[-1]), -1, dtype=interp_tiles.dtype, device=interp_tiles.device)
    luts_x_interp_tiles[:, 0::gh - 1, 0::gw - 1, 0] = luts[:, 0::max(gh // 2 - 1, 1), 0::max(gw // 2 - 1, 1)]
    luts_x_interp_tiles[:, 1:-1, 0::gw - 1, 0] = luts[:, j_idxs[:, 0], 0::max(gw // 2 - 1, 1)]
    luts_x_interp_tiles[:, 1:-1, 0::gw - 1, 1] = luts[:, j_idxs[:, 2], 0::max(gw // 2 - 1, 1)]
    luts_x_interp_tiles[:, 0::gh - 1, 1:-1, 0] = luts[:, 0::max(gh // 2 - 1, 1), i_idxs[:, 0]]
    luts_x_interp_tiles[:, 0::gh - 1, 1:-1, 1] = luts[:, 0::max(gh // 2 - 1, 1), i_idxs[:, 1]]
    luts_x_interp_tiles[:, 1:-1, 1:-1, :] = luts[:, j_idxs.repeat(max(gh - 2, 1), 1, 1).permute(1, 0, 2), i_idxs.repeat(max(gw - 2, 1), 1, 1)]
    return luts_x_interp_tiles

def _compute_equalized_tiles(interp_tiles: torch.Tensor, luts: torch.Tensor) -> torch.Tensor:
    if False:
        i = 10
        return i + 15
    'Equalize the tiles.\n\n    Args:\n        interp_tiles: set of interpolation tiles, values must be in the range [0, 1].\n          (B, 2GH, 2GW, C, TH/2, TW/2)\n        luts: luts for each one of the original tiles. (B, GH, GW, C, 256)\n\n    Returns:\n        equalized tiles (B, 2GH, 2GW, C, TH/2, TW/2)\n    '
    if interp_tiles.dim() != 6:
        raise AssertionError('interp_tiles tensor must be 6D.')
    if luts.dim() != 5:
        raise AssertionError('luts tensor must be 5D.')
    mapped_luts: torch.Tensor = _map_luts(interp_tiles, luts)
    (num_imgs, gh, gw, c, th, tw) = interp_tiles.shape
    flatten_interp_tiles: torch.Tensor = (interp_tiles * 255).long().flatten(-2, -1)
    flatten_interp_tiles = flatten_interp_tiles.unsqueeze(-3).expand(num_imgs, gh, gw, 4, c, th * tw)
    preinterp_tiles_equalized = torch.gather(mapped_luts, 5, flatten_interp_tiles).to(interp_tiles).reshape(num_imgs, gh, gw, 4, c, th, tw)
    tiles_equalized: torch.Tensor = torch.zeros_like(interp_tiles)
    ih = torch.arange(2 * th - 1, -1, -1, dtype=interp_tiles.dtype, device=interp_tiles.device).div(2.0 * th - 1)[None].transpose(-2, -1).expand(2 * th, tw)
    ih = ih.unfold(0, th, th).unfold(1, tw, tw)
    iw = torch.arange(2 * tw - 1, -1, -1, dtype=interp_tiles.dtype, device=interp_tiles.device).div(2.0 * tw - 1).expand(th, 2 * tw)
    iw = iw.unfold(0, th, th).unfold(1, tw, tw)
    tiw = iw.expand((gw - 2) // 2, 2, th, tw).reshape(gw - 2, 1, th, tw).unsqueeze(0)
    tih = ih.repeat((gh - 2) // 2, 1, 1, 1).unsqueeze(1)
    (tl, tr, bl, br) = preinterp_tiles_equalized[:, 1:-1, 1:-1].unbind(3)
    t = torch.addcmul(tr, tiw, torch.sub(tl, tr))
    b = torch.addcmul(br, tiw, torch.sub(bl, br))
    tiles_equalized[:, 1:-1, 1:-1] = torch.addcmul(b, tih, torch.sub(t, b))
    tiles_equalized[:, 0::gh - 1, 0::gw - 1] = preinterp_tiles_equalized[:, 0::gh - 1, 0::gw - 1, 0]
    (t, b, _, _) = preinterp_tiles_equalized[:, 1:-1, 0].unbind(2)
    tiles_equalized[:, 1:-1, 0] = torch.addcmul(b, tih.squeeze(1), torch.sub(t, b))
    (t, b, _, _) = preinterp_tiles_equalized[:, 1:-1, gh - 1].unbind(2)
    tiles_equalized[:, 1:-1, gh - 1] = torch.addcmul(b, tih.squeeze(1), torch.sub(t, b))
    (left, right, _, _) = preinterp_tiles_equalized[:, 0, 1:-1].unbind(2)
    tiles_equalized[:, 0, 1:-1] = torch.addcmul(right, tiw, torch.sub(left, right))
    (left, right, _, _) = preinterp_tiles_equalized[:, gw - 1, 1:-1].unbind(2)
    tiles_equalized[:, gw - 1, 1:-1] = torch.addcmul(right, tiw, torch.sub(left, right))
    return tiles_equalized.div(255.0)

@perform_keep_shape_image
def equalize_clahe(input: torch.Tensor, clip_limit: float=40.0, grid_size: Tuple[int, int]=(8, 8), slow_and_differentiable: bool=False) -> torch.Tensor:
    if False:
        i = 10
        return i + 15
    'Apply clahe equalization on the input tensor.\n\n    .. image:: _static/img/equalize_clahe.png\n\n    NOTE: Lut computation uses the same approach as in OpenCV, in next versions this can change.\n\n    Args:\n        input: images tensor to equalize with values in the range [0, 1] and shape :math:`(*, C, H, W)`.\n        clip_limit: threshold value for contrast limiting. If 0 clipping is disabled.\n        grid_size: number of tiles to be cropped in each direction (GH, GW).\n        slow_and_differentiable: flag to select implementation\n\n    Returns:\n        Equalized image or images with shape as the input.\n\n    Examples:\n        >>> img = torch.rand(1, 10, 20)\n        >>> res = equalize_clahe(img)\n        >>> res.shape\n        torch.Size([1, 10, 20])\n\n        >>> img = torch.rand(2, 3, 10, 20)\n        >>> res = equalize_clahe(img)\n        >>> res.shape\n        torch.Size([2, 3, 10, 20])\n    '
    if not isinstance(clip_limit, float):
        raise TypeError(f'Input clip_limit type is not float. Got {type(clip_limit)}')
    if not isinstance(grid_size, tuple):
        raise TypeError(f'Input grid_size type is not Tuple. Got {type(grid_size)}')
    if len(grid_size) != 2:
        raise TypeError(f'Input grid_size is not a Tuple with 2 elements. Got {len(grid_size)}')
    if isinstance(grid_size[0], float) or isinstance(grid_size[1], float):
        raise TypeError('Input grid_size type is not valid, must be a Tuple[int, int].')
    if grid_size[0] <= 0 or grid_size[1] <= 0:
        raise ValueError(f'Input grid_size elements must be positive. Got {grid_size}')
    imgs: torch.Tensor = input
    (hist_tiles, img_padded) = _compute_tiles(imgs, grid_size, True)
    tile_size: Tuple[int, int] = (hist_tiles.shape[-2], hist_tiles.shape[-1])
    interp_tiles: torch.Tensor = _compute_interpolation_tiles(img_padded, tile_size)
    luts: torch.Tensor = _compute_luts(hist_tiles, clip=clip_limit, diff=slow_and_differentiable)
    equalized_tiles: torch.Tensor = _compute_equalized_tiles(interp_tiles, luts)
    eq_imgs: torch.Tensor = equalized_tiles.permute(0, 3, 1, 4, 2, 5).reshape_as(img_padded)
    (h, w) = imgs.shape[-2:]
    eq_imgs = eq_imgs[..., :h, :w]
    if input.dim() != eq_imgs.dim():
        eq_imgs = eq_imgs.squeeze(0)
    return eq_imgs