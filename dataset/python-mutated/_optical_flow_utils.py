"""Common tools to optical flow algorithms.

"""
import numpy as np
from scipy import ndimage as ndi
from ..transform import pyramid_reduce
from ..util.dtype import _convert

def get_warp_points(grid, flow):
    if False:
        print('Hello World!')
    'Compute warp point coordinates.\n\n    Parameters\n    ----------\n    grid : iterable\n        The sparse grid to be warped (obtained using\n        ``np.meshgrid(..., sparse=True)).``)\n    flow : ndarray\n        The warping motion field.\n\n    Returns\n    -------\n    out : ndarray\n        The warp point coordinates.\n\n    '
    out = flow.copy()
    for (idx, g) in enumerate(grid):
        out[idx, ...] += g
    return out

def resize_flow(flow, shape):
    if False:
        i = 10
        return i + 15
    'Rescale the values of the vector field (u, v) to the desired shape.\n\n    The values of the output vector field are scaled to the new\n    resolution.\n\n    Parameters\n    ----------\n    flow : ndarray\n        The motion field to be processed.\n    shape : iterable\n        Couple of integers representing the output shape.\n\n    Returns\n    -------\n    rflow : ndarray\n        The resized and rescaled motion field.\n\n    '
    scale = [n / o for (n, o) in zip(shape, flow.shape[1:])]
    scale_factor = np.array(scale, dtype=flow.dtype)
    for _ in shape:
        scale_factor = scale_factor[..., np.newaxis]
    rflow = scale_factor * ndi.zoom(flow, [1] + scale, order=0, mode='nearest', prefilter=False)
    return rflow

def get_pyramid(I, downscale=2.0, nlevel=10, min_size=16):
    if False:
        i = 10
        return i + 15
    'Construct image pyramid.\n\n    Parameters\n    ----------\n    I : ndarray\n        The image to be preprocessed (Gray scale or RGB).\n    downscale : float\n        The pyramid downscale factor.\n    nlevel : int\n        The maximum number of pyramid levels.\n    min_size : int\n        The minimum size for any dimension of the pyramid levels.\n\n    Returns\n    -------\n    pyramid : list[ndarray]\n        The coarse to fine images pyramid.\n\n    '
    pyramid = [I]
    size = min(I.shape)
    count = 1
    while count < nlevel and size > downscale * min_size:
        J = pyramid_reduce(pyramid[-1], downscale, channel_axis=None)
        pyramid.append(J)
        size = min(J.shape)
        count += 1
    return pyramid[::-1]

def coarse_to_fine(I0, I1, solver, downscale=2, nlevel=10, min_size=16, dtype=np.float32):
    if False:
        return 10
    'Generic coarse to fine solver.\n\n    Parameters\n    ----------\n    I0 : ndarray\n        The first gray scale image of the sequence.\n    I1 : ndarray\n        The second gray scale image of the sequence.\n    solver : callable\n        The solver applied at each pyramid level.\n    downscale : float\n        The pyramid downscale factor.\n    nlevel : int\n        The maximum number of pyramid levels.\n    min_size : int\n        The minimum size for any dimension of the pyramid levels.\n    dtype : dtype\n        Output data type.\n\n    Returns\n    -------\n    flow : ndarray\n        The estimated optical flow components for each axis.\n\n    '
    if I0.shape != I1.shape:
        raise ValueError('Input images should have the same shape')
    if np.dtype(dtype).char not in 'efdg':
        raise ValueError('Only floating point data type are valid for optical flow')
    pyramid = list(zip(get_pyramid(_convert(I0, dtype), downscale, nlevel, min_size), get_pyramid(_convert(I1, dtype), downscale, nlevel, min_size)))
    flow = np.zeros((pyramid[0][0].ndim,) + pyramid[0][0].shape, dtype=dtype)
    flow = solver(pyramid[0][0], pyramid[0][1], flow)
    for (J0, J1) in pyramid[1:]:
        flow = solver(J0, J1, resize_flow(flow, J0.shape))
    return flow