"""
Adapted from "Contrast Limited Adaptive Histogram Equalization" by Karel
Zuiderveld, Graphics Gems IV, Academic Press, 1994.

http://tog.acm.org/resources/GraphicsGems/

Relicensed with permission of the author under the Modified BSD license.
"""
import math
import numbers
import numpy as np
from .._shared.utils import _supported_float_type
from ..color.adapt_rgb import adapt_rgb, hsv_value
from .exposure import rescale_intensity
from ..util import img_as_uint
NR_OF_GRAY = 2 ** 14

@adapt_rgb(hsv_value)
def equalize_adapthist(image, kernel_size=None, clip_limit=0.01, nbins=256):
    if False:
        return 10
    'Contrast Limited Adaptive Histogram Equalization (CLAHE).\n\n    An algorithm for local contrast enhancement, that uses histograms computed\n    over different tile regions of the image. Local details can therefore be\n    enhanced even in regions that are darker or lighter than most of the image.\n\n    Parameters\n    ----------\n    image : (M[, ...][, C]) ndarray\n        Input image.\n    kernel_size : int or array_like, optional\n        Defines the shape of contextual regions used in the algorithm. If\n        iterable is passed, it must have the same number of elements as\n        ``image.ndim`` (without color channel). If integer, it is broadcasted\n        to each `image` dimension. By default, ``kernel_size`` is 1/8 of\n        ``image`` height by 1/8 of its width.\n    clip_limit : float, optional\n        Clipping limit, normalized between 0 and 1 (higher values give more\n        contrast).\n    nbins : int, optional\n        Number of gray bins for histogram ("data range").\n\n    Returns\n    -------\n    out : (M[, ...][, C]) ndarray\n        Equalized image with float64 dtype.\n\n    See Also\n    --------\n    equalize_hist, rescale_intensity\n\n    Notes\n    -----\n    * For color images, the following steps are performed:\n       - The image is converted to HSV color space\n       - The CLAHE algorithm is run on the V (Value) channel\n       - The image is converted back to RGB space and returned\n    * For RGBA images, the original alpha channel is removed.\n\n    .. versionchanged:: 0.17\n        The values returned by this function are slightly shifted upwards\n        because of an internal change in rounding behavior.\n\n    References\n    ----------\n    .. [1] http://tog.acm.org/resources/GraphicsGems/\n    .. [2] https://en.wikipedia.org/wiki/CLAHE#CLAHE\n    '
    float_dtype = _supported_float_type(image.dtype)
    image = img_as_uint(image)
    image = np.round(rescale_intensity(image, out_range=(0, NR_OF_GRAY - 1))).astype(np.min_scalar_type(NR_OF_GRAY))
    if kernel_size is None:
        kernel_size = tuple([max(s // 8, 1) for s in image.shape])
    elif isinstance(kernel_size, numbers.Number):
        kernel_size = (kernel_size,) * image.ndim
    elif len(kernel_size) != image.ndim:
        raise ValueError(f'Incorrect value of `kernel_size`: {kernel_size}')
    kernel_size = [int(k) for k in kernel_size]
    image = _clahe(image, kernel_size, clip_limit, nbins)
    image = image.astype(float_dtype, copy=False)
    return rescale_intensity(image)

def _clahe(image, kernel_size, clip_limit, nbins):
    if False:
        print('Hello World!')
    'Contrast Limited Adaptive Histogram Equalization.\n\n    Parameters\n    ----------\n    image : (M[, ...]) ndarray\n        Input image.\n    kernel_size : int or N-tuple of int\n        Defines the shape of contextual regions used in the algorithm.\n    clip_limit : float\n        Normalized clipping limit between 0 and 1 (higher values give more\n        contrast).\n    nbins : int\n        Number of gray bins for histogram ("data range").\n\n    Returns\n    -------\n    out : (M[, ...]) ndarray\n        Equalized image.\n\n    The number of "effective" graylevels in the output image is set by `nbins`;\n    selecting a small value (e.g. 128) speeds up processing and still produces\n    an output image of good quality. A clip limit of 0 or larger than or equal\n    to 1 results in standard (non-contrast limited) AHE.\n    '
    ndim = image.ndim
    dtype = image.dtype
    pad_start_per_dim = [k // 2 for k in kernel_size]
    pad_end_per_dim = [(k - s % k) % k + int(np.ceil(k / 2.0)) for (k, s) in zip(kernel_size, image.shape)]
    image = np.pad(image, [[p_i, p_f] for (p_i, p_f) in zip(pad_start_per_dim, pad_end_per_dim)], mode='reflect')
    bin_size = 1 + NR_OF_GRAY // nbins
    lut = np.arange(NR_OF_GRAY, dtype=np.min_scalar_type(NR_OF_GRAY))
    lut //= bin_size
    image = lut[image]
    ns_hist = [int(s / k) - 1 for (s, k) in zip(image.shape, kernel_size)]
    hist_blocks_shape = np.array([ns_hist, kernel_size]).T.flatten()
    hist_blocks_axis_order = np.array([np.arange(0, ndim * 2, 2), np.arange(1, ndim * 2, 2)]).flatten()
    hist_slices = [slice(k // 2, k // 2 + n * k) for (k, n) in zip(kernel_size, ns_hist)]
    hist_blocks = image[tuple(hist_slices)].reshape(hist_blocks_shape)
    hist_blocks = np.transpose(hist_blocks, axes=hist_blocks_axis_order)
    hist_block_assembled_shape = hist_blocks.shape
    hist_blocks = hist_blocks.reshape((math.prod(ns_hist), -1))
    kernel_elements = math.prod(kernel_size)
    if clip_limit > 0.0:
        clim = int(np.clip(clip_limit * kernel_elements, 1, None))
    else:
        clim = kernel_elements
    hist = np.apply_along_axis(np.bincount, -1, hist_blocks, minlength=nbins)
    hist = np.apply_along_axis(clip_histogram, -1, hist, clip_limit=clim)
    hist = map_histogram(hist, 0, NR_OF_GRAY - 1, kernel_elements)
    hist = hist.reshape(hist_block_assembled_shape[:ndim] + (-1,))
    map_array = np.pad(hist, [[1, 1] for _ in range(ndim)] + [[0, 0]], mode='edge')
    ns_proc = [int(s / k) for (s, k) in zip(image.shape, kernel_size)]
    blocks_shape = np.array([ns_proc, kernel_size]).T.flatten()
    blocks_axis_order = np.array([np.arange(0, ndim * 2, 2), np.arange(1, ndim * 2, 2)]).flatten()
    blocks = image.reshape(blocks_shape)
    blocks = np.transpose(blocks, axes=blocks_axis_order)
    blocks_flattened_shape = blocks.shape
    blocks = np.reshape(blocks, (math.prod(ns_proc), math.prod(blocks.shape[ndim:])))
    coeffs = np.meshgrid(*tuple([np.arange(k) / k for k in kernel_size[::-1]]), indexing='ij')
    coeffs = [np.transpose(c).flatten() for c in coeffs]
    inv_coeffs = [1 - c for (dim, c) in enumerate(coeffs)]
    result = np.zeros(blocks.shape, dtype=np.float32)
    for (iedge, edge) in enumerate(np.ndindex(*[2] * ndim)):
        edge_maps = map_array[tuple([slice(e, e + n) for (e, n) in zip(edge, ns_proc)])]
        edge_maps = edge_maps.reshape((math.prod(ns_proc), -1))
        edge_mapped = np.take_along_axis(edge_maps, blocks, axis=-1)
        edge_coeffs = np.prod([[inv_coeffs, coeffs][e][d] for (d, e) in enumerate(edge[::-1])], 0)
        result += (edge_mapped * edge_coeffs).astype(result.dtype)
    result = result.astype(dtype)
    result = result.reshape(blocks_flattened_shape)
    blocks_axis_rebuild_order = np.array([np.arange(0, ndim), np.arange(ndim, ndim * 2)]).T.flatten()
    result = np.transpose(result, axes=blocks_axis_rebuild_order)
    result = result.reshape(image.shape)
    unpad_slices = tuple([slice(p_i, s - p_f) for (p_i, p_f, s) in zip(pad_start_per_dim, pad_end_per_dim, image.shape)])
    result = result[unpad_slices]
    return result

def clip_histogram(hist, clip_limit):
    if False:
        while True:
            i = 10
    'Perform clipping of the histogram and redistribution of bins.\n\n    The histogram is clipped and the number of excess pixels is counted.\n    Afterwards the excess pixels are equally redistributed across the\n    whole histogram (providing the bin count is smaller than the cliplimit).\n\n    Parameters\n    ----------\n    hist : ndarray\n        Histogram array.\n    clip_limit : int\n        Maximum allowed bin count.\n\n    Returns\n    -------\n    hist : ndarray\n        Clipped histogram.\n    '
    excess_mask = hist > clip_limit
    excess = hist[excess_mask]
    n_excess = excess.sum() - excess.size * clip_limit
    hist[excess_mask] = clip_limit
    bin_incr = n_excess // hist.size
    upper = clip_limit - bin_incr
    low_mask = hist < upper
    n_excess -= hist[low_mask].size * bin_incr
    hist[low_mask] += bin_incr
    mid_mask = np.logical_and(hist >= upper, hist < clip_limit)
    mid = hist[mid_mask]
    n_excess += mid.sum() - mid.size * clip_limit
    hist[mid_mask] = clip_limit
    while n_excess > 0:
        prev_n_excess = n_excess
        for index in range(hist.size):
            under_mask = hist < clip_limit
            step_size = max(1, np.count_nonzero(under_mask) // n_excess)
            under_mask = under_mask[index::step_size]
            hist[index::step_size][under_mask] += 1
            n_excess -= np.count_nonzero(under_mask)
            if n_excess <= 0:
                break
        if prev_n_excess == n_excess:
            break
    return hist

def map_histogram(hist, min_val, max_val, n_pixels):
    if False:
        return 10
    'Calculate the equalized lookup table (mapping).\n\n    It does so by cumulating the input histogram.\n    Histogram bins are assumed to be represented by the last array dimension.\n\n    Parameters\n    ----------\n    hist : ndarray\n        Clipped histogram.\n    min_val : int\n        Minimum value for mapping.\n    max_val : int\n        Maximum value for mapping.\n    n_pixels : int\n        Number of pixels in the region.\n\n    Returns\n    -------\n    out : ndarray\n       Mapped intensity LUT.\n    '
    out = np.cumsum(hist, axis=-1).astype(float)
    out *= (max_val - min_val) / n_pixels
    out += min_val
    np.clip(out, a_min=None, a_max=max_val, out=out)
    return out.astype(int)