"""
canny.py - Canny Edge detector

Reference: Canny, J., A Computational Approach To Edge Detection, IEEE Trans.
    Pattern Analysis and Machine Intelligence, 8:679-714, 1986
"""
import numpy as np
import scipy.ndimage as ndi
from ..util.dtype import dtype_limits
from .._shared.filters import gaussian
from .._shared.utils import _supported_float_type, check_nD
from ._canny_cy import _nonmaximum_suppression_bilinear

def _preprocess(image, mask, sigma, mode, cval):
    if False:
        for i in range(10):
            print('nop')
    "Generate a smoothed image and an eroded mask.\n\n    The image is smoothed using a gaussian filter ignoring masked\n    pixels and the mask is eroded.\n\n    Parameters\n    ----------\n    image : array\n        Image to be smoothed.\n    mask : array\n        Mask with 1's for significant pixels, 0's for masked pixels.\n    sigma : scalar or sequence of scalars\n        Standard deviation for Gaussian kernel. The standard\n        deviations of the Gaussian filter are given for each axis as a\n        sequence, or as a single number, in which case it is equal for\n        all axes.\n    mode : str, {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}\n        The ``mode`` parameter determines how the array borders are\n        handled, where ``cval`` is the value when mode is equal to\n        'constant'.\n    cval : float, optional\n        Value to fill past edges of input if `mode` is 'constant'.\n\n    Returns\n    -------\n    smoothed_image : ndarray\n        The smoothed array\n    eroded_mask : ndarray\n        The eroded mask.\n\n    Notes\n    -----\n    This function calculates the fractional contribution of masked pixels\n    by applying the function to the mask (which gets you the fraction of\n    the pixel data that's due to significant points). We then mask the image\n    and apply the function. The resulting values will be lower by the\n    bleed-over fraction, so you can recalibrate by dividing by the function\n    on the mask to recover the effect of smoothing from just the significant\n    pixels.\n    "
    gaussian_kwargs = dict(sigma=sigma, mode=mode, cval=cval, preserve_range=False)
    compute_bleedover = mode == 'constant' or mask is not None
    float_type = _supported_float_type(image.dtype)
    if mask is None:
        if compute_bleedover:
            mask = np.ones(image.shape, dtype=float_type)
        masked_image = image
        eroded_mask = np.ones(image.shape, dtype=bool)
        eroded_mask[:1, :] = 0
        eroded_mask[-1:, :] = 0
        eroded_mask[:, :1] = 0
        eroded_mask[:, -1:] = 0
    else:
        mask = mask.astype(bool, copy=False)
        masked_image = np.zeros_like(image)
        masked_image[mask] = image[mask]
        s = ndi.generate_binary_structure(2, 2)
        eroded_mask = ndi.binary_erosion(mask, s, border_value=0)
    if compute_bleedover:
        bleed_over = gaussian(mask.astype(float_type, copy=False), **gaussian_kwargs) + np.finfo(float_type).eps
    smoothed_image = gaussian(masked_image, **gaussian_kwargs)
    if compute_bleedover:
        smoothed_image /= bleed_over
    return (smoothed_image, eroded_mask)

def canny(image, sigma=1.0, low_threshold=None, high_threshold=None, mask=None, use_quantiles=False, *, mode='constant', cval=0.0):
    if False:
        while True:
            i = 10
    "Edge filter an image using the Canny algorithm.\n\n    Parameters\n    ----------\n    image : 2D array\n        Grayscale input image to detect edges on; can be of any dtype.\n    sigma : float, optional\n        Standard deviation of the Gaussian filter.\n    low_threshold : float, optional\n        Lower bound for hysteresis thresholding (linking edges).\n        If None, low_threshold is set to 10% of dtype's max.\n    high_threshold : float, optional\n        Upper bound for hysteresis thresholding (linking edges).\n        If None, high_threshold is set to 20% of dtype's max.\n    mask : array, dtype=bool, optional\n        Mask to limit the application of Canny to a certain area.\n    use_quantiles : bool, optional\n        If ``True`` then treat low_threshold and high_threshold as\n        quantiles of the edge magnitude image, rather than absolute\n        edge magnitude values. If ``True`` then the thresholds must be\n        in the range [0, 1].\n    mode : str, {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}\n        The ``mode`` parameter determines how the array borders are\n        handled during Gaussian filtering, where ``cval`` is the value when\n        mode is equal to 'constant'.\n    cval : float, optional\n        Value to fill past edges of input if `mode` is 'constant'.\n\n    Returns\n    -------\n    output : 2D array (image)\n        The binary edge map.\n\n    See also\n    --------\n    skimage.filters.sobel\n\n    Notes\n    -----\n    The steps of the algorithm are as follows:\n\n    * Smooth the image using a Gaussian with ``sigma`` width.\n\n    * Apply the horizontal and vertical Sobel operators to get the gradients\n      within the image. The edge strength is the norm of the gradient.\n\n    * Thin potential edges to 1-pixel wide curves. First, find the normal\n      to the edge at each point. This is done by looking at the\n      signs and the relative magnitude of the X-Sobel and Y-Sobel\n      to sort the points into 4 categories: horizontal, vertical,\n      diagonal and antidiagonal. Then look in the normal and reverse\n      directions to see if the values in either of those directions are\n      greater than the point in question. Use interpolation to get a mix of\n      points instead of picking the one that's the closest to the normal.\n\n    * Perform a hysteresis thresholding: first label all points above the\n      high threshold as edges. Then recursively label any point above the\n      low threshold that is 8-connected to a labeled point as an edge.\n\n    References\n    ----------\n    .. [1] Canny, J., A Computational Approach To Edge Detection, IEEE Trans.\n           Pattern Analysis and Machine Intelligence, 8:679-714, 1986\n           :DOI:`10.1109/TPAMI.1986.4767851`\n    .. [2] William Green's Canny tutorial\n           https://en.wikipedia.org/wiki/Canny_edge_detector\n\n    Examples\n    --------\n    >>> from skimage import feature\n    >>> rng = np.random.default_rng()\n    >>> # Generate noisy image of a square\n    >>> im = np.zeros((256, 256))\n    >>> im[64:-64, 64:-64] = 1\n    >>> im += 0.2 * rng.random(im.shape)\n    >>> # First trial with the Canny filter, with the default smoothing\n    >>> edges1 = feature.canny(im)\n    >>> # Increase the smoothing for better results\n    >>> edges2 = feature.canny(im, sigma=3)\n\n    "
    if np.issubdtype(image.dtype, np.int64) or np.issubdtype(image.dtype, np.uint64):
        raise ValueError('64-bit integer images are not supported')
    check_nD(image, 2)
    dtype_max = dtype_limits(image, clip_negative=False)[1]
    if low_threshold is None:
        low_threshold = 0.1
    elif use_quantiles:
        if not 0.0 <= low_threshold <= 1.0:
            raise ValueError('Quantile thresholds must be between 0 and 1.')
    else:
        low_threshold /= dtype_max
    if high_threshold is None:
        high_threshold = 0.2
    elif use_quantiles:
        if not 0.0 <= high_threshold <= 1.0:
            raise ValueError('Quantile thresholds must be between 0 and 1.')
    else:
        high_threshold /= dtype_max
    if high_threshold < low_threshold:
        raise ValueError('low_threshold should be lower then high_threshold')
    (smoothed, eroded_mask) = _preprocess(image, mask, sigma, mode, cval)
    jsobel = ndi.sobel(smoothed, axis=1)
    isobel = ndi.sobel(smoothed, axis=0)
    magnitude = isobel * isobel
    magnitude += jsobel * jsobel
    np.sqrt(magnitude, out=magnitude)
    if use_quantiles:
        (low_threshold, high_threshold) = np.percentile(magnitude, [100.0 * low_threshold, 100.0 * high_threshold])
    low_masked = _nonmaximum_suppression_bilinear(isobel, jsobel, magnitude, eroded_mask, low_threshold)
    low_mask = low_masked > 0
    strel = np.ones((3, 3), bool)
    (labels, count) = ndi.label(low_mask, strel)
    if count == 0:
        return low_mask
    high_mask = low_mask & (low_masked >= high_threshold)
    nonzero_sums = np.unique(labels[high_mask])
    good_label = np.zeros((count + 1,), bool)
    good_label[nonzero_sums] = True
    output_mask = good_label[labels]
    return output_mask