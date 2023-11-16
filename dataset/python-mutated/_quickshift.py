import numpy as np
from .._shared.filters import gaussian
from .._shared.utils import _supported_float_type, deprecate_kwarg
from ..color import rgb2lab
from ..util import img_as_float
from ._quickshift_cy import _quickshift_cython

@deprecate_kwarg({'random_seed': 'rng'}, deprecated_version='0.21', removed_version='0.23')
def quickshift(image, ratio=1.0, kernel_size=5, max_dist=10, return_tree=False, sigma=0, convert2lab=True, rng=42, *, channel_axis=-1):
    if False:
        i = 10
        return i + 15
    'Segment image using quickshift clustering in Color-(x,y) space.\n\n    Produces an oversegmentation of the image using the quickshift mode-seeking\n    algorithm.\n\n    Parameters\n    ----------\n    image : (M, N, C) ndarray\n        Input image. The axis corresponding to color channels can be specified\n        via the `channel_axis` argument.\n    ratio : float, optional, between 0 and 1\n        Balances color-space proximity and image-space proximity.\n        Higher values give more weight to color-space.\n    kernel_size : float, optional\n        Width of Gaussian kernel used in smoothing the\n        sample density. Higher means fewer clusters.\n    max_dist : float, optional\n        Cut-off point for data distances.\n        Higher means fewer clusters.\n    return_tree : bool, optional\n        Whether to return the full segmentation hierarchy tree and distances.\n    sigma : float, optional\n        Width for Gaussian smoothing as preprocessing. Zero means no smoothing.\n    convert2lab : bool, optional\n        Whether the input should be converted to Lab colorspace prior to\n        segmentation. For this purpose, the input is assumed to be RGB.\n    rng : {`numpy.random.Generator`, int}, optional\n        Pseudo-random number generator.\n        By default, a PCG64 generator is used (see :func:`numpy.random.default_rng`).\n        If `rng` is an int, it is used to seed the generator.\n\n        The PRNG is used to break ties, and is seeded with 42 by default.\n    channel_axis : int, optional\n        The axis of `image` corresponding to color channels. Defaults to the\n        last axis.\n\n    Returns\n    -------\n    segment_mask : (M, N) ndarray\n        Integer mask indicating segment labels.\n\n    Notes\n    -----\n    The authors advocate to convert the image to Lab color space prior to\n    segmentation, though this is not strictly necessary. For this to work, the\n    image must be given in RGB format.\n\n    References\n    ----------\n    .. [1] Quick shift and kernel methods for mode seeking,\n           Vedaldi, A. and Soatto, S.\n           European Conference on Computer Vision, 2008\n    '
    image = img_as_float(np.atleast_3d(image))
    float_dtype = _supported_float_type(image.dtype)
    image = image.astype(float_dtype, copy=False)
    if image.ndim > 3:
        raise ValueError('Only 2D color images are supported')
    image = np.moveaxis(image, source=channel_axis, destination=-1)
    if convert2lab:
        if image.shape[-1] != 3:
            raise ValueError('Only RGB images can be converted to Lab space.')
        image = rgb2lab(image)
    if kernel_size < 1:
        raise ValueError('`kernel_size` should be >= 1.')
    image = gaussian(image, [sigma, sigma, 0], mode='reflect', channel_axis=-1)
    image = np.ascontiguousarray(image * ratio)
    segment_mask = _quickshift_cython(image, kernel_size=kernel_size, max_dist=max_dist, return_tree=return_tree, rng=rng)
    return segment_mask