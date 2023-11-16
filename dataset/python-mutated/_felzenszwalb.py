import numpy as np
from ._felzenszwalb_cy import _felzenszwalb_cython
from .._shared import utils

@utils.channel_as_last_axis(multichannel_output=False)
def felzenszwalb(image, scale=1, sigma=0.8, min_size=20, *, channel_axis=-1):
    if False:
        for i in range(10):
            print('nop')
    "Computes Felsenszwalb's efficient graph based image segmentation.\n\n    Produces an oversegmentation of a multichannel (i.e. RGB) image\n    using a fast, minimum spanning tree based clustering on the image grid.\n    The parameter ``scale`` sets an observation level. Higher scale means\n    less and larger segments. ``sigma`` is the diameter of a Gaussian kernel,\n    used for smoothing the image prior to segmentation.\n\n    The number of produced segments as well as their size can only be\n    controlled indirectly through ``scale``. Segment size within an image can\n    vary greatly depending on local contrast.\n\n    For RGB images, the algorithm uses the euclidean distance between pixels in\n    color space.\n\n    Parameters\n    ----------\n    image : (M, N[, 3]) ndarray\n        Input image.\n    scale : float\n        Free parameter. Higher means larger clusters.\n    sigma : float\n        Width (standard deviation) of Gaussian kernel used in preprocessing.\n    min_size : int\n        Minimum component size. Enforced using postprocessing.\n    channel_axis : int or None, optional\n        If None, the image is assumed to be a grayscale (single channel) image.\n        Otherwise, this parameter indicates which axis of the array corresponds\n        to channels.\n\n        .. versionadded:: 0.19\n           ``channel_axis`` was added in 0.19.\n\n    Returns\n    -------\n    segment_mask : (M, N) ndarray\n        Integer mask indicating segment labels.\n\n    References\n    ----------\n    .. [1] Efficient graph-based image segmentation, Felzenszwalb, P.F. and\n           Huttenlocher, D.P.  International Journal of Computer Vision, 2004\n\n    Notes\n    -----\n        The `k` parameter used in the original paper renamed to `scale` here.\n\n    Examples\n    --------\n    >>> from skimage.segmentation import felzenszwalb\n    >>> from skimage.data import coffee\n    >>> img = coffee()\n    >>> segments = felzenszwalb(img, scale=3.0, sigma=0.95, min_size=5)\n    "
    if channel_axis is None and image.ndim > 2:
        raise ValueError('This algorithm works only on single or multi-channel 2d images. ')
    image = np.atleast_3d(image)
    return _felzenszwalb_cython(image, scale=scale, sigma=sigma, min_size=min_size)