import numpy as np
from ..util.dtype import img_as_float
from .._shared import utils
from .._shared.filters import gaussian

def _unsharp_mask_single_channel(image, radius, amount, vrange):
    if False:
        while True:
            i = 10
    'Single channel implementation of the unsharp masking filter.'
    blurred = gaussian(image, sigma=radius, mode='reflect')
    result = image + (image - blurred) * amount
    if vrange is not None:
        return np.clip(result, vrange[0], vrange[1], out=result)
    return result

def unsharp_mask(image, radius=1.0, amount=1.0, preserve_range=False, *, channel_axis=None):
    if False:
        return 10
    'Unsharp masking filter.\n\n    The sharp details are identified as the difference between the original\n    image and its blurred version. These details are then scaled, and added\n    back to the original image.\n\n    Parameters\n    ----------\n    image : (M[, ...][, C]) ndarray\n        Input image.\n    radius : scalar or sequence of scalars, optional\n        If a scalar is given, then its value is used for all dimensions.\n        If sequence is given, then there must be exactly one radius\n        for each dimension except the last dimension for multichannel images.\n        Note that 0 radius means no blurring, and negative values are\n        not allowed.\n    amount : scalar, optional\n        The details will be amplified with this factor. The factor could be 0\n        or negative. Typically, it is a small positive number, e.g. 1.0.\n    preserve_range : bool, optional\n        Whether to keep the original range of values. Otherwise, the input\n        image is converted according to the conventions of ``img_as_float``.\n        Also see https://scikit-image.org/docs/dev/user_guide/data_types.html\n    channel_axis : int or None, optional\n        If None, the image is assumed to be a grayscale (single channel) image.\n        Otherwise, this parameter indicates which axis of the array corresponds\n        to channels.\n\n        .. versionadded:: 0.19\n           ``channel_axis`` was added in 0.19.\n\n    Returns\n    -------\n    output : (M[, ...][, C]) ndarray of float\n        Image with unsharp mask applied.\n\n    Notes\n    -----\n    Unsharp masking is an image sharpening technique. It is a linear image\n    operation, and numerically stable, unlike deconvolution which is an\n    ill-posed problem. Because of this stability, it is often\n    preferred over deconvolution.\n\n    The main idea is as follows: sharp details are identified as the\n    difference between the original image and its blurred version.\n    These details are added back to the original image after a scaling step:\n\n        enhanced image = original + amount * (original - blurred)\n\n    When applying this filter to several color layers independently,\n    color bleeding may occur. More visually pleasing result can be\n    achieved by processing only the brightness/lightness/intensity\n    channel in a suitable color space such as HSV, HSL, YUV, or YCbCr.\n\n    Unsharp masking is described in most introductory digital image\n    processing books. This implementation is based on [1]_.\n\n    Examples\n    --------\n    >>> array = np.ones(shape=(5,5), dtype=np.uint8)*100\n    >>> array[2,2] = 120\n    >>> array\n    array([[100, 100, 100, 100, 100],\n           [100, 100, 100, 100, 100],\n           [100, 100, 120, 100, 100],\n           [100, 100, 100, 100, 100],\n           [100, 100, 100, 100, 100]], dtype=uint8)\n    >>> np.around(unsharp_mask(array, radius=0.5, amount=2),2)\n    array([[0.39, 0.39, 0.39, 0.39, 0.39],\n           [0.39, 0.39, 0.38, 0.39, 0.39],\n           [0.39, 0.38, 0.53, 0.38, 0.39],\n           [0.39, 0.39, 0.38, 0.39, 0.39],\n           [0.39, 0.39, 0.39, 0.39, 0.39]])\n\n    >>> array = np.ones(shape=(5,5), dtype=np.int8)*100\n    >>> array[2,2] = 127\n    >>> np.around(unsharp_mask(array, radius=0.5, amount=2),2)\n    array([[0.79, 0.79, 0.79, 0.79, 0.79],\n           [0.79, 0.78, 0.75, 0.78, 0.79],\n           [0.79, 0.75, 1.  , 0.75, 0.79],\n           [0.79, 0.78, 0.75, 0.78, 0.79],\n           [0.79, 0.79, 0.79, 0.79, 0.79]])\n\n    >>> np.around(unsharp_mask(array, radius=0.5, amount=2, preserve_range=True), 2)\n    array([[100.  , 100.  ,  99.99, 100.  , 100.  ],\n           [100.  ,  99.39,  95.48,  99.39, 100.  ],\n           [ 99.99,  95.48, 147.59,  95.48,  99.99],\n           [100.  ,  99.39,  95.48,  99.39, 100.  ],\n           [100.  , 100.  ,  99.99, 100.  , 100.  ]])\n\n\n    References\n    ----------\n    .. [1]  Maria Petrou, Costas Petrou\n            "Image Processing: The Fundamentals", (2010), ed ii., page 357,\n            ISBN 13: 9781119994398  :DOI:`10.1002/9781119994398`\n    .. [2]  Wikipedia. Unsharp masking\n            https://en.wikipedia.org/wiki/Unsharp_masking\n\n    '
    vrange = None
    float_dtype = utils._supported_float_type(image.dtype)
    if preserve_range:
        fimg = image.astype(float_dtype, copy=False)
    else:
        fimg = img_as_float(image).astype(float_dtype, copy=False)
        negative = np.any(fimg < 0)
        if negative:
            vrange = [-1.0, 1.0]
        else:
            vrange = [0.0, 1.0]
    if channel_axis is not None:
        result = np.empty_like(fimg, dtype=float_dtype)
        for channel in range(image.shape[channel_axis]):
            sl = utils.slice_at_axis(channel, channel_axis)
            result[sl] = _unsharp_mask_single_channel(fimg[sl], radius, amount, vrange)
        return result
    else:
        return _unsharp_mask_single_channel(fimg, radius, amount, vrange)