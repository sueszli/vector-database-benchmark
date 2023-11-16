"""Vendored code from scikit-image in order to limit the number of dependencies
Extracted from scikit-image/skimage/exposure/exposure.py
"""
import numpy as np
from warnings import warn
_integer_types = (np.byte, np.ubyte, np.short, np.ushort, np.intc, np.uintc, np.int_, np.uint, np.longlong, np.ulonglong)
_integer_ranges = {t: (np.iinfo(t).min, np.iinfo(t).max) for t in _integer_types}
dtype_range = {np.bool_: (False, True), np.float16: (-1, 1), np.float32: (-1, 1), np.float64: (-1, 1)}
dtype_range.update(_integer_ranges)
DTYPE_RANGE = dtype_range.copy()
DTYPE_RANGE.update(((d.__name__, limits) for (d, limits) in dtype_range.items()))
DTYPE_RANGE.update({'uint10': (0, 2 ** 10 - 1), 'uint12': (0, 2 ** 12 - 1), 'uint14': (0, 2 ** 14 - 1), 'bool': dtype_range[np.bool_], 'float': dtype_range[np.float64]})

def intensity_range(image, range_values='image', clip_negative=False):
    if False:
        i = 10
        return i + 15
    "Return image intensity range (min, max) based on desired value type.\n\n    Parameters\n    ----------\n    image : array\n        Input image.\n    range_values : str or 2-tuple, optional\n        The image intensity range is configured by this parameter.\n        The possible values for this parameter are enumerated below.\n\n        'image'\n            Return image min/max as the range.\n        'dtype'\n            Return min/max of the image's dtype as the range.\n        dtype-name\n            Return intensity range based on desired `dtype`. Must be valid key\n            in `DTYPE_RANGE`. Note: `image` is ignored for this range type.\n        2-tuple\n            Return `range_values` as min/max intensities. Note that there's no\n            reason to use this function if you just want to specify the\n            intensity range explicitly. This option is included for functions\n            that use `intensity_range` to support all desired range types.\n\n    clip_negative : bool, optional\n        If True, clip the negative range (i.e. return 0 for min intensity)\n        even if the image dtype allows negative values.\n    "
    if range_values == 'dtype':
        range_values = image.dtype.type
    if range_values == 'image':
        i_min = np.min(image)
        i_max = np.max(image)
    elif range_values in DTYPE_RANGE:
        (i_min, i_max) = DTYPE_RANGE[range_values]
        if clip_negative:
            i_min = 0
    else:
        (i_min, i_max) = range_values
    return (i_min, i_max)

def _output_dtype(dtype_or_range):
    if False:
        print('Hello World!')
    "Determine the output dtype for rescale_intensity.\n\n    The dtype is determined according to the following rules:\n    - if ``dtype_or_range`` is a dtype, that is the output dtype.\n    - if ``dtype_or_range`` is a dtype string, that is the dtype used, unless\n      it is not a NumPy data type (e.g. 'uint12' for 12-bit unsigned integers),\n      in which case the data type that can contain it will be used\n      (e.g. uint16 in this case).\n    - if ``dtype_or_range`` is a pair of values, the output data type will be\n      float.\n\n    Parameters\n    ----------\n    dtype_or_range : type, string, or 2-tuple of int/float\n        The desired range for the output, expressed as either a NumPy dtype or\n        as a (min, max) pair of numbers.\n\n    Returns\n    -------\n    out_dtype : type\n        The data type appropriate for the desired output.\n    "
    if type(dtype_or_range) in [list, tuple, np.ndarray]:
        return np.float_
    if type(dtype_or_range) == type:
        return dtype_or_range
    if dtype_or_range in DTYPE_RANGE:
        try:
            return np.dtype(dtype_or_range).type
        except TypeError:
            return np.uint16
    else:
        raise ValueError('Incorrect value for out_range, should be a valid image data type or a pair of values, got %s.' % str(dtype_or_range))

def rescale_intensity(image, in_range='image', out_range='dtype'):
    if False:
        while True:
            i = 10
    "Return image after stretching or shrinking its intensity levels.\n\n    The desired intensity range of the input and output, `in_range` and\n    `out_range` respectively, are used to stretch or shrink the intensity range\n    of the input image. See examples below.\n\n    Parameters\n    ----------\n    image : array\n        Image array.\n    in_range, out_range : str or 2-tuple, optional\n        Min and max intensity values of input and output image.\n        The possible values for this parameter are enumerated below.\n\n        'image'\n            Use image min/max as the intensity range.\n        'dtype'\n            Use min/max of the image's dtype as the intensity range.\n        dtype-name\n            Use intensity range based on desired `dtype`. Must be valid key\n            in `DTYPE_RANGE`.\n        2-tuple\n            Use `range_values` as explicit min/max intensities.\n\n    Returns\n    -------\n    out : array\n        Image array after rescaling its intensity. This image is the same dtype\n        as the input image.\n\n    Notes\n    -----\n    .. versionchanged:: 0.17\n        The dtype of the output array has changed to match the output dtype, or\n        float if the output range is specified by a pair of floats.\n\n    See Also\n    --------\n    equalize_hist\n\n    Examples\n    --------\n    By default, the min/max intensities of the input image are stretched to\n    the limits allowed by the image's dtype, since `in_range` defaults to\n    'image' and `out_range` defaults to 'dtype':\n\n    >>> image = np.array([51, 102, 153], dtype=np.uint8)\n    >>> rescale_intensity(image)\n    array([  0, 127, 255], dtype=uint8)\n\n    It's easy to accidentally convert an image dtype from uint8 to float:\n\n    >>> 1.0 * image\n    array([ 51., 102., 153.])\n\n    Use `rescale_intensity` to rescale to the proper range for float dtypes:\n\n    >>> image_float = 1.0 * image\n    >>> rescale_intensity(image_float)\n    array([0. , 0.5, 1. ])\n\n    To maintain the low contrast of the original, use the `in_range` parameter:\n\n    >>> rescale_intensity(image_float, in_range=(0, 255))\n    array([0.2, 0.4, 0.6])\n\n    If the min/max value of `in_range` is more/less than the min/max image\n    intensity, then the intensity levels are clipped:\n\n    >>> rescale_intensity(image_float, in_range=(0, 102))\n    array([0.5, 1. , 1. ])\n\n    If you have an image with signed integers but want to rescale the image to\n    just the positive range, use the `out_range` parameter. In that case, the\n    output dtype will be float:\n\n    >>> image = np.array([-10, 0, 10], dtype=np.int8)\n    >>> rescale_intensity(image, out_range=(0, 127))\n    array([  0. ,  63.5, 127. ])\n\n    To get the desired range with a specific dtype, use ``.astype()``:\n\n    >>> rescale_intensity(image, out_range=(0, 127)).astype(np.int8)\n    array([  0,  63, 127], dtype=int8)\n\n    If the input image is constant, the output will be clipped directly to the\n    output range:\n    >>> image = np.array([130, 130, 130], dtype=np.int32)\n    >>> rescale_intensity(image, out_range=(0, 127)).astype(np.int32)\n    array([127, 127, 127], dtype=int32)\n    "
    if out_range in ['dtype', 'image']:
        out_dtype = _output_dtype(image.dtype.type)
    else:
        out_dtype = _output_dtype(out_range)
    (imin, imax) = map(float, intensity_range(image, in_range))
    (omin, omax) = map(float, intensity_range(image, out_range, clip_negative=imin >= 0))
    if np.any(np.isnan([imin, imax, omin, omax])):
        warn('One or more intensity levels are NaN. Rescaling will broadcast NaN to the full image. Provide intensity levels yourself to avoid this. E.g. with np.nanmin(image), np.nanmax(image).', stacklevel=2)
    image = np.clip(image, imin, imax)
    if imin != imax:
        image = (image - imin) / (imax - imin)
        return np.asarray(image * (omax - omin) + omin, dtype=out_dtype)
    else:
        return np.clip(image, omin, omax).astype(out_dtype)