import numpy as np
from ..util.dtype import dtype_range, dtype_limits
from .._shared import utils
__all__ = ['histogram', 'cumulative_distribution', 'equalize_hist', 'rescale_intensity', 'adjust_gamma', 'adjust_log', 'adjust_sigmoid']
DTYPE_RANGE = dtype_range.copy()
DTYPE_RANGE.update(((d.__name__, limits) for (d, limits) in dtype_range.items()))
DTYPE_RANGE.update({'uint10': (0, 2 ** 10 - 1), 'uint12': (0, 2 ** 12 - 1), 'uint14': (0, 2 ** 14 - 1), 'bool': dtype_range[bool], 'float': dtype_range[np.float64]})

def _offset_array(arr, low_boundary, high_boundary):
    if False:
        for i in range(10):
            print('nop')
    'Offset the array to get the lowest value at 0 if negative.'
    if low_boundary < 0:
        offset = low_boundary
        dyn_range = high_boundary - low_boundary
        offset_dtype = np.promote_types(np.min_scalar_type(dyn_range), np.min_scalar_type(low_boundary))
        if arr.dtype != offset_dtype:
            arr = arr.astype(offset_dtype)
        arr = arr - offset
    return arr

def _bincount_histogram_centers(image, source_range):
    if False:
        while True:
            i = 10
    'Compute bin centers for bincount-based histogram.'
    if source_range not in ['image', 'dtype']:
        raise ValueError(f'Incorrect value for `source_range` argument: {source_range}')
    if source_range == 'image':
        image_min = int(image.min().astype(np.int64))
        image_max = int(image.max().astype(np.int64))
    elif source_range == 'dtype':
        (image_min, image_max) = dtype_limits(image, clip_negative=False)
    bin_centers = np.arange(image_min, image_max + 1)
    return bin_centers

def _bincount_histogram(image, source_range, bin_centers=None):
    if False:
        i = 10
        return i + 15
    "\n    Efficient histogram calculation for an image of integers.\n\n    This function is significantly more efficient than np.histogram but\n    works only on images of integers. It is based on np.bincount.\n\n    Parameters\n    ----------\n    image : array\n        Input image.\n    source_range : string\n        'image' determines the range from the input image.\n        'dtype' determines the range from the expected range of the images\n        of that data type.\n\n    Returns\n    -------\n    hist : array\n        The values of the histogram.\n    bin_centers : array\n        The values at the center of the bins.\n    "
    if bin_centers is None:
        bin_centers = _bincount_histogram_centers(image, source_range)
    (image_min, image_max) = (bin_centers[0], bin_centers[-1])
    image = _offset_array(image, image_min, image_max)
    hist = np.bincount(image.ravel(), minlength=image_max - min(image_min, 0) + 1)
    if source_range == 'image':
        idx = max(image_min, 0)
        hist = hist[idx:]
    return (hist, bin_centers)

def _get_outer_edges(image, hist_range):
    if False:
        while True:
            i = 10
    'Determine the outer bin edges to use for `numpy.histogram`.\n\n    These are obtained from either the image or hist_range.\n\n    Parameters\n    ----------\n    image : ndarray\n        Image for which the histogram is to be computed.\n    hist_range: 2-tuple of int or None\n        Range of values covered by the histogram bins. If None, the minimum\n        and maximum values of `image` are used.\n\n    Returns\n    -------\n    first_edge, last_edge : int\n        The range spanned by the histogram bins.\n\n    Notes\n    -----\n    This function is adapted from ``np.lib.histograms._get_outer_edges``.\n    '
    if hist_range is not None:
        (first_edge, last_edge) = hist_range
        if first_edge > last_edge:
            raise ValueError('max must be larger than min in hist_range parameter.')
        if not (np.isfinite(first_edge) and np.isfinite(last_edge)):
            raise ValueError(f'supplied hist_range of [{first_edge}, {last_edge}] is not finite')
    elif image.size == 0:
        (first_edge, last_edge) = (0, 1)
    else:
        (first_edge, last_edge) = (image.min(), image.max())
        if not (np.isfinite(first_edge) and np.isfinite(last_edge)):
            raise ValueError(f'autodetected hist_range of [{first_edge}, {last_edge}] is not finite')
    if first_edge == last_edge:
        first_edge = first_edge - 0.5
        last_edge = last_edge + 0.5
    return (first_edge, last_edge)

def _get_bin_edges(image, nbins, hist_range):
    if False:
        return 10
    'Computes histogram bins for use with `numpy.histogram`.\n\n    Parameters\n    ----------\n    image : ndarray\n        Image for which the histogram is to be computed.\n    nbins : int\n        The number of bins.\n    hist_range: 2-tuple of int\n        Range of values covered by the histogram bins.\n\n    Returns\n    -------\n    bin_edges : ndarray\n        The histogram bin edges.\n\n    Notes\n    -----\n    This function is a simplified version of\n    ``np.lib.histograms._get_bin_edges`` that only supports uniform bins.\n    '
    (first_edge, last_edge) = _get_outer_edges(image, hist_range)
    bin_type = np.result_type(first_edge, last_edge, image)
    if np.issubdtype(bin_type, np.integer):
        bin_type = np.result_type(bin_type, float)
    bin_edges = np.linspace(first_edge, last_edge, nbins + 1, endpoint=True, dtype=bin_type)
    return bin_edges

def _get_numpy_hist_range(image, source_range):
    if False:
        while True:
            i = 10
    if source_range == 'image':
        hist_range = None
    elif source_range == 'dtype':
        hist_range = dtype_limits(image, clip_negative=False)
    else:
        raise ValueError(f'Incorrect value for `source_range` argument: {source_range}')
    return hist_range

@utils.channel_as_last_axis(multichannel_output=False)
def histogram(image, nbins=256, source_range='image', normalize=False, *, channel_axis=None):
    if False:
        print('Hello World!')
    "Return histogram of image.\n\n    Unlike `numpy.histogram`, this function returns the centers of bins and\n    does not rebin integer arrays. For integer arrays, each integer value has\n    its own bin, which improves speed and intensity-resolution.\n\n    If `channel_axis` is not set, the histogram is computed on the flattened\n    image. For color or multichannel images, set ``channel_axis`` to use a\n    common binning for all channels. Alternatively, one may apply the function\n    separately on each channel to obtain a histogram for each color channel\n    with separate binning.\n\n    Parameters\n    ----------\n    image : array\n        Input image.\n    nbins : int, optional\n        Number of bins used to calculate histogram. This value is ignored for\n        integer arrays.\n    source_range : string, optional\n        'image' (default) determines the range from the input image.\n        'dtype' determines the range from the expected range of the images\n        of that data type.\n    normalize : bool, optional\n        If True, normalize the histogram by the sum of its values.\n    channel_axis : int or None, optional\n        If None, the image is assumed to be a grayscale (single channel) image.\n        Otherwise, this parameter indicates which axis of the array corresponds\n        to channels.\n\n    Returns\n    -------\n    hist : array\n        The values of the histogram. When ``channel_axis`` is not None, hist\n        will be a 2D array where the first axis corresponds to channels.\n    bin_centers : array\n        The values at the center of the bins.\n\n    See Also\n    --------\n    cumulative_distribution\n\n    Examples\n    --------\n    >>> from skimage import data, exposure, img_as_float\n    >>> image = img_as_float(data.camera())\n    >>> np.histogram(image, bins=2)\n    (array([ 93585, 168559]), array([0. , 0.5, 1. ]))\n    >>> exposure.histogram(image, nbins=2)\n    (array([ 93585, 168559]), array([0.25, 0.75]))\n    "
    sh = image.shape
    if len(sh) == 3 and sh[-1] < 4 and (channel_axis is None):
        utils.warn('This might be a color image. The histogram will be computed on the flattened image. You can instead apply this function to each color channel, or set channel_axis.')
    if channel_axis is not None:
        channels = sh[-1]
        hist = []
        if np.issubdtype(image.dtype, np.integer):
            bins = _bincount_histogram_centers(image, source_range)
        else:
            hist_range = _get_numpy_hist_range(image, source_range)
            bins = _get_bin_edges(image, nbins, hist_range)
        for chan in range(channels):
            (h, bc) = _histogram(image[..., chan], bins, source_range, normalize)
            hist.append(h)
        bin_centers = np.asarray(bc)
        hist = np.stack(hist, axis=0)
    else:
        (hist, bin_centers) = _histogram(image, nbins, source_range, normalize)
    return (hist, bin_centers)

def _histogram(image, bins, source_range, normalize):
    if False:
        for i in range(10):
            print('nop')
    "\n\n    Parameters\n    ----------\n    image : ndarray\n        Image for which the histogram is to be computed.\n    bins : int or ndarray\n        The number of histogram bins. For images with integer dtype, an array\n        containing the bin centers can also be provided. For images with\n        floating point dtype, this can be an array of bin_edges for use by\n        ``np.histogram``.\n    source_range : string, optional\n        'image' (default) determines the range from the input image.\n        'dtype' determines the range from the expected range of the images\n        of that data type.\n    normalize : bool, optional\n        If True, normalize the histogram by the sum of its values.\n    "
    image = image.flatten()
    if np.issubdtype(image.dtype, np.integer):
        bin_centers = bins if isinstance(bins, np.ndarray) else None
        (hist, bin_centers) = _bincount_histogram(image, source_range, bin_centers)
    else:
        hist_range = _get_numpy_hist_range(image, source_range)
        (hist, bin_edges) = np.histogram(image, bins=bins, range=hist_range)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    if normalize:
        hist = hist / np.sum(hist)
    return (hist, bin_centers)

def cumulative_distribution(image, nbins=256):
    if False:
        print('Hello World!')
    'Return cumulative distribution function (cdf) for the given image.\n\n    Parameters\n    ----------\n    image : array\n        Image array.\n    nbins : int, optional\n        Number of bins for image histogram.\n\n    Returns\n    -------\n    img_cdf : array\n        Values of cumulative distribution function.\n    bin_centers : array\n        Centers of bins.\n\n    See Also\n    --------\n    histogram\n\n    References\n    ----------\n    .. [1] https://en.wikipedia.org/wiki/Cumulative_distribution_function\n\n    Examples\n    --------\n    >>> from skimage import data, exposure, img_as_float\n    >>> image = img_as_float(data.camera())\n    >>> hi = exposure.histogram(image)\n    >>> cdf = exposure.cumulative_distribution(image)\n    >>> all(cdf[0] == np.cumsum(hi[0])/float(image.size))\n    True\n    '
    (hist, bin_centers) = histogram(image, nbins)
    img_cdf = hist.cumsum()
    img_cdf = img_cdf / float(img_cdf[-1])
    cdf_dtype = utils._supported_float_type(image.dtype)
    img_cdf = img_cdf.astype(cdf_dtype, copy=False)
    return (img_cdf, bin_centers)

def equalize_hist(image, nbins=256, mask=None):
    if False:
        i = 10
        return i + 15
    "Return image after histogram equalization.\n\n    Parameters\n    ----------\n    image : array\n        Image array.\n    nbins : int, optional\n        Number of bins for image histogram. Note: this argument is\n        ignored for integer images, for which each integer is its own\n        bin.\n    mask : ndarray of bools or 0s and 1s, optional\n        Array of same shape as `image`. Only points at which mask == True\n        are used for the equalization, which is applied to the whole image.\n\n    Returns\n    -------\n    out : float array\n        Image array after histogram equalization.\n\n    Notes\n    -----\n    This function is adapted from [1]_ with the author's permission.\n\n    References\n    ----------\n    .. [1] http://www.janeriksolem.net/histogram-equalization-with-python-and.html\n    .. [2] https://en.wikipedia.org/wiki/Histogram_equalization\n\n    "
    if mask is not None:
        mask = np.array(mask, dtype=bool)
        (cdf, bin_centers) = cumulative_distribution(image[mask], nbins)
    else:
        (cdf, bin_centers) = cumulative_distribution(image, nbins)
    out = np.interp(image.flat, bin_centers, cdf)
    out = out.reshape(image.shape)
    return out.astype(utils._supported_float_type(image.dtype), copy=False)

def intensity_range(image, range_values='image', clip_negative=False):
    if False:
        return 10
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

def _output_dtype(dtype_or_range, image_dtype):
    if False:
        return 10
    "Determine the output dtype for rescale_intensity.\n\n    The dtype is determined according to the following rules:\n    - if ``dtype_or_range`` is a dtype, that is the output dtype.\n    - if ``dtype_or_range`` is a dtype string, that is the dtype used, unless\n      it is not a NumPy data type (e.g. 'uint12' for 12-bit unsigned integers),\n      in which case the data type that can contain it will be used\n      (e.g. uint16 in this case).\n    - if ``dtype_or_range`` is a pair of values, the output data type will be\n      ``_supported_float_type(image_dtype)``. This preserves float32 output for\n      float32 inputs.\n\n    Parameters\n    ----------\n    dtype_or_range : type, string, or 2-tuple of int/float\n        The desired range for the output, expressed as either a NumPy dtype or\n        as a (min, max) pair of numbers.\n    image_dtype : np.dtype\n        The input image dtype.\n\n    Returns\n    -------\n    out_dtype : type\n        The data type appropriate for the desired output.\n    "
    if type(dtype_or_range) in [list, tuple, np.ndarray]:
        return utils._supported_float_type(image_dtype)
    if type(dtype_or_range) == type:
        return dtype_or_range
    if dtype_or_range in DTYPE_RANGE:
        try:
            return np.dtype(dtype_or_range).type
        except TypeError:
            return np.uint16
    else:
        raise ValueError(f'Incorrect value for out_range, should be a valid image data type or a pair of values, got {dtype_or_range}.')

def rescale_intensity(image, in_range='image', out_range='dtype'):
    if False:
        for i in range(10):
            print('nop')
    "Return image after stretching or shrinking its intensity levels.\n\n    The desired intensity range of the input and output, `in_range` and\n    `out_range` respectively, are used to stretch or shrink the intensity range\n    of the input image. See examples below.\n\n    Parameters\n    ----------\n    image : array\n        Image array.\n    in_range, out_range : str or 2-tuple, optional\n        Min and max intensity values of input and output image.\n        The possible values for this parameter are enumerated below.\n\n        'image'\n            Use image min/max as the intensity range.\n        'dtype'\n            Use min/max of the image's dtype as the intensity range.\n        dtype-name\n            Use intensity range based on desired `dtype`. Must be valid key\n            in `DTYPE_RANGE`.\n        2-tuple\n            Use `range_values` as explicit min/max intensities.\n\n    Returns\n    -------\n    out : array\n        Image array after rescaling its intensity. This image is the same dtype\n        as the input image.\n\n    Notes\n    -----\n    .. versionchanged:: 0.17\n        The dtype of the output array has changed to match the input dtype, or\n        float if the output range is specified by a pair of values.\n\n    See Also\n    --------\n    equalize_hist\n\n    Examples\n    --------\n    By default, the min/max intensities of the input image are stretched to\n    the limits allowed by the image's dtype, since `in_range` defaults to\n    'image' and `out_range` defaults to 'dtype':\n\n    >>> image = np.array([51, 102, 153], dtype=np.uint8)\n    >>> rescale_intensity(image)\n    array([  0, 127, 255], dtype=uint8)\n\n    It's easy to accidentally convert an image dtype from uint8 to float:\n\n    >>> 1.0 * image\n    array([ 51., 102., 153.])\n\n    Use `rescale_intensity` to rescale to the proper range for float dtypes:\n\n    >>> image_float = 1.0 * image\n    >>> rescale_intensity(image_float)\n    array([0. , 0.5, 1. ])\n\n    To maintain the low contrast of the original, use the `in_range` parameter:\n\n    >>> rescale_intensity(image_float, in_range=(0, 255))\n    array([0.2, 0.4, 0.6])\n\n    If the min/max value of `in_range` is more/less than the min/max image\n    intensity, then the intensity levels are clipped:\n\n    >>> rescale_intensity(image_float, in_range=(0, 102))\n    array([0.5, 1. , 1. ])\n\n    If you have an image with signed integers but want to rescale the image to\n    just the positive range, use the `out_range` parameter. In that case, the\n    output dtype will be float:\n\n    >>> image = np.array([-10, 0, 10], dtype=np.int8)\n    >>> rescale_intensity(image, out_range=(0, 127))\n    array([  0. ,  63.5, 127. ])\n\n    To get the desired range with a specific dtype, use ``.astype()``:\n\n    >>> rescale_intensity(image, out_range=(0, 127)).astype(np.int8)\n    array([  0,  63, 127], dtype=int8)\n\n    If the input image is constant, the output will be clipped directly to the\n    output range:\n    >>> image = np.array([130, 130, 130], dtype=np.int32)\n    >>> rescale_intensity(image, out_range=(0, 127)).astype(np.int32)\n    array([127, 127, 127], dtype=int32)\n    "
    if out_range in ['dtype', 'image']:
        out_dtype = _output_dtype(image.dtype.type, image.dtype)
    else:
        out_dtype = _output_dtype(out_range, image.dtype)
    (imin, imax) = map(float, intensity_range(image, in_range))
    (omin, omax) = map(float, intensity_range(image, out_range, clip_negative=imin >= 0))
    if np.any(np.isnan([imin, imax, omin, omax])):
        utils.warn('One or more intensity levels are NaN. Rescaling will broadcast NaN to the full image. Provide intensity levels yourself to avoid this. E.g. with np.nanmin(image), np.nanmax(image).', stacklevel=2)
    image = np.clip(image, imin, imax)
    if imin != imax:
        image = (image - imin) / (imax - imin)
        return (image * (omax - omin) + omin).astype(out_dtype)
    else:
        return np.clip(image, omin, omax).astype(out_dtype)

def _assert_non_negative(image):
    if False:
        return 10
    if np.any(image < 0):
        raise ValueError('Image Correction methods work correctly only on images with non-negative values. Use skimage.exposure.rescale_intensity.')

def _adjust_gamma_u8(image, gamma, gain):
    if False:
        return 10
    'LUT based implementation of gamma adjustment.'
    lut = 255 * gain * np.linspace(0, 1, 256) ** gamma
    lut = np.minimum(np.rint(lut), 255).astype('uint8')
    return lut[image]

def adjust_gamma(image, gamma=1, gain=1):
    if False:
        print('Hello World!')
    'Performs Gamma Correction on the input image.\n\n    Also known as Power Law Transform.\n    This function transforms the input image pixelwise according to the\n    equation ``O = I**gamma`` after scaling each pixel to the range 0 to 1.\n\n    Parameters\n    ----------\n    image : ndarray\n        Input image.\n    gamma : float, optional\n        Non negative real number. Default value is 1.\n    gain : float, optional\n        The constant multiplier. Default value is 1.\n\n    Returns\n    -------\n    out : ndarray\n        Gamma corrected output image.\n\n    See Also\n    --------\n    adjust_log\n\n    Notes\n    -----\n    For gamma greater than 1, the histogram will shift towards left and\n    the output image will be darker than the input image.\n\n    For gamma less than 1, the histogram will shift towards right and\n    the output image will be brighter than the input image.\n\n    References\n    ----------\n    .. [1] https://en.wikipedia.org/wiki/Gamma_correction\n\n    Examples\n    --------\n    >>> from skimage import data, exposure, img_as_float\n    >>> image = img_as_float(data.moon())\n    >>> gamma_corrected = exposure.adjust_gamma(image, 2)\n    >>> # Output is darker for gamma > 1\n    >>> image.mean() > gamma_corrected.mean()\n    True\n    '
    if gamma < 0:
        raise ValueError('Gamma should be a non-negative real number.')
    dtype = image.dtype.type
    if dtype is np.uint8:
        out = _adjust_gamma_u8(image, gamma, gain)
    else:
        _assert_non_negative(image)
        scale = float(dtype_limits(image, True)[1] - dtype_limits(image, True)[0])
        out = ((image / scale) ** gamma * scale * gain).astype(dtype)
    return out

def adjust_log(image, gain=1, inv=False):
    if False:
        for i in range(10):
            print('nop')
    'Performs Logarithmic correction on the input image.\n\n    This function transforms the input image pixelwise according to the\n    equation ``O = gain*log(1 + I)`` after scaling each pixel to the range\n    0 to 1. For inverse logarithmic correction, the equation is\n    ``O = gain*(2**I - 1)``.\n\n    Parameters\n    ----------\n    image : ndarray\n        Input image.\n    gain : float, optional\n        The constant multiplier. Default value is 1.\n    inv : float, optional\n        If True, it performs inverse logarithmic correction,\n        else correction will be logarithmic. Defaults to False.\n\n    Returns\n    -------\n    out : ndarray\n        Logarithm corrected output image.\n\n    See Also\n    --------\n    adjust_gamma\n\n    References\n    ----------\n    .. [1] http://www.ece.ucsb.edu/Faculty/Manjunath/courses/ece178W03/EnhancePart1.pdf\n\n    '
    _assert_non_negative(image)
    dtype = image.dtype.type
    scale = float(dtype_limits(image, True)[1] - dtype_limits(image, True)[0])
    if inv:
        out = (2 ** (image / scale) - 1) * scale * gain
        return dtype(out)
    out = np.log2(1 + image / scale) * scale * gain
    return out.astype(dtype)

def adjust_sigmoid(image, cutoff=0.5, gain=10, inv=False):
    if False:
        print('Hello World!')
    'Performs Sigmoid Correction on the input image.\n\n    Also known as Contrast Adjustment.\n    This function transforms the input image pixelwise according to the\n    equation ``O = 1/(1 + exp*(gain*(cutoff - I)))`` after scaling each pixel\n    to the range 0 to 1.\n\n    Parameters\n    ----------\n    image : ndarray\n        Input image.\n    cutoff : float, optional\n        Cutoff of the sigmoid function that shifts the characteristic curve\n        in horizontal direction. Default value is 0.5.\n    gain : float, optional\n        The constant multiplier in exponential\'s power of sigmoid function.\n        Default value is 10.\n    inv : bool, optional\n        If True, returns the negative sigmoid correction. Defaults to False.\n\n    Returns\n    -------\n    out : ndarray\n        Sigmoid corrected output image.\n\n    See Also\n    --------\n    adjust_gamma\n\n    References\n    ----------\n    .. [1] Gustav J. Braun, "Image Lightness Rescaling Using Sigmoidal Contrast\n           Enhancement Functions",\n           http://markfairchild.org/PDFs/PAP07.pdf\n\n    '
    _assert_non_negative(image)
    dtype = image.dtype.type
    scale = float(dtype_limits(image, True)[1] - dtype_limits(image, True)[0])
    if inv:
        out = (1 - 1 / (1 + np.exp(gain * (cutoff - image / scale)))) * scale
        return dtype(out)
    out = 1 / (1 + np.exp(gain * (cutoff - image / scale))) * scale
    return out.astype(dtype)

def is_low_contrast(image, fraction_threshold=0.05, lower_percentile=1, upper_percentile=99, method='linear'):
    if False:
        i = 10
        return i + 15
    'Determine if an image is low contrast.\n\n    Parameters\n    ----------\n    image : array-like\n        The image under test.\n    fraction_threshold : float, optional\n        The low contrast fraction threshold. An image is considered low-\n        contrast when its range of brightness spans less than this\n        fraction of its data type\'s full range. [1]_\n    lower_percentile : float, optional\n        Disregard values below this percentile when computing image contrast.\n    upper_percentile : float, optional\n        Disregard values above this percentile when computing image contrast.\n    method : str, optional\n        The contrast determination method.  Right now the only available\n        option is "linear".\n\n    Returns\n    -------\n    out : bool\n        True when the image is determined to be low contrast.\n\n    Notes\n    -----\n    For boolean images, this function returns False only if all values are\n    the same (the method, threshold, and percentile arguments are ignored).\n\n    References\n    ----------\n    .. [1] https://scikit-image.org/docs/dev/user_guide/data_types.html\n\n    Examples\n    --------\n    >>> image = np.linspace(0, 0.04, 100)\n    >>> is_low_contrast(image)\n    True\n    >>> image[-1] = 1\n    >>> is_low_contrast(image)\n    True\n    >>> is_low_contrast(image, upper_percentile=100)\n    False\n    '
    image = np.asanyarray(image)
    if image.dtype == bool:
        return not (image.max() == 1 and image.min() == 0)
    if image.ndim == 3:
        from ..color import rgb2gray, rgba2rgb
        if image.shape[2] == 4:
            image = rgba2rgb(image)
        if image.shape[2] == 3:
            image = rgb2gray(image)
    dlimits = dtype_limits(image, clip_negative=False)
    limits = np.percentile(image, [lower_percentile, upper_percentile])
    ratio = (limits[1] - limits[0]) / (dlimits[1] - dlimits[0])
    return ratio < fraction_threshold