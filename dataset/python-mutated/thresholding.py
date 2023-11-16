import inspect
import itertools
import math
from collections import OrderedDict
from collections.abc import Iterable
import numpy as np
from scipy import ndimage as ndi
from .._shared.filters import gaussian
from .._shared.utils import _supported_float_type, warn
from .._shared.version_requirements import require
from ..exposure import histogram
from ..filters._multiotsu import _get_multiotsu_thresh_indices, _get_multiotsu_thresh_indices_lut
from ..transform import integral_image
from ..util import dtype_limits
from ._sparse import _correlate_sparse, _validate_window_size
__all__ = ['try_all_threshold', 'threshold_otsu', 'threshold_yen', 'threshold_isodata', 'threshold_li', 'threshold_local', 'threshold_minimum', 'threshold_mean', 'threshold_niblack', 'threshold_sauvola', 'threshold_triangle', 'apply_hysteresis_threshold', 'threshold_multiotsu']

def _try_all(image, methods=None, figsize=None, num_cols=2, verbose=True):
    if False:
        i = 10
        return i + 15
    'Returns a figure comparing the outputs of different methods.\n\n    Parameters\n    ----------\n    image : (M, N) ndarray\n        Input image.\n    methods : dict, optional\n        Names and associated functions.\n        Functions must take and return an image.\n    figsize : tuple, optional\n        Figure size (in inches).\n    num_cols : int, optional\n        Number of columns.\n    verbose : bool, optional\n        Print function name for each method.\n\n    Returns\n    -------\n    fig, ax : tuple\n        Matplotlib figure and axes.\n    '
    from matplotlib import pyplot as plt
    nbins = 256
    hist = histogram(image.reshape(-1), nbins, source_range='image')
    methods = methods or {}
    num_rows = math.ceil((len(methods) + 1.0) / num_cols)
    (fig, ax) = plt.subplots(num_rows, num_cols, figsize=figsize, sharex=True, sharey=True)
    ax = ax.reshape(-1)
    ax[0].imshow(image, cmap=plt.cm.gray)
    ax[0].set_title('Original')
    i = 1
    for (name, func) in methods.items():
        sig = inspect.signature(func)
        _kwargs = dict(hist=hist) if 'hist' in sig.parameters else {}
        ax[i].set_title(name)
        try:
            ax[i].imshow(func(image, **_kwargs), cmap=plt.cm.gray)
        except Exception as e:
            ax[i].text(0.5, 0.5, f'{type(e).__name__}', ha='center', va='center', transform=ax[i].transAxes)
        i += 1
        if verbose:
            print(func.__orifunc__)
    for a in ax:
        a.axis('off')
    fig.tight_layout()
    return (fig, ax)

@require('matplotlib', '>=3.3')
def try_all_threshold(image, figsize=(8, 5), verbose=True):
    if False:
        i = 10
        return i + 15
    'Returns a figure comparing the outputs of different thresholding methods.\n\n    Parameters\n    ----------\n    image : (M, N) ndarray\n        Input image.\n    figsize : tuple, optional\n        Figure size (in inches).\n    verbose : bool, optional\n        Print function name for each method.\n\n    Returns\n    -------\n    fig, ax : tuple\n        Matplotlib figure and axes.\n\n    Notes\n    -----\n    The following algorithms are used:\n\n    * isodata\n    * li\n    * mean\n    * minimum\n    * otsu\n    * triangle\n    * yen\n\n    Examples\n    --------\n    >>> from skimage.data import text\n    >>> fig, ax = try_all_threshold(text(), figsize=(10, 6), verbose=False)\n    '

    def thresh(func):
        if False:
            i = 10
            return i + 15
        '\n        A wrapper function to return a thresholded image.\n        '

        def wrapper(im):
            if False:
                for i in range(10):
                    print('nop')
            return im > func(im)
        try:
            wrapper.__orifunc__ = func.__orifunc__
        except AttributeError:
            wrapper.__orifunc__ = func.__module__ + '.' + func.__name__
        return wrapper
    methods = OrderedDict({'Isodata': thresh(threshold_isodata), 'Li': thresh(threshold_li), 'Mean': thresh(threshold_mean), 'Minimum': thresh(threshold_minimum), 'Otsu': thresh(threshold_otsu), 'Triangle': thresh(threshold_triangle), 'Yen': thresh(threshold_yen)})
    return _try_all(image, figsize=figsize, methods=methods, verbose=verbose)

def threshold_local(image, block_size=3, method='gaussian', offset=0, mode='reflect', param=None, cval=0):
    if False:
        for i in range(10):
            print('nop')
    'Compute a threshold mask image based on local pixel neighborhood.\n\n    Also known as adaptive or dynamic thresholding. The threshold value is\n    the weighted mean for the local neighborhood of a pixel subtracted by a\n    constant. Alternatively the threshold can be determined dynamically by a\n    given function, using the \'generic\' method.\n\n    Parameters\n    ----------\n    image : (M, N[, ...]) ndarray\n        Grayscale input image.\n    block_size : int or sequence of int\n        Odd size of pixel neighborhood which is used to calculate the\n        threshold value (e.g. 3, 5, 7, ..., 21, ...).\n    method : {\'generic\', \'gaussian\', \'mean\', \'median\'}, optional\n        Method used to determine adaptive threshold for local neighborhood in\n        weighted mean image.\n\n        * \'generic\': use custom function (see ``param`` parameter)\n        * \'gaussian\': apply gaussian filter (see ``param`` parameter for custom                      sigma value)\n        * \'mean\': apply arithmetic mean filter\n        * \'median\': apply median rank filter\n\n        By default, the \'gaussian\' method is used.\n    offset : float, optional\n        Constant subtracted from weighted mean of neighborhood to calculate\n        the local threshold value. Default offset is 0.\n    mode : {\'reflect\', \'constant\', \'nearest\', \'mirror\', \'wrap\'}, optional\n        The mode parameter determines how the array borders are handled, where\n        cval is the value when mode is equal to \'constant\'.\n        Default is \'reflect\'.\n    param : {int, function}, optional\n        Either specify sigma for \'gaussian\' method or function object for\n        \'generic\' method. This functions takes the flat array of local\n        neighborhood as a single argument and returns the calculated\n        threshold for the centre pixel.\n    cval : float, optional\n        Value to fill past edges of input if mode is \'constant\'.\n\n    Returns\n    -------\n    threshold : (M, N[, ...]) ndarray\n        Threshold image. All pixels in the input image higher than the\n        corresponding pixel in the threshold image are considered foreground.\n\n    References\n    ----------\n    .. [1] Gonzalez, R. C. and Wood, R. E. "Digital Image Processing\n           (2nd Edition)." Prentice-Hall Inc., 2002: 600--612.\n           ISBN: 0-201-18075-8\n\n    Examples\n    --------\n    >>> from skimage.data import camera\n    >>> image = camera()[:50, :50]\n    >>> binary_image1 = image > threshold_local(image, 15, \'mean\')\n    >>> func = lambda arr: arr.mean()\n    >>> binary_image2 = image > threshold_local(image, 15, \'generic\',\n    ...                                         param=func)\n\n    '
    if np.isscalar(block_size):
        block_size = (block_size,) * image.ndim
    elif len(block_size) != image.ndim:
        raise ValueError('len(block_size) must equal image.ndim.')
    block_size = tuple(block_size)
    if any((b % 2 == 0 for b in block_size)):
        raise ValueError(f'block_size must be odd! Given block_size {block_size} contains even values.')
    float_dtype = _supported_float_type(image.dtype)
    image = image.astype(float_dtype, copy=False)
    thresh_image = np.zeros(image.shape, dtype=float_dtype)
    if method == 'generic':
        ndi.generic_filter(image, param, block_size, output=thresh_image, mode=mode, cval=cval)
    elif method == 'gaussian':
        if param is None:
            sigma = tuple([(b - 1) / 6.0 for b in block_size])
        else:
            sigma = param
        gaussian(image, sigma, output=thresh_image, mode=mode, cval=cval)
    elif method == 'mean':
        ndi.uniform_filter(image, block_size, output=thresh_image, mode=mode, cval=cval)
    elif method == 'median':
        ndi.median_filter(image, block_size, output=thresh_image, mode=mode, cval=cval)
    else:
        raise ValueError('Invalid method specified. Please use `generic`, `gaussian`, `mean`, or `median`.')
    return thresh_image - offset

def _validate_image_histogram(image, hist, nbins=None, normalize=False):
    if False:
        for i in range(10):
            print('nop')
    'Ensure that either image or hist were given, return valid histogram.\n\n    If hist is given, image is ignored.\n\n    Parameters\n    ----------\n    image : array or None\n        Grayscale image.\n    hist : array, 2-tuple of array, or None\n        Histogram, either a 1D counts array, or an array of counts together\n        with an array of bin centers.\n    nbins : int, optional\n        The number of bins with which to compute the histogram, if `hist` is\n        None.\n    normalize : bool\n        If hist is not given, it will be computed by this function. This\n        parameter determines whether the computed histogram is normalized\n        (i.e. entries sum up to 1) or not.\n\n    Returns\n    -------\n    counts : 1D array of float\n        Each element is the number of pixels falling in each intensity bin.\n    bin_centers : 1D array\n        Each element is the value corresponding to the center of each intensity\n        bin.\n\n    Raises\n    ------\n    ValueError : if image and hist are both None\n    '
    if image is None and hist is None:
        raise Exception('Either image or hist must be provided.')
    if hist is not None:
        if isinstance(hist, tuple | list):
            (counts, bin_centers) = hist
        else:
            counts = hist
            bin_centers = np.arange(counts.size)
        if counts[0] == 0 or counts[-1] == 0:
            cond = counts > 0
            start = np.argmax(cond)
            end = cond.size - np.argmax(cond[::-1])
            (counts, bin_centers) = (counts[start:end], bin_centers[start:end])
    else:
        (counts, bin_centers) = histogram(image.reshape(-1), nbins, source_range='image', normalize=normalize)
    return (counts.astype('float32', copy=False), bin_centers)

def threshold_otsu(image=None, nbins=256, *, hist=None):
    if False:
        for i in range(10):
            print('nop')
    "Return threshold value based on Otsu's method.\n\n    Either image or hist must be provided. If hist is provided, the actual\n    histogram of the image is ignored.\n\n    Parameters\n    ----------\n    image : (M, N[, ...]) ndarray, optional\n        Grayscale input image.\n    nbins : int, optional\n        Number of bins used to calculate histogram. This value is ignored for\n        integer arrays.\n    hist : array, or 2-tuple of arrays, optional\n        Histogram from which to determine the threshold, and optionally a\n        corresponding array of bin center intensities. If no hist provided,\n        this function will compute it from the image.\n\n\n    Returns\n    -------\n    threshold : float\n        Upper threshold value. All pixels with an intensity higher than\n        this value are assumed to be foreground.\n\n    References\n    ----------\n    .. [1] Wikipedia, https://en.wikipedia.org/wiki/Otsu's_Method\n\n    Examples\n    --------\n    >>> from skimage.data import camera\n    >>> image = camera()\n    >>> thresh = threshold_otsu(image)\n    >>> binary = image <= thresh\n\n    Notes\n    -----\n    The input image must be grayscale.\n    "
    if image is not None and image.ndim > 2 and (image.shape[-1] in (3, 4)):
        warn(f'threshold_otsu is expected to work correctly only for grayscale images; image shape {image.shape} looks like that of an RGB image.')
    if image is not None:
        first_pixel = image.reshape(-1)[0]
        if np.all(image == first_pixel):
            return first_pixel
    (counts, bin_centers) = _validate_image_histogram(image, hist, nbins)
    weight1 = np.cumsum(counts)
    weight2 = np.cumsum(counts[::-1])[::-1]
    mean1 = np.cumsum(counts * bin_centers) / weight1
    mean2 = (np.cumsum((counts * bin_centers)[::-1]) / weight2[::-1])[::-1]
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    idx = np.argmax(variance12)
    threshold = bin_centers[idx]
    return threshold

def threshold_yen(image=None, nbins=256, *, hist=None):
    if False:
        print('Hello World!')
    'Return threshold value based on Yen\'s method.\n    Either image or hist must be provided. In case hist is given, the actual\n    histogram of the image is ignored.\n\n    Parameters\n    ----------\n    image : (M, N[, ...]) ndarray\n        Grayscale input image.\n    nbins : int, optional\n        Number of bins used to calculate histogram. This value is ignored for\n        integer arrays.\n    hist : array, or 2-tuple of arrays, optional\n        Histogram from which to determine the threshold, and optionally a\n        corresponding array of bin center intensities.\n        An alternative use of this function is to pass it only hist.\n\n    Returns\n    -------\n    threshold : float\n        Upper threshold value. All pixels with an intensity higher than\n        this value are assumed to be foreground.\n\n    References\n    ----------\n    .. [1] Yen J.C., Chang F.J., and Chang S. (1995) "A New Criterion\n           for Automatic Multilevel Thresholding" IEEE Trans. on Image\n           Processing, 4(3): 370-378. :DOI:`10.1109/83.366472`\n    .. [2] Sezgin M. and Sankur B. (2004) "Survey over Image Thresholding\n           Techniques and Quantitative Performance Evaluation" Journal of\n           Electronic Imaging, 13(1): 146-165, :DOI:`10.1117/1.1631315`\n           http://www.busim.ee.boun.edu.tr/~sankur/SankurFolder/Threshold_survey.pdf\n    .. [3] ImageJ AutoThresholder code, http://fiji.sc/wiki/index.php/Auto_Threshold\n\n    Examples\n    --------\n    >>> from skimage.data import camera\n    >>> image = camera()\n    >>> thresh = threshold_yen(image)\n    >>> binary = image <= thresh\n    '
    (counts, bin_centers) = _validate_image_histogram(image, hist, nbins)
    if bin_centers.size == 1:
        return bin_centers[0]
    pmf = counts.astype('float32', copy=False) / counts.sum()
    P1 = np.cumsum(pmf)
    P1_sq = np.cumsum(pmf ** 2)
    P2_sq = np.cumsum(pmf[::-1] ** 2)[::-1]
    crit = np.log((P1_sq[:-1] * P2_sq[1:]) ** (-1) * (P1[:-1] * (1.0 - P1[:-1])) ** 2)
    return bin_centers[crit.argmax()]

def threshold_isodata(image=None, nbins=256, return_all=False, *, hist=None):
    if False:
        return 10
    'Return threshold value(s) based on ISODATA method.\n\n    Histogram-based threshold, known as Ridler-Calvard method or inter-means.\n    Threshold values returned satisfy the following equality::\n\n        threshold = (image[image <= threshold].mean() +\n                     image[image > threshold].mean()) / 2.0\n\n    That is, returned thresholds are intensities that separate the image into\n    two groups of pixels, where the threshold intensity is midway between the\n    mean intensities of these groups.\n\n    For integer images, the above equality holds to within one; for floating-\n    point images, the equality holds to within the histogram bin-width.\n\n    Either image or hist must be provided. In case hist is given, the actual\n    histogram of the image is ignored.\n\n    Parameters\n    ----------\n    image : (M, N[, ...]) ndarray\n        Grayscale input image.\n    nbins : int, optional\n        Number of bins used to calculate histogram. This value is ignored for\n        integer arrays.\n    return_all : bool, optional\n        If False (default), return only the lowest threshold that satisfies\n        the above equality. If True, return all valid thresholds.\n    hist : array, or 2-tuple of arrays, optional\n        Histogram to determine the threshold from and a corresponding array\n        of bin center intensities. Alternatively, only the histogram can be\n        passed.\n\n    Returns\n    -------\n    threshold : float or int or array\n        Threshold value(s).\n\n    References\n    ----------\n    .. [1] Ridler, TW & Calvard, S (1978), "Picture thresholding using an\n           iterative selection method"\n           IEEE Transactions on Systems, Man and Cybernetics 8: 630-632,\n           :DOI:`10.1109/TSMC.1978.4310039`\n    .. [2] Sezgin M. and Sankur B. (2004) "Survey over Image Thresholding\n           Techniques and Quantitative Performance Evaluation" Journal of\n           Electronic Imaging, 13(1): 146-165,\n           http://www.busim.ee.boun.edu.tr/~sankur/SankurFolder/Threshold_survey.pdf\n           :DOI:`10.1117/1.1631315`\n    .. [3] ImageJ AutoThresholder code,\n           http://fiji.sc/wiki/index.php/Auto_Threshold\n\n    Examples\n    --------\n    >>> from skimage.data import coins\n    >>> image = coins()\n    >>> thresh = threshold_isodata(image)\n    >>> binary = image > thresh\n    '
    (counts, bin_centers) = _validate_image_histogram(image, hist, nbins)
    if len(bin_centers) == 1:
        if return_all:
            return bin_centers
        else:
            return bin_centers[0]
    counts = counts.astype('float32', copy=False)
    csuml = np.cumsum(counts)
    csumh = csuml[-1] - csuml
    intensity_sum = counts * bin_centers
    csum_intensity = np.cumsum(intensity_sum)
    lower = csum_intensity[:-1] / csuml[:-1]
    higher = (csum_intensity[-1] - csum_intensity[:-1]) / csumh[:-1]
    all_mean = (lower + higher) / 2.0
    bin_width = bin_centers[1] - bin_centers[0]
    distances = all_mean - bin_centers[:-1]
    thresholds = bin_centers[:-1][(distances >= 0) & (distances < bin_width)]
    if return_all:
        return thresholds
    else:
        return thresholds[0]
_DEFAULT_ENTROPY_BINS = tuple(np.arange(-0.5, 255.51, 1))

def _cross_entropy(image, threshold, bins=_DEFAULT_ENTROPY_BINS):
    if False:
        print('Hello World!')
    'Compute cross-entropy between distributions above and below a threshold.\n\n    Parameters\n    ----------\n    image : array\n        The input array of values.\n    threshold : float\n        The value dividing the foreground and background in ``image``.\n    bins : int or array of float, optional\n        The number of bins or the bin edges. (Any valid value to the ``bins``\n        argument of ``np.histogram`` will work here.) For an exact calculation,\n        each unique value should have its own bin. The default value for bins\n        ensures exact handling of uint8 images: ``bins=256`` results in\n        aliasing problems due to bin width not being equal to 1.\n\n    Returns\n    -------\n    nu : float\n        The cross-entropy target value as defined in [1]_.\n\n    Notes\n    -----\n    See Li and Lee, 1993 [1]_; this is the objective function ``threshold_li``\n    minimizes. This function can be improved but this implementation most\n    closely matches equation 8 in [1]_ and equations 1-3 in [2]_.\n\n    References\n    ----------\n    .. [1] Li C.H. and Lee C.K. (1993) "Minimum Cross Entropy Thresholding"\n           Pattern Recognition, 26(4): 617-625\n           :DOI:`10.1016/0031-3203(93)90115-D`\n    .. [2] Li C.H. and Tam P.K.S. (1998) "An Iterative Algorithm for Minimum\n           Cross Entropy Thresholding" Pattern Recognition Letters, 18(8): 771-776\n           :DOI:`10.1016/S0167-8655(98)00057-9`\n    '
    (histogram, bin_edges) = np.histogram(image, bins=bins, density=True)
    bin_centers = np.convolve(bin_edges, [0.5, 0.5], mode='valid')
    t = np.flatnonzero(bin_centers > threshold)[0]
    m0a = np.sum(histogram[:t])
    m0b = np.sum(histogram[t:])
    m1a = np.sum(histogram[:t] * bin_centers[:t])
    m1b = np.sum(histogram[t:] * bin_centers[t:])
    mua = m1a / m0a
    mub = m1b / m0b
    nu = -m1a * np.log(mua) - m1b * np.log(mub)
    return nu

def threshold_li(image, *, tolerance=None, initial_guess=None, iter_callback=None):
    if False:
        return 10
    'Compute threshold value by Li\'s iterative Minimum Cross Entropy method.\n\n    Parameters\n    ----------\n    image : (M, N[, ...]) ndarray\n        Grayscale input image.\n    tolerance : float, optional\n        Finish the computation when the change in the threshold in an iteration\n        is less than this value. By default, this is half the smallest\n        difference between intensity values in ``image``.\n    initial_guess : float or Callable[[array[float]], float], optional\n        Li\'s iterative method uses gradient descent to find the optimal\n        threshold. If the image intensity histogram contains more than two\n        modes (peaks), the gradient descent could get stuck in a local optimum.\n        An initial guess for the iteration can help the algorithm find the\n        globally-optimal threshold. A float value defines a specific start\n        point, while a callable should take in an array of image intensities\n        and return a float value. Example valid callables include\n        ``numpy.mean`` (default), ``lambda arr: numpy.quantile(arr, 0.95)``,\n        or even :func:`skimage.filters.threshold_otsu`.\n    iter_callback : Callable[[float], Any], optional\n        A function that will be called on the threshold at every iteration of\n        the algorithm.\n\n    Returns\n    -------\n    threshold : float\n        Upper threshold value. All pixels with an intensity higher than\n        this value are assumed to be foreground.\n\n    References\n    ----------\n    .. [1] Li C.H. and Lee C.K. (1993) "Minimum Cross Entropy Thresholding"\n           Pattern Recognition, 26(4): 617-625\n           :DOI:`10.1016/0031-3203(93)90115-D`\n    .. [2] Li C.H. and Tam P.K.S. (1998) "An Iterative Algorithm for Minimum\n           Cross Entropy Thresholding" Pattern Recognition Letters, 18(8): 771-776\n           :DOI:`10.1016/S0167-8655(98)00057-9`\n    .. [3] Sezgin M. and Sankur B. (2004) "Survey over Image Thresholding\n           Techniques and Quantitative Performance Evaluation" Journal of\n           Electronic Imaging, 13(1): 146-165\n           :DOI:`10.1117/1.1631315`\n    .. [4] ImageJ AutoThresholder code, http://fiji.sc/wiki/index.php/Auto_Threshold\n\n    Examples\n    --------\n    >>> from skimage.data import camera\n    >>> image = camera()\n    >>> thresh = threshold_li(image)\n    >>> binary = image > thresh\n    '
    image = image[~np.isnan(image)]
    if image.size == 0:
        return np.nan
    if np.all(image == image.flat[0]):
        return image.flat[0]
    image = image[np.isfinite(image)]
    if image.size == 0:
        return 0.0
    image_min = np.min(image)
    image -= image_min
    if image.dtype.kind in 'iu':
        tolerance = tolerance or 0.5
    else:
        tolerance = tolerance or np.min(np.diff(np.unique(image))) / 2
    if initial_guess is None:
        t_next = np.mean(image)
    elif callable(initial_guess):
        t_next = initial_guess(image)
    elif np.isscalar(initial_guess):
        t_next = initial_guess - image_min
        image_max = np.max(image) + image_min
        if not 0 < t_next < np.max(image):
            msg = f'The initial guess for threshold_li must be within the range of the image. Got {initial_guess} for image min {image_min} and max {image_max}.'
            raise ValueError(msg)
    else:
        raise TypeError('Incorrect type for `initial_guess`; should be a floating point value, or a function mapping an array to a floating point value.')
    t_curr = -2 * tolerance
    if iter_callback is not None:
        iter_callback(t_next + image_min)
    if image.dtype.kind in 'iu':
        (hist, bin_centers) = histogram(image.reshape(-1), source_range='image')
        hist = hist.astype('float32', copy=False)
        while abs(t_next - t_curr) > tolerance:
            t_curr = t_next
            foreground = bin_centers > t_curr
            background = ~foreground
            mean_fore = np.average(bin_centers[foreground], weights=hist[foreground])
            mean_back = np.average(bin_centers[background], weights=hist[background])
            if mean_back == 0:
                break
            t_next = (mean_back - mean_fore) / (np.log(mean_back) - np.log(mean_fore))
            if iter_callback is not None:
                iter_callback(t_next + image_min)
    else:
        while abs(t_next - t_curr) > tolerance:
            t_curr = t_next
            foreground = image > t_curr
            mean_fore = np.mean(image[foreground])
            mean_back = np.mean(image[~foreground])
            if mean_back == 0.0:
                break
            t_next = (mean_back - mean_fore) / (np.log(mean_back) - np.log(mean_fore))
            if iter_callback is not None:
                iter_callback(t_next + image_min)
    threshold = t_next + image_min
    return threshold

def threshold_minimum(image=None, nbins=256, max_num_iter=10000, *, hist=None):
    if False:
        while True:
            i = 10
    'Return threshold value based on minimum method.\n\n    The histogram of the input ``image`` is computed if not provided and\n    smoothed until there are only two maxima. Then the minimum in between is\n    the threshold value.\n\n    Either image or hist must be provided. In case hist is given, the actual\n    histogram of the image is ignored.\n\n    Parameters\n    ----------\n    image : (M, N[, ...]) ndarray, optional\n        Grayscale input image.\n    nbins : int, optional\n        Number of bins used to calculate histogram. This value is ignored for\n        integer arrays.\n    max_num_iter : int, optional\n        Maximum number of iterations to smooth the histogram.\n    hist : array, or 2-tuple of arrays, optional\n        Histogram to determine the threshold from and a corresponding array\n        of bin center intensities. Alternatively, only the histogram can be\n        passed.\n\n    Returns\n    -------\n    threshold : float\n        Upper threshold value. All pixels with an intensity higher than\n        this value are assumed to be foreground.\n\n    Raises\n    ------\n    RuntimeError\n        If unable to find two local maxima in the histogram or if the\n        smoothing takes more than 1e4 iterations.\n\n    References\n    ----------\n    .. [1] C. A. Glasbey, "An analysis of histogram-based thresholding\n           algorithms," CVGIP: Graphical Models and Image Processing,\n           vol. 55, pp. 532-537, 1993.\n    .. [2] Prewitt, JMS & Mendelsohn, ML (1966), "The analysis of cell\n           images", Annals of the New York Academy of Sciences 128: 1035-1053\n           :DOI:`10.1111/j.1749-6632.1965.tb11715.x`\n\n    Examples\n    --------\n    >>> from skimage.data import camera\n    >>> image = camera()\n    >>> thresh = threshold_minimum(image)\n    >>> binary = image > thresh\n    '

    def find_local_maxima_idx(hist):
        if False:
            return 10
        maximum_idxs = list()
        direction = 1
        for i in range(hist.shape[0] - 1):
            if direction > 0:
                if hist[i + 1] < hist[i]:
                    direction = -1
                    maximum_idxs.append(i)
            elif hist[i + 1] > hist[i]:
                direction = 1
        return maximum_idxs
    (counts, bin_centers) = _validate_image_histogram(image, hist, nbins)
    smooth_hist = counts.astype('float32', copy=False)
    for counter in range(max_num_iter):
        smooth_hist = ndi.uniform_filter1d(smooth_hist, 3)
        maximum_idxs = find_local_maxima_idx(smooth_hist)
        if len(maximum_idxs) < 3:
            break
    if len(maximum_idxs) != 2:
        raise RuntimeError('Unable to find two maxima in histogram')
    elif counter == max_num_iter - 1:
        raise RuntimeError('Maximum iteration reached for histogramsmoothing')
    threshold_idx = np.argmin(smooth_hist[maximum_idxs[0]:maximum_idxs[1] + 1])
    return bin_centers[maximum_idxs[0] + threshold_idx]

def threshold_mean(image):
    if False:
        return 10
    'Return threshold value based on the mean of grayscale values.\n\n    Parameters\n    ----------\n    image : (M, N[, ...]) ndarray\n        Grayscale input image.\n\n    Returns\n    -------\n    threshold : float\n        Upper threshold value. All pixels with an intensity higher than\n        this value are assumed to be foreground.\n\n    References\n    ----------\n    .. [1] C. A. Glasbey, "An analysis of histogram-based thresholding\n        algorithms," CVGIP: Graphical Models and Image Processing,\n        vol. 55, pp. 532-537, 1993.\n        :DOI:`10.1006/cgip.1993.1040`\n\n    Examples\n    --------\n    >>> from skimage.data import camera\n    >>> image = camera()\n    >>> thresh = threshold_mean(image)\n    >>> binary = image > thresh\n    '
    return np.mean(image)

def threshold_triangle(image, nbins=256):
    if False:
        i = 10
        return i + 15
    'Return threshold value based on the triangle algorithm.\n\n    Parameters\n    ----------\n    image : (M, N[, ...]) ndarray\n        Grayscale input image.\n    nbins : int, optional\n        Number of bins used to calculate histogram. This value is ignored for\n        integer arrays.\n\n    Returns\n    -------\n    threshold : float\n        Upper threshold value. All pixels with an intensity higher than\n        this value are assumed to be foreground.\n\n    References\n    ----------\n    .. [1] Zack, G. W., Rogers, W. E. and Latt, S. A., 1977,\n       Automatic Measurement of Sister Chromatid Exchange Frequency,\n       Journal of Histochemistry and Cytochemistry 25 (7), pp. 741-753\n       :DOI:`10.1177/25.7.70454`\n    .. [2] ImageJ AutoThresholder code,\n       http://fiji.sc/wiki/index.php/Auto_Threshold\n\n    Examples\n    --------\n    >>> from skimage.data import camera\n    >>> image = camera()\n    >>> thresh = threshold_triangle(image)\n    >>> binary = image > thresh\n    '
    (hist, bin_centers) = histogram(image.reshape(-1), nbins, source_range='image')
    nbins = len(hist)
    arg_peak_height = np.argmax(hist)
    peak_height = hist[arg_peak_height]
    (arg_low_level, arg_high_level) = np.flatnonzero(hist)[[0, -1]]
    if arg_low_level == arg_high_level:
        return image.ravel()[0]
    flip = arg_peak_height - arg_low_level < arg_high_level - arg_peak_height
    if flip:
        hist = hist[::-1]
        arg_low_level = nbins - arg_high_level - 1
        arg_peak_height = nbins - arg_peak_height - 1
    del arg_high_level
    width = arg_peak_height - arg_low_level
    x1 = np.arange(width)
    y1 = hist[x1 + arg_low_level]
    norm = np.sqrt(peak_height ** 2 + width ** 2)
    peak_height /= norm
    width /= norm
    length = peak_height * x1 - width * y1
    arg_level = np.argmax(length) + arg_low_level
    if flip:
        arg_level = nbins - arg_level - 1
    return bin_centers[arg_level]

def _mean_std(image, w):
    if False:
        i = 10
        return i + 15
    'Return local mean and standard deviation of each pixel using a\n    neighborhood defined by a rectangular window size ``w``.\n    The algorithm uses integral images to speedup computation. This is\n    used by :func:`threshold_niblack` and :func:`threshold_sauvola`.\n\n    Parameters\n    ----------\n    image : (M, N[, ...]) ndarray\n        Grayscale input image.\n    w : int, or iterable of int\n        Window size specified as a single odd integer (3, 5, 7, …),\n        or an iterable of length ``image.ndim`` containing only odd\n        integers (e.g. ``(1, 5, 5)``).\n\n    Returns\n    -------\n    m : ndarray of float, same shape as ``image``\n        Local mean of the image.\n    s : ndarray of float, same shape as ``image``\n        Local standard deviation of the image.\n\n    References\n    ----------\n    .. [1] F. Shafait, D. Keysers, and T. M. Breuel, "Efficient\n           implementation of local adaptive thresholding techniques\n           using integral images." in Document Recognition and\n           Retrieval XV, (San Jose, USA), Jan. 2008.\n           :DOI:`10.1117/12.767755`\n    '
    if not isinstance(w, Iterable):
        w = (w,) * image.ndim
    _validate_window_size(w)
    float_dtype = _supported_float_type(image.dtype)
    pad_width = tuple(((k // 2 + 1, k // 2) for k in w))
    padded = np.pad(image.astype(float_dtype, copy=False), pad_width, mode='reflect')
    integral = integral_image(padded, dtype=np.float64)
    padded *= padded
    integral_sq = integral_image(padded, dtype=np.float64)
    kernel_indices = list(itertools.product(*tuple([(0, _w) for _w in w])))
    kernel_values = [(-1) ** (image.ndim % 2 != np.sum(indices) % 2) for indices in kernel_indices]
    total_window_size = math.prod(w)
    kernel_shape = tuple((_w + 1 for _w in w))
    m = _correlate_sparse(integral, kernel_shape, kernel_indices, kernel_values)
    m = m.astype(float_dtype, copy=False)
    m /= total_window_size
    g2 = _correlate_sparse(integral_sq, kernel_shape, kernel_indices, kernel_values)
    g2 = g2.astype(float_dtype, copy=False)
    g2 /= total_window_size
    s = np.sqrt(np.clip(g2 - m * m, 0, None))
    return (m, s)

def threshold_niblack(image, window_size=15, k=0.2):
    if False:
        i = 10
        return i + 15
    'Applies Niblack local threshold to an array.\n\n    A threshold T is calculated for every pixel in the image using the\n    following formula::\n\n        T = m(x,y) - k * s(x,y)\n\n    where m(x,y) and s(x,y) are the mean and standard deviation of\n    pixel (x,y) neighborhood defined by a rectangular window with size w\n    times w centered around the pixel. k is a configurable parameter\n    that weights the effect of standard deviation.\n\n    Parameters\n    ----------\n    image : (M, N[, ...]) ndarray\n        Grayscale input image.\n    window_size : int, or iterable of int, optional\n        Window size specified as a single odd integer (3, 5, 7, …),\n        or an iterable of length ``image.ndim`` containing only odd\n        integers (e.g. ``(1, 5, 5)``).\n    k : float, optional\n        Value of parameter k in threshold formula.\n\n    Returns\n    -------\n    threshold : (M, N[, ...]) ndarray\n        Threshold mask. All pixels with an intensity higher than\n        this value are assumed to be foreground.\n\n    Notes\n    -----\n    This algorithm is originally designed for text recognition.\n\n    The Bradley threshold is a particular case of the Niblack\n    one, being equivalent to\n\n    >>> from skimage import data\n    >>> image = data.page()\n    >>> q = 1\n    >>> threshold_image = threshold_niblack(image, k=0) * q\n\n    for some value ``q``. By default, Bradley and Roth use ``q=1``.\n\n\n    References\n    ----------\n    .. [1] W. Niblack, An introduction to Digital Image Processing,\n           Prentice-Hall, 1986.\n    .. [2] D. Bradley and G. Roth, "Adaptive thresholding using Integral\n           Image", Journal of Graphics Tools 12(2), pp. 13-21, 2007.\n           :DOI:`10.1080/2151237X.2007.10129236`\n\n    Examples\n    --------\n    >>> from skimage import data\n    >>> image = data.page()\n    >>> threshold_image = threshold_niblack(image, window_size=7, k=0.1)\n    '
    (m, s) = _mean_std(image, window_size)
    return m - k * s

def threshold_sauvola(image, window_size=15, k=0.2, r=None):
    if False:
        return 10
    'Applies Sauvola local threshold to an array. Sauvola is a\n    modification of Niblack technique.\n\n    In the original method a threshold T is calculated for every pixel\n    in the image using the following formula::\n\n        T = m(x,y) * (1 + k * ((s(x,y) / R) - 1))\n\n    where m(x,y) and s(x,y) are the mean and standard deviation of\n    pixel (x,y) neighborhood defined by a rectangular window with size w\n    times w centered around the pixel. k is a configurable parameter\n    that weights the effect of standard deviation.\n    R is the maximum standard deviation of a grayscale image.\n\n    Parameters\n    ----------\n    image : (M, N[, ...]) ndarray\n        Grayscale input image.\n    window_size : int, or iterable of int, optional\n        Window size specified as a single odd integer (3, 5, 7, …),\n        or an iterable of length ``image.ndim`` containing only odd\n        integers (e.g. ``(1, 5, 5)``).\n    k : float, optional\n        Value of the positive parameter k.\n    r : float, optional\n        Value of R, the dynamic range of standard deviation.\n        If None, set to the half of the image dtype range.\n\n    Returns\n    -------\n    threshold : (M, N[, ...]) ndarray\n        Threshold mask. All pixels with an intensity higher than\n        this value are assumed to be foreground.\n\n    Notes\n    -----\n    This algorithm is originally designed for text recognition.\n\n    References\n    ----------\n    .. [1] J. Sauvola and M. Pietikainen, "Adaptive document image\n           binarization," Pattern Recognition 33(2),\n           pp. 225-236, 2000.\n           :DOI:`10.1016/S0031-3203(99)00055-2`\n\n    Examples\n    --------\n    >>> from skimage import data\n    >>> image = data.page()\n    >>> t_sauvola = threshold_sauvola(image, window_size=15, k=0.2)\n    >>> binary_image = image > t_sauvola\n    '
    if r is None:
        (imin, imax) = dtype_limits(image, clip_negative=False)
        r = 0.5 * (imax - imin)
    (m, s) = _mean_std(image, window_size)
    return m * (1 + k * (s / r - 1))

def apply_hysteresis_threshold(image, low, high):
    if False:
        for i in range(10):
            print('nop')
    'Apply hysteresis thresholding to ``image``.\n\n    This algorithm finds regions where ``image`` is greater than ``high``\n    OR ``image`` is greater than ``low`` *and* that region is connected to\n    a region greater than ``high``.\n\n    Parameters\n    ----------\n    image : (M[, ...]) ndarray\n        Grayscale input image.\n    low : float, or array of same shape as ``image``\n        Lower threshold.\n    high : float, or array of same shape as ``image``\n        Higher threshold.\n\n    Returns\n    -------\n    thresholded : (M[, ...]) array of bool\n        Array in which ``True`` indicates the locations where ``image``\n        was above the hysteresis threshold.\n\n    Examples\n    --------\n    >>> image = np.array([1, 2, 3, 2, 1, 2, 1, 3, 2])\n    >>> apply_hysteresis_threshold(image, 1.5, 2.5).astype(int)\n    array([0, 1, 1, 1, 0, 0, 0, 1, 1])\n\n    References\n    ----------\n    .. [1] J. Canny. A computational approach to edge detection.\n           IEEE Transactions on Pattern Analysis and Machine Intelligence.\n           1986; vol. 8, pp.679-698.\n           :DOI:`10.1109/TPAMI.1986.4767851`\n    '
    low = np.clip(low, a_min=None, a_max=high)
    mask_low = image > low
    mask_high = image > high
    (labels_low, num_labels) = ndi.label(mask_low)
    sums = ndi.sum(mask_high, labels_low, np.arange(num_labels + 1))
    connected_to_high = sums > 0
    thresholded = connected_to_high[labels_low]
    return thresholded

def threshold_multiotsu(image=None, classes=3, nbins=256, *, hist=None):
    if False:
        while True:
            i = 10
    'Generate `classes`-1 threshold values to divide gray levels in `image`,\n    following Otsu\'s method for multiple classes.\n\n    The threshold values are chosen to maximize the total sum of pairwise\n    variances between the thresholded graylevel classes. See Notes and [1]_\n    for more details.\n\n    Either image or hist must be provided. If hist is provided, the actual\n    histogram of the image is ignored.\n\n    Parameters\n    ----------\n    image : (M, N[, ...]) ndarray, optional\n        Grayscale input image.\n    classes : int, optional\n        Number of classes to be thresholded, i.e. the number of resulting\n        regions.\n    nbins : int, optional\n        Number of bins used to calculate the histogram. This value is ignored\n        for integer arrays.\n    hist : array, or 2-tuple of arrays, optional\n        Histogram from which to determine the threshold, and optionally a\n        corresponding array of bin center intensities. If no hist provided,\n        this function will compute it from the image (see notes).\n\n    Returns\n    -------\n    thresh : array\n        Array containing the threshold values for the desired classes.\n\n    Raises\n    ------\n    ValueError\n         If ``image`` contains less grayscale value then the desired\n         number of classes.\n\n    Notes\n    -----\n    This implementation relies on a Cython function whose complexity\n    is :math:`O\\left(\\frac{Ch^{C-1}}{(C-1)!}\\right)`, where :math:`h`\n    is the number of histogram bins and :math:`C` is the number of\n    classes desired.\n\n    If no hist is given, this function will make use of\n    `skimage.exposure.histogram`, which behaves differently than\n    `np.histogram`. While both allowed, use the former for consistent\n    behaviour.\n\n    The input image must be grayscale.\n\n    References\n    ----------\n    .. [1] Liao, P-S., Chen, T-S. and Chung, P-C., "A fast algorithm for\n           multilevel thresholding", Journal of Information Science and\n           Engineering 17 (5): 713-727, 2001. Available at:\n           <https://ftp.iis.sinica.edu.tw/JISE/2001/200109_01.pdf>\n           :DOI:`10.6688/JISE.2001.17.5.1`\n    .. [2] Tosa, Y., "Multi-Otsu Threshold", a java plugin for ImageJ.\n           Available at:\n           <http://imagej.net/plugins/download/Multi_OtsuThreshold.java>\n\n    Examples\n    --------\n    >>> from skimage.color import label2rgb\n    >>> from skimage import data\n    >>> image = data.camera()\n    >>> thresholds = threshold_multiotsu(image)\n    >>> regions = np.digitize(image, bins=thresholds)\n    >>> regions_colorized = label2rgb(regions)\n    '
    if image is not None and image.ndim > 2 and (image.shape[-1] in (3, 4)):
        warn(f'threshold_multiotsu is expected to work correctly only for grayscale images; image shape {image.shape} looks like that of an RGB image.')
    (prob, bin_centers) = _validate_image_histogram(image, hist, nbins, normalize=True)
    prob = prob.astype('float32', copy=False)
    nvalues = np.count_nonzero(prob)
    if nvalues < classes:
        msg = f'After discretization into bins, the input image has only {nvalues} different values. It cannot be thresholded in {classes} classes. If there are more unique values before discretization, try increasing the number of bins (`nbins`).'
        raise ValueError(msg)
    elif nvalues == classes:
        thresh_idx = np.flatnonzero(prob)[:-1]
    else:
        try:
            thresh_idx = _get_multiotsu_thresh_indices_lut(prob, classes - 1)
        except MemoryError:
            thresh_idx = _get_multiotsu_thresh_indices(prob, classes - 1)
    thresh = bin_centers[thresh_idx]
    return thresh