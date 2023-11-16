import numpy as np
from scipy.stats import entropy
from ..util.dtype import dtype_range
from .._shared.utils import _supported_float_type, check_shape_equality, warn
__all__ = ['mean_squared_error', 'normalized_root_mse', 'peak_signal_noise_ratio', 'normalized_mutual_information']

def _as_floats(image0, image1):
    if False:
        for i in range(10):
            print('nop')
    '\n    Promote im1, im2 to nearest appropriate floating point precision.\n    '
    float_type = _supported_float_type((image0.dtype, image1.dtype))
    image0 = np.asarray(image0, dtype=float_type)
    image1 = np.asarray(image1, dtype=float_type)
    return (image0, image1)

def mean_squared_error(image0, image1):
    if False:
        for i in range(10):
            print('nop')
    '\n    Compute the mean-squared error between two images.\n\n    Parameters\n    ----------\n    image0, image1 : ndarray\n        Images.  Any dimensionality, must have same shape.\n\n    Returns\n    -------\n    mse : float\n        The mean-squared error (MSE) metric.\n\n    Notes\n    -----\n    .. versionchanged:: 0.16\n        This function was renamed from ``skimage.measure.compare_mse`` to\n        ``skimage.metrics.mean_squared_error``.\n\n    '
    check_shape_equality(image0, image1)
    (image0, image1) = _as_floats(image0, image1)
    return np.mean((image0 - image1) ** 2, dtype=np.float64)

def normalized_root_mse(image_true, image_test, *, normalization='euclidean'):
    if False:
        print('Hello World!')
    "\n    Compute the normalized root mean-squared error (NRMSE) between two\n    images.\n\n    Parameters\n    ----------\n    image_true : ndarray\n        Ground-truth image, same shape as im_test.\n    image_test : ndarray\n        Test image.\n    normalization : {'euclidean', 'min-max', 'mean'}, optional\n        Controls the normalization method to use in the denominator of the\n        NRMSE.  There is no standard method of normalization across the\n        literature [1]_.  The methods available here are as follows:\n\n        - 'euclidean' : normalize by the averaged Euclidean norm of\n          ``im_true``::\n\n              NRMSE = RMSE * sqrt(N) / || im_true ||\n\n          where || . || denotes the Frobenius norm and ``N = im_true.size``.\n          This result is equivalent to::\n\n              NRMSE = || im_true - im_test || / || im_true ||.\n\n        - 'min-max'   : normalize by the intensity range of ``im_true``.\n        - 'mean'      : normalize by the mean of ``im_true``\n\n    Returns\n    -------\n    nrmse : float\n        The NRMSE metric.\n\n    Notes\n    -----\n    .. versionchanged:: 0.16\n        This function was renamed from ``skimage.measure.compare_nrmse`` to\n        ``skimage.metrics.normalized_root_mse``.\n\n    References\n    ----------\n    .. [1] https://en.wikipedia.org/wiki/Root-mean-square_deviation\n\n    "
    check_shape_equality(image_true, image_test)
    (image_true, image_test) = _as_floats(image_true, image_test)
    normalization = normalization.lower()
    if normalization == 'euclidean':
        denom = np.sqrt(np.mean(image_true * image_true, dtype=np.float64))
    elif normalization == 'min-max':
        denom = image_true.max() - image_true.min()
    elif normalization == 'mean':
        denom = image_true.mean()
    else:
        raise ValueError('Unsupported norm_type')
    return np.sqrt(mean_squared_error(image_true, image_test)) / denom

def peak_signal_noise_ratio(image_true, image_test, *, data_range=None):
    if False:
        while True:
            i = 10
    '\n    Compute the peak signal to noise ratio (PSNR) for an image.\n\n    Parameters\n    ----------\n    image_true : ndarray\n        Ground-truth image, same shape as im_test.\n    image_test : ndarray\n        Test image.\n    data_range : int, optional\n        The data range of the input image (distance between minimum and\n        maximum possible values).  By default, this is estimated from the image\n        data-type.\n\n    Returns\n    -------\n    psnr : float\n        The PSNR metric.\n\n    Notes\n    -----\n    .. versionchanged:: 0.16\n        This function was renamed from ``skimage.measure.compare_psnr`` to\n        ``skimage.metrics.peak_signal_noise_ratio``.\n\n    References\n    ----------\n    .. [1] https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio\n\n    '
    check_shape_equality(image_true, image_test)
    if data_range is None:
        if image_true.dtype != image_test.dtype:
            warn('Inputs have mismatched dtype.  Setting data_range based on image_true.')
        (dmin, dmax) = dtype_range[image_true.dtype.type]
        (true_min, true_max) = (np.min(image_true), np.max(image_true))
        if true_max > dmax or true_min < dmin:
            raise ValueError('image_true has intensity values outside the range expected for its data type. Please manually specify the data_range.')
        if true_min >= 0:
            data_range = dmax
        else:
            data_range = dmax - dmin
    (image_true, image_test) = _as_floats(image_true, image_test)
    err = mean_squared_error(image_true, image_test)
    return 10 * np.log10(data_range ** 2 / err)

def _pad_to(arr, shape):
    if False:
        return 10
    'Pad an array with trailing zeros to a given target shape.\n\n    Parameters\n    ----------\n    arr : ndarray\n        The input array.\n    shape : tuple\n        The target shape.\n\n    Returns\n    -------\n    padded : ndarray\n        The padded array.\n\n    Examples\n    --------\n    >>> _pad_to(np.ones((1, 1), dtype=int), (1, 3))\n    array([[1, 0, 0]])\n    '
    if not all((s >= i for (s, i) in zip(shape, arr.shape))):
        raise ValueError(f'Target shape {shape} cannot be smaller than inputshape {arr.shape} along any axis.')
    padding = [(0, s - i) for (s, i) in zip(shape, arr.shape)]
    return np.pad(arr, pad_width=padding, mode='constant', constant_values=0)

def normalized_mutual_information(image0, image1, *, bins=100):
    if False:
        for i in range(10):
            print('nop')
    "Compute the normalized mutual information (NMI).\n\n    The normalized mutual information of :math:`A` and :math:`B` is given by::\n\n    .. math::\n\n        Y(A, B) = \\frac{H(A) + H(B)}{H(A, B)}\n\n    where :math:`H(X) := - \\sum_{x \\in X}{x \\log x}` is the entropy.\n\n    It was proposed to be useful in registering images by Colin Studholme and\n    colleagues [1]_. It ranges from 1 (perfectly uncorrelated image values)\n    to 2 (perfectly correlated image values, whether positively or negatively).\n\n    Parameters\n    ----------\n    image0, image1 : ndarray\n        Images to be compared. The two input images must have the same number\n        of dimensions.\n    bins : int or sequence of int, optional\n        The number of bins along each axis of the joint histogram.\n\n    Returns\n    -------\n    nmi : float\n        The normalized mutual information between the two arrays, computed at\n        the granularity given by ``bins``. Higher NMI implies more similar\n        input images.\n\n    Raises\n    ------\n    ValueError\n        If the images don't have the same number of dimensions.\n\n    Notes\n    -----\n    If the two input images are not the same shape, the smaller image is padded\n    with zeros.\n\n    References\n    ----------\n    .. [1] C. Studholme, D.L.G. Hill, & D.J. Hawkes (1999). An overlap\n           invariant entropy measure of 3D medical image alignment.\n           Pattern Recognition 32(1):71-86\n           :DOI:`10.1016/S0031-3203(98)00091-0`\n    "
    if image0.ndim != image1.ndim:
        raise ValueError(f'NMI requires images of same number of dimensions. Got {image0.ndim}D for `image0` and {image1.ndim}D for `image1`.')
    if image0.shape != image1.shape:
        max_shape = np.maximum(image0.shape, image1.shape)
        padded0 = _pad_to(image0, max_shape)
        padded1 = _pad_to(image1, max_shape)
    else:
        (padded0, padded1) = (image0, image1)
    (hist, bin_edges) = np.histogramdd([np.reshape(padded0, -1), np.reshape(padded1, -1)], bins=bins, density=True)
    H0 = entropy(np.sum(hist, axis=0))
    H1 = entropy(np.sum(hist, axis=1))
    H01 = entropy(np.reshape(hist, -1))
    return (H0 + H1) / H01