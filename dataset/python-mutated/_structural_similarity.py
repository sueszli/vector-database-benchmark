import functools
import numpy as np
from scipy.ndimage import uniform_filter
from .._shared import utils
from .._shared.filters import gaussian
from .._shared.utils import _supported_float_type, check_shape_equality, warn
from ..util.arraycrop import crop
from ..util.dtype import dtype_range
__all__ = ['structural_similarity']

def structural_similarity(im1, im2, *, win_size=None, gradient=False, data_range=None, channel_axis=None, gaussian_weights=False, full=False, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Compute the mean structural similarity index between two images.\n    Please pay attention to the `data_range` parameter with floating-point images.\n\n    Parameters\n    ----------\n    im1, im2 : ndarray\n        Images. Any dimensionality with same shape.\n    win_size : int or None, optional\n        The side-length of the sliding window used in comparison. Must be an\n        odd value. If `gaussian_weights` is True, this is ignored and the\n        window size will depend on `sigma`.\n    gradient : bool, optional\n        If True, also return the gradient with respect to im2.\n    data_range : float, optional\n        The data range of the input image (distance between minimum and\n        maximum possible values). By default, this is estimated from the image\n        data type. This estimate may be wrong for floating-point image data.\n        Therefore it is recommended to always pass this value explicitly\n        (see note below).\n    channel_axis : int or None, optional\n        If None, the image is assumed to be a grayscale (single channel) image.\n        Otherwise, this parameter indicates which axis of the array corresponds\n        to channels.\n\n        .. versionadded:: 0.19\n           ``channel_axis`` was added in 0.19.\n    gaussian_weights : bool, optional\n        If True, each patch has its mean and variance spatially weighted by a\n        normalized Gaussian kernel of width sigma=1.5.\n    full : bool, optional\n        If True, also return the full structural similarity image.\n\n    Other Parameters\n    ----------------\n    use_sample_covariance : bool\n        If True, normalize covariances by N-1 rather than, N where N is the\n        number of pixels within the sliding window.\n    K1 : float\n        Algorithm parameter, K1 (small constant, see [1]_).\n    K2 : float\n        Algorithm parameter, K2 (small constant, see [1]_).\n    sigma : float\n        Standard deviation for the Gaussian when `gaussian_weights` is True.\n\n    Returns\n    -------\n    mssim : float\n        The mean structural similarity index over the image.\n    grad : ndarray\n        The gradient of the structural similarity between im1 and im2 [2]_.\n        This is only returned if `gradient` is set to True.\n    S : ndarray\n        The full SSIM image.  This is only returned if `full` is set to True.\n\n    Notes\n    -----\n    If `data_range` is not specified, the range is automatically guessed\n    based on the image data type. However for floating-point image data, this\n    estimate yields a result double the value of the desired range, as the\n    `dtype_range` in `skimage.util.dtype.py` has defined intervals from -1 to\n    +1. This yields an estimate of 2, instead of 1, which is most often\n    required when working with image data (as negative light intentsities are\n    nonsensical). In case of working with YCbCr-like color data, note that\n    these ranges are different per channel (Cb and Cr have double the range\n    of Y), so one cannot calculate a channel-averaged SSIM with a single call\n    to this function, as identical ranges are assumed for each channel.\n\n    To match the implementation of Wang et al. [1]_, set `gaussian_weights`\n    to True, `sigma` to 1.5, `use_sample_covariance` to False, and\n    specify the `data_range` argument.\n\n    .. versionchanged:: 0.16\n        This function was renamed from ``skimage.measure.compare_ssim`` to\n        ``skimage.metrics.structural_similarity``.\n\n    References\n    ----------\n    .. [1] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P.\n       (2004). Image quality assessment: From error visibility to\n       structural similarity. IEEE Transactions on Image Processing,\n       13, 600-612.\n       https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf,\n       :DOI:`10.1109/TIP.2003.819861`\n\n    .. [2] Avanaki, A. N. (2009). Exact global histogram specification\n       optimized for structural similarity. Optical Review, 16, 613-621.\n       :arxiv:`0901.0065`\n       :DOI:`10.1007/s10043-009-0119-z`\n\n    '
    check_shape_equality(im1, im2)
    float_type = _supported_float_type(im1.dtype)
    if channel_axis is not None:
        args = dict(win_size=win_size, gradient=gradient, data_range=data_range, channel_axis=None, gaussian_weights=gaussian_weights, full=full)
        args.update(kwargs)
        nch = im1.shape[channel_axis]
        mssim = np.empty(nch, dtype=float_type)
        if gradient:
            G = np.empty(im1.shape, dtype=float_type)
        if full:
            S = np.empty(im1.shape, dtype=float_type)
        channel_axis = channel_axis % im1.ndim
        _at = functools.partial(utils.slice_at_axis, axis=channel_axis)
        for ch in range(nch):
            ch_result = structural_similarity(im1[_at(ch)], im2[_at(ch)], **args)
            if gradient and full:
                (mssim[ch], G[_at(ch)], S[_at(ch)]) = ch_result
            elif gradient:
                (mssim[ch], G[_at(ch)]) = ch_result
            elif full:
                (mssim[ch], S[_at(ch)]) = ch_result
            else:
                mssim[ch] = ch_result
        mssim = mssim.mean()
        if gradient and full:
            return (mssim, G, S)
        elif gradient:
            return (mssim, G)
        elif full:
            return (mssim, S)
        else:
            return mssim
    K1 = kwargs.pop('K1', 0.01)
    K2 = kwargs.pop('K2', 0.03)
    sigma = kwargs.pop('sigma', 1.5)
    if K1 < 0:
        raise ValueError('K1 must be positive')
    if K2 < 0:
        raise ValueError('K2 must be positive')
    if sigma < 0:
        raise ValueError('sigma must be positive')
    use_sample_covariance = kwargs.pop('use_sample_covariance', True)
    if gaussian_weights:
        truncate = 3.5
    if win_size is None:
        if gaussian_weights:
            r = int(truncate * sigma + 0.5)
            win_size = 2 * r + 1
        else:
            win_size = 7
    if np.any(np.asarray(im1.shape) - win_size < 0):
        raise ValueError('win_size exceeds image extent. Either ensure that your images are at least 7x7; or pass win_size explicitly in the function call, with an odd value less than or equal to the smaller side of your images. If your images are multichannel (with color channels), set channel_axis to the axis number corresponding to the channels.')
    if not win_size % 2 == 1:
        raise ValueError('Window size must be odd.')
    if data_range is None:
        if np.issubdtype(im1.dtype, np.floating) or np.issubdtype(im2.dtype, np.floating):
            raise ValueError('Since image dtype is floating point, you must specify the data_range parameter. Please read the documentation carefully (including the note). It is recommended that you always specify the data_range anyway.')
        if im1.dtype != im2.dtype:
            warn('Inputs have mismatched dtypes. Setting data_range based on im1.dtype.', stacklevel=2)
        (dmin, dmax) = dtype_range[im1.dtype.type]
        data_range = dmax - dmin
        if np.issubdtype(im1.dtype, np.integer) and im1.dtype != np.uint8:
            warn('Setting data_range based on im1.dtype. ' + f'data_range = {data_range:.0f}. ' + 'Please specify data_range explicitly to avoid mistakes.', stacklevel=2)
    ndim = im1.ndim
    if gaussian_weights:
        filter_func = gaussian
        filter_args = {'sigma': sigma, 'truncate': truncate, 'mode': 'reflect'}
    else:
        filter_func = uniform_filter
        filter_args = {'size': win_size}
    im1 = im1.astype(float_type, copy=False)
    im2 = im2.astype(float_type, copy=False)
    NP = win_size ** ndim
    if use_sample_covariance:
        cov_norm = NP / (NP - 1)
    else:
        cov_norm = 1.0
    ux = filter_func(im1, **filter_args)
    uy = filter_func(im2, **filter_args)
    uxx = filter_func(im1 * im1, **filter_args)
    uyy = filter_func(im2 * im2, **filter_args)
    uxy = filter_func(im1 * im2, **filter_args)
    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)
    R = data_range
    C1 = (K1 * R) ** 2
    C2 = (K2 * R) ** 2
    (A1, A2, B1, B2) = (2 * ux * uy + C1, 2 * vxy + C2, ux ** 2 + uy ** 2 + C1, vx + vy + C2)
    D = B1 * B2
    S = A1 * A2 / D
    pad = (win_size - 1) // 2
    mssim = crop(S, pad).mean(dtype=np.float64)
    if gradient:
        grad = filter_func(A1 / D, **filter_args) * im1
        grad += filter_func(-S / B2, **filter_args) * im2
        grad += filter_func((ux * (A2 - A1) - uy * (B2 - B1) * S) / D, **filter_args)
        grad *= 2 / im1.size
        if full:
            return (mssim, grad, S)
        else:
            return (mssim, grad)
    elif full:
        return (mssim, S)
    else:
        return mssim