import functools
from math import ceil
import numbers
import scipy.stats
import numpy as np
from ..util.dtype import img_as_float
from .._shared import utils
from .._shared.utils import _supported_float_type, warn
from ._denoise_cy import _denoise_bilateral, _denoise_tv_bregman
from .. import color
from ..color.colorconv import ycbcr_from_rgb

def _gaussian_weight(array, sigma_squared, *, dtype=float):
    if False:
        for i in range(10):
            print('nop')
    'Helping function. Define a Gaussian weighting from array and\n    sigma_square.\n\n    Parameters\n    ----------\n    array : ndarray\n        Input array.\n    sigma_squared : float\n        The squared standard deviation used in the filter.\n    dtype : data type object, optional (default : float)\n        The type and size of the data to be returned.\n\n    Returns\n    -------\n    gaussian : ndarray\n        The input array filtered by the Gaussian.\n    '
    return np.exp(-0.5 * (array ** 2 / sigma_squared), dtype=dtype)

def _compute_color_lut(bins, sigma, max_value, *, dtype=float):
    if False:
        i = 10
        return i + 15
    'Helping function. Define a lookup table containing Gaussian filter\n    values using the color distance sigma.\n\n    Parameters\n    ----------\n    bins : int\n        Number of discrete values for Gaussian weights of color filtering.\n        A larger value results in improved accuracy.\n    sigma : float\n        Standard deviation for grayvalue/color distance (radiometric\n        similarity). A larger value results in averaging of pixels with larger\n        radiometric differences. Note, that the image will be converted using\n        the `img_as_float` function and thus the standard deviation is in\n        respect to the range ``[0, 1]``. If the value is ``None`` the standard\n        deviation of the ``image`` will be used.\n    max_value : float\n        Maximum value of the input image.\n    dtype : data type object, optional (default : float)\n        The type and size of the data to be returned.\n\n    Returns\n    -------\n    color_lut : ndarray\n        Lookup table for the color distance sigma.\n    '
    values = np.linspace(0, max_value, bins, endpoint=False)
    return _gaussian_weight(values, sigma ** 2, dtype=dtype)

def _compute_spatial_lut(win_size, sigma, *, dtype=float):
    if False:
        while True:
            i = 10
    'Helping function. Define a lookup table containing Gaussian filter\n    values using the spatial sigma.\n\n    Parameters\n    ----------\n    win_size : int\n        Window size for filtering.\n        If win_size is not specified, it is calculated as\n        ``max(5, 2 * ceil(3 * sigma_spatial) + 1)``.\n    sigma : float\n        Standard deviation for range distance. A larger value results in\n        averaging of pixels with larger spatial differences.\n    dtype : data type object\n        The type and size of the data to be returned.\n\n    Returns\n    -------\n    spatial_lut : ndarray\n        Lookup table for the spatial sigma.\n    '
    grid_points = np.arange(-win_size // 2, win_size // 2 + 1)
    (rr, cc) = np.meshgrid(grid_points, grid_points, indexing='ij')
    distances = np.hypot(rr, cc)
    return _gaussian_weight(distances, sigma ** 2, dtype=dtype).ravel()

@utils.channel_as_last_axis()
def denoise_bilateral(image, win_size=None, sigma_color=None, sigma_spatial=1, bins=10000, mode='constant', cval=0, *, channel_axis=None):
    if False:
        print('Hello World!')
    'Denoise image using bilateral filter.\n\n    Parameters\n    ----------\n    image : ndarray, shape (M, N[, 3])\n        Input image, 2D grayscale or RGB.\n    win_size : int\n        Window size for filtering.\n        If win_size is not specified, it is calculated as\n        ``max(5, 2 * ceil(3 * sigma_spatial) + 1)``.\n    sigma_color : float\n        Standard deviation for grayvalue/color distance (radiometric\n        similarity). A larger value results in averaging of pixels with larger\n        radiometric differences. If ``None``, the standard deviation of\n        ``image`` will be used.\n    sigma_spatial : float\n        Standard deviation for range distance. A larger value results in\n        averaging of pixels with larger spatial differences.\n    bins : int\n        Number of discrete values for Gaussian weights of color filtering.\n        A larger value results in improved accuracy.\n    mode : {\'constant\', \'edge\', \'symmetric\', \'reflect\', \'wrap\'}\n        How to handle values outside the image borders. See\n        `numpy.pad` for detail.\n    cval : string\n        Used in conjunction with mode \'constant\', the value outside\n        the image boundaries.\n    channel_axis : int or None, optional\n        If ``None``, the image is assumed to be grayscale (single-channel).\n        Otherwise, this parameter indicates which axis of the array corresponds\n        to channels.\n\n        .. versionadded:: 0.19\n           ``channel_axis`` was added in 0.19.\n\n    Returns\n    -------\n    denoised : ndarray\n        Denoised image.\n\n    Notes\n    -----\n    This is an edge-preserving, denoising filter. It averages pixels based on\n    their spatial closeness and radiometric similarity [1]_.\n\n    Spatial closeness is measured by the Gaussian function of the Euclidean\n    distance between two pixels and a certain standard deviation\n    (`sigma_spatial`).\n\n    Radiometric similarity is measured by the Gaussian function of the\n    Euclidean distance between two color values and a certain standard\n    deviation (`sigma_color`).\n\n    Note that, if the image is of any `int` dtype, ``image`` will be\n    converted using the `img_as_float` function and thus the standard\n    deviation (`sigma_color`) will be in range ``[0, 1]``.\n\n    For more information on scikit-image\'s data type conversions and how\n    images are rescaled in these conversions,\n    see: https://scikit-image.org/docs/stable/user_guide/data_types.html.\n\n    References\n    ----------\n    .. [1] C. Tomasi and R. Manduchi. "Bilateral Filtering for Gray and Color\n           Images." IEEE International Conference on Computer Vision (1998)\n           839-846. :DOI:`10.1109/ICCV.1998.710815`\n\n    Examples\n    --------\n    >>> from skimage import data, img_as_float\n    >>> astro = img_as_float(data.astronaut())\n    >>> astro = astro[220:300, 220:320]\n    >>> rng = np.random.default_rng()\n    >>> noisy = astro + 0.6 * astro.std() * rng.random(astro.shape)\n    >>> noisy = np.clip(noisy, 0, 1)\n    >>> denoised = denoise_bilateral(noisy, sigma_color=0.05, sigma_spatial=15,\n    ...                              channel_axis=-1)\n    '
    if channel_axis is not None:
        if image.ndim != 3:
            if image.ndim == 2:
                raise ValueError('Use ``channel_axis=None`` for 2D grayscale images. The last axis of the input image must be multiple color channels not another spatial dimension.')
            else:
                raise ValueError(f'Bilateral filter is only implemented for 2D grayscale images (image.ndim == 2) and 2D multichannel (image.ndim == 3) images, but the input image has {image.ndim} dimensions.')
        elif image.shape[2] not in (3, 4):
            if image.shape[2] > 4:
                msg = f'The last axis of the input image is interpreted as channels. Input image with shape {image.shape} has {image.shape[2]} channels in last axis. ``denoise_bilateral``is implemented for 2D grayscale and color images only.'
                warn(msg)
            else:
                msg = f'Input image must be grayscale, RGB, or RGBA; but has shape {image.shape}.'
                warn(msg)
    elif image.ndim > 2:
        raise ValueError(f'Bilateral filter is not implemented for grayscale images of 3 or more dimensions, but input image has {image.shape} shape. Use ``channel_axis=-1`` for 2D RGB images.')
    if win_size is None:
        win_size = max(5, 2 * int(ceil(3 * sigma_spatial)) + 1)
    min_value = image.min()
    max_value = image.max()
    if min_value == max_value:
        return image
    image = np.atleast_3d(img_as_float(image))
    image = np.ascontiguousarray(image)
    sigma_color = sigma_color or image.std()
    color_lut = _compute_color_lut(bins, sigma_color, max_value, dtype=image.dtype)
    range_lut = _compute_spatial_lut(win_size, sigma_spatial, dtype=image.dtype)
    out = np.empty(image.shape, dtype=image.dtype)
    dims = image.shape[2]
    empty_dims = np.empty(dims, dtype=image.dtype)
    if min_value < 0:
        image = image - min_value
        max_value -= min_value
    _denoise_bilateral(image, max_value, win_size, sigma_color, sigma_spatial, bins, mode, cval, color_lut, range_lut, empty_dims, out)
    out = np.squeeze(out)
    if min_value < 0:
        out += min_value
    return out

@utils.channel_as_last_axis()
def denoise_tv_bregman(image, weight=5.0, max_num_iter=100, eps=0.001, isotropic=True, *, channel_axis=None):
    if False:
        return 10
    'Perform total variation denoising using split-Bregman optimization.\n\n    Given :math:`f`, a noisy image (input data),\n    total variation denoising (also known as total variation regularization)\n    aims to find an image :math:`u` with less total variation than :math:`f`,\n    under the constraint that :math:`u` remain similar to :math:`f`.\n    This can be expressed by the Rudin--Osher--Fatemi (ROF) minimization\n    problem:\n\n    .. math::\n\n        \\min_{u} \\sum_{i=0}^{N-1} \\left( \\left| \\nabla{u_i} \\right| + \\frac{\\lambda}{2}(f_i - u_i)^2 \\right)\n\n    where :math:`\\lambda` is a positive parameter.\n    The first term of this cost function is the total variation;\n    the second term represents data fidelity. As :math:`\\lambda \\to 0`,\n    the total variation term dominates, forcing the solution to have smaller\n    total variation, at the expense of looking less like the input data.\n\n    This code is an implementation of the split Bregman algorithm of Goldstein\n    and Osher to solve the ROF problem ([1]_, [2]_, [3]_).\n\n    Parameters\n    ----------\n    image : ndarray\n        Input image to be denoised (converted using :func:`~.img_as_float`).\n    weight : float, optional\n        Denoising weight. It is equal to :math:`\\frac{\\lambda}{2}`. Therefore,\n        the smaller the `weight`, the more denoising (at\n        the expense of less similarity to `image`).\n    eps : float, optional\n        Tolerance :math:`\\varepsilon > 0` for the stop criterion:\n        The algorithm stops when :math:`\\|u_n - u_{n-1}\\|_2 < \\varepsilon`.\n    max_num_iter : int, optional\n        Maximal number of iterations used for the optimization.\n    isotropic : boolean, optional\n        Switch between isotropic and anisotropic TV denoising.\n    channel_axis : int or None, optional\n        If ``None``, the image is assumed to be grayscale (single-channel).\n        Otherwise, this parameter indicates which axis of the array corresponds\n        to channels.\n\n        .. versionadded:: 0.19\n           ``channel_axis`` was added in 0.19.\n\n    Returns\n    -------\n    u : ndarray\n        Denoised image.\n\n    Notes\n    -----\n    Ensure that `channel_axis` parameter is set appropriately for color\n    images.\n\n    The principle of total variation denoising is explained in [4]_.\n    It is about minimizing the total variation of an image,\n    which can be roughly described as\n    the integral of the norm of the image gradient. Total variation\n    denoising tends to produce cartoon-like images, that is,\n    piecewise-constant images.\n\n    See Also\n    --------\n    denoise_tv_chambolle : Perform total variation denoising in nD.\n\n    References\n    ----------\n    .. [1] Tom Goldstein and Stanley Osher, "The Split Bregman Method For L1\n           Regularized Problems",\n           https://ww3.math.ucla.edu/camreport/cam08-29.pdf\n    .. [2] Pascal Getreuer, "Rudin–Osher–Fatemi Total Variation Denoising\n           using Split Bregman" in Image Processing On Line on 2012–05–19,\n           https://www.ipol.im/pub/art/2012/g-tvd/article_lr.pdf\n    .. [3] https://web.math.ucsb.edu/~cgarcia/UGProjects/BregmanAlgorithms_JacquelineBush.pdf\n    .. [4] https://en.wikipedia.org/wiki/Total_variation_denoising\n\n    '
    image = np.atleast_3d(img_as_float(image))
    rows = image.shape[0]
    cols = image.shape[1]
    dims = image.shape[2]
    shape_ext = (rows + 2, cols + 2, dims)
    out = np.zeros(shape_ext, image.dtype)
    if channel_axis is not None:
        channel_out = np.zeros(shape_ext[:2] + (1,), dtype=out.dtype)
        for c in range(image.shape[-1]):
            channel_in = np.ascontiguousarray(image[..., c:c + 1])
            _denoise_tv_bregman(channel_in, image.dtype.type(weight), max_num_iter, eps, isotropic, channel_out)
            out[..., c] = channel_out[..., 0]
    else:
        image = np.ascontiguousarray(image)
        _denoise_tv_bregman(image, image.dtype.type(weight), max_num_iter, eps, isotropic, out)
    return np.squeeze(out[1:-1, 1:-1])

def _denoise_tv_chambolle_nd(image, weight=0.1, eps=0.0002, max_num_iter=200):
    if False:
        for i in range(10):
            print('nop')
    'Perform total-variation denoising on n-dimensional images.\n\n    Parameters\n    ----------\n    image : ndarray\n        n-D input data to be denoised.\n    weight : float, optional\n        Denoising weight. The greater `weight`, the more denoising (at\n        the expense of fidelity to `input`).\n    eps : float, optional\n        Relative difference of the value of the cost function that determines\n        the stop criterion. The algorithm stops when:\n\n            (E_(n-1) - E_n) < eps * E_0\n\n    max_num_iter : int, optional\n        Maximal number of iterations used for the optimization.\n\n    Returns\n    -------\n    out : ndarray\n        Denoised array of floats.\n\n    Notes\n    -----\n    Rudin, Osher and Fatemi algorithm.\n    '
    ndim = image.ndim
    p = np.zeros((image.ndim,) + image.shape, dtype=image.dtype)
    g = np.zeros_like(p)
    d = np.zeros_like(image)
    i = 0
    while i < max_num_iter:
        if i > 0:
            d = -p.sum(0)
            slices_d = [slice(None)] * ndim
            slices_p = [slice(None)] * (ndim + 1)
            for ax in range(ndim):
                slices_d[ax] = slice(1, None)
                slices_p[ax + 1] = slice(0, -1)
                slices_p[0] = ax
                d[tuple(slices_d)] += p[tuple(slices_p)]
                slices_d[ax] = slice(None)
                slices_p[ax + 1] = slice(None)
            out = image + d
        else:
            out = image
        E = (d ** 2).sum()
        slices_g = [slice(None)] * (ndim + 1)
        for ax in range(ndim):
            slices_g[ax + 1] = slice(0, -1)
            slices_g[0] = ax
            g[tuple(slices_g)] = np.diff(out, axis=ax)
            slices_g[ax + 1] = slice(None)
        norm = np.sqrt((g ** 2).sum(axis=0))[np.newaxis, ...]
        E += weight * norm.sum()
        tau = 1.0 / (2.0 * ndim)
        norm *= tau / weight
        norm += 1.0
        p -= tau * g
        p /= norm
        E /= float(image.size)
        if i == 0:
            E_init = E
            E_previous = E
        elif np.abs(E_previous - E) < eps * E_init:
            break
        else:
            E_previous = E
        i += 1
    return out

def denoise_tv_chambolle(image, weight=0.1, eps=0.0002, max_num_iter=200, *, channel_axis=None):
    if False:
        return 10
    'Perform total variation denoising in nD.\n\n    Given :math:`f`, a noisy image (input data),\n    total variation denoising (also known as total variation regularization)\n    aims to find an image :math:`u` with less total variation than :math:`f`,\n    under the constraint that :math:`u` remain similar to :math:`f`.\n    This can be expressed by the Rudin--Osher--Fatemi (ROF) minimization\n    problem:\n\n    .. math::\n\n        \\min_{u} \\sum_{i=0}^{N-1} \\left( \\left| \\nabla{u_i} \\right| + \\frac{\\lambda}{2}(f_i - u_i)^2 \\right)\n\n    where :math:`\\lambda` is a positive parameter.\n    The first term of this cost function is the total variation;\n    the second term represents data fidelity. As :math:`\\lambda \\to 0`,\n    the total variation term dominates, forcing the solution to have smaller\n    total variation, at the expense of looking less like the input data.\n\n    This code is an implementation of the algorithm proposed by Chambolle\n    in [1]_ to solve the ROF problem.\n\n    Parameters\n    ----------\n    image : ndarray\n        Input image to be denoised. If its dtype is not float, it gets\n        converted with :func:`~.img_as_float`.\n    weight : float, optional\n        Denoising weight. It is equal to :math:`\\frac{1}{\\lambda}`. Therefore,\n        the greater the `weight`, the more denoising (at the expense of\n        fidelity to `image`).\n    eps : float, optional\n        Tolerance :math:`\\varepsilon > 0` for the stop criterion (compares to\n        absolute value of relative difference of the cost function :math:`E`):\n        The algorithm stops when :math:`|E_{n-1} - E_n| < \\varepsilon * E_0`.\n    max_num_iter : int, optional\n        Maximal number of iterations used for the optimization.\n    channel_axis : int or None, optional\n        If ``None``, the image is assumed to be grayscale (single-channel).\n        Otherwise, this parameter indicates which axis of the array corresponds\n        to channels.\n\n        .. versionadded:: 0.19\n           ``channel_axis`` was added in 0.19.\n\n    Returns\n    -------\n    u : ndarray\n        Denoised image.\n\n    Notes\n    -----\n    Make sure to set the `channel_axis` parameter appropriately for color\n    images.\n\n    The principle of total variation denoising is explained in [2]_.\n    It is about minimizing the total variation of an image,\n    which can be roughly described as\n    the integral of the norm of the image gradient. Total variation\n    denoising tends to produce cartoon-like images, that is,\n    piecewise-constant images.\n\n    See Also\n    --------\n    denoise_tv_bregman : Perform total variation denoising using split-Bregman\n        optimization.\n\n    References\n    ----------\n    .. [1] A. Chambolle, An algorithm for total variation minimization and\n           applications, Journal of Mathematical Imaging and Vision,\n           Springer, 2004, 20, 89-97.\n    .. [2] https://en.wikipedia.org/wiki/Total_variation_denoising\n\n    Examples\n    --------\n    2D example on astronaut image:\n\n    >>> from skimage import color, data\n    >>> img = color.rgb2gray(data.astronaut())[:50, :50]\n    >>> rng = np.random.default_rng()\n    >>> img += 0.5 * img.std() * rng.standard_normal(img.shape)\n    >>> denoised_img = denoise_tv_chambolle(img, weight=60)\n\n    3D example on synthetic data:\n\n    >>> x, y, z = np.ogrid[0:20, 0:20, 0:20]\n    >>> mask = (x - 22)**2 + (y - 20)**2 + (z - 17)**2 < 8**2\n    >>> mask = mask.astype(float)\n    >>> rng = np.random.default_rng()\n    >>> mask += 0.2 * rng.standard_normal(mask.shape)\n    >>> res = denoise_tv_chambolle(mask, weight=100)\n\n    '
    im_type = image.dtype
    if not im_type.kind == 'f':
        image = img_as_float(image)
    float_dtype = _supported_float_type(image.dtype)
    image = image.astype(float_dtype, copy=False)
    if channel_axis is not None:
        channel_axis = channel_axis % image.ndim
        _at = functools.partial(utils.slice_at_axis, axis=channel_axis)
        out = np.zeros_like(image)
        for c in range(image.shape[channel_axis]):
            out[_at(c)] = _denoise_tv_chambolle_nd(image[_at(c)], weight, eps, max_num_iter)
    else:
        out = _denoise_tv_chambolle_nd(image, weight, eps, max_num_iter)
    return out

def _bayes_thresh(details, var):
    if False:
        for i in range(10):
            print('nop')
    'BayesShrink threshold for a zero-mean details coeff array.'
    dvar = np.mean(details * details)
    eps = np.finfo(details.dtype).eps
    thresh = var / np.sqrt(max(dvar - var, eps))
    return thresh

def _universal_thresh(img, sigma):
    if False:
        i = 10
        return i + 15
    'Universal threshold used by the VisuShrink method'
    return sigma * np.sqrt(2 * np.log(img.size))

def _sigma_est_dwt(detail_coeffs, distribution='Gaussian'):
    if False:
        while True:
            i = 10
    'Calculate the robust median estimator of the noise standard deviation.\n\n    Parameters\n    ----------\n    detail_coeffs : ndarray\n        The detail coefficients corresponding to the discrete wavelet\n        transform of an image.\n    distribution : str\n        The underlying noise distribution.\n\n    Returns\n    -------\n    sigma : float\n        The estimated noise standard deviation (see section 4.2 of [1]_).\n\n    References\n    ----------\n    .. [1] D. L. Donoho and I. M. Johnstone. "Ideal spatial adaptation\n       by wavelet shrinkage." Biometrika 81.3 (1994): 425-455.\n       :DOI:`10.1093/biomet/81.3.425`\n    '
    detail_coeffs = detail_coeffs[np.nonzero(detail_coeffs)]
    if distribution.lower() == 'gaussian':
        denom = scipy.stats.norm.ppf(0.75)
        sigma = np.median(np.abs(detail_coeffs)) / denom
    else:
        raise ValueError('Only Gaussian noise estimation is currently supported')
    return sigma

def _wavelet_threshold(image, wavelet, method=None, threshold=None, sigma=None, mode='soft', wavelet_levels=None):
    if False:
        i = 10
        return i + 15
    'Perform wavelet thresholding.\n\n    Parameters\n    ----------\n    image : ndarray (2d or 3d) of ints, uints or floats\n        Input data to be denoised. `image` can be of any numeric type,\n        but it is cast into an ndarray of floats for the computation\n        of the denoised image.\n    wavelet : string\n        The type of wavelet to perform. Can be any of the options\n        pywt.wavelist outputs. For example, this may be any of ``{db1, db2,\n        db3, db4, haar}``.\n    method : {\'BayesShrink\', \'VisuShrink\'}, optional\n        Thresholding method to be used. The currently supported methods are\n        "BayesShrink" [1]_ and "VisuShrink" [2]_. If it is set to None, a\n        user-specified ``threshold`` must be supplied instead.\n    threshold : float, optional\n        The thresholding value to apply during wavelet coefficient\n        thresholding. The default value (None) uses the selected ``method`` to\n        estimate appropriate threshold(s) for noise removal.\n    sigma : float, optional\n        The standard deviation of the noise. The noise is estimated when sigma\n        is None (the default) by the method in [2]_.\n    mode : {\'soft\', \'hard\'}, optional\n        An optional argument to choose the type of denoising performed. It\n        noted that choosing soft thresholding given additive noise finds the\n        best approximation of the original image.\n    wavelet_levels : int or None, optional\n        The number of wavelet decomposition levels to use.  The default is\n        three less than the maximum number of possible decomposition levels\n        (see Notes below).\n\n    Returns\n    -------\n    out : ndarray\n        Denoised image.\n\n    References\n    ----------\n    .. [1] Chang, S. Grace, Bin Yu, and Martin Vetterli. "Adaptive wavelet\n           thresholding for image denoising and compression." Image Processing,\n           IEEE Transactions on 9.9 (2000): 1532-1546.\n           :DOI:`10.1109/83.862633`\n    .. [2] D. L. Donoho and I. M. Johnstone. "Ideal spatial adaptation\n           by wavelet shrinkage." Biometrika 81.3 (1994): 425-455.\n           :DOI:`10.1093/biomet/81.3.425`\n    '
    try:
        import pywt
    except ImportError:
        raise ImportError('PyWavelets is not installed. Please ensure it is installed in order to use this function.')
    wavelet = pywt.Wavelet(wavelet)
    if not wavelet.orthogonal:
        warn(f'Wavelet thresholding was designed for use with orthogonal wavelets. For nonorthogonal wavelets such as {wavelet.name},results are likely to be suboptimal.')
    original_extent = tuple((slice(s) for s in image.shape))
    if wavelet_levels is None:
        wavelet_levels = pywt.dwtn_max_level(image.shape, wavelet)
        wavelet_levels = max(wavelet_levels - 3, 1)
    coeffs = pywt.wavedecn(image, wavelet=wavelet, level=wavelet_levels)
    dcoeffs = coeffs[1:]
    if sigma is None:
        detail_coeffs = dcoeffs[-1]['d' * image.ndim]
        sigma = _sigma_est_dwt(detail_coeffs, distribution='Gaussian')
    if method is not None and threshold is not None:
        warn(f'Thresholding method {method} selected. The user-specified threshold will be ignored.')
    if threshold is None:
        var = sigma ** 2
        if method is None:
            raise ValueError('If method is None, a threshold must be provided.')
        elif method == 'BayesShrink':
            threshold = [{key: _bayes_thresh(level[key], var) for key in level} for level in dcoeffs]
        elif method == 'VisuShrink':
            threshold = _universal_thresh(image, sigma)
        else:
            raise ValueError(f'Unrecognized method: {method}')
    if np.isscalar(threshold):
        denoised_detail = [{key: pywt.threshold(level[key], value=threshold, mode=mode) for key in level} for level in dcoeffs]
    else:
        denoised_detail = [{key: pywt.threshold(level[key], value=thresh[key], mode=mode) for key in level} for (thresh, level) in zip(threshold, dcoeffs)]
    denoised_coeffs = [coeffs[0]] + denoised_detail
    return pywt.waverecn(denoised_coeffs, wavelet)[original_extent]

def _scale_sigma_and_image_consistently(image, sigma, multichannel, rescale_sigma):
    if False:
        while True:
            i = 10
    'If the ``image`` is rescaled, also rescale ``sigma`` consistently.\n\n    Images that are not floating point will be rescaled via ``img_as_float``.\n    Half-precision images will be promoted to single precision.\n    '
    if multichannel:
        if isinstance(sigma, numbers.Number) or sigma is None:
            sigma = [sigma] * image.shape[-1]
        elif len(sigma) != image.shape[-1]:
            raise ValueError('When channel_axis is not None, sigma must be a scalar or have length equal to the number of channels')
    if image.dtype.kind != 'f':
        if rescale_sigma:
            range_pre = image.max() - image.min()
        image = img_as_float(image)
        if rescale_sigma:
            range_post = image.max() - image.min()
            scale_factor = range_post / range_pre
            if multichannel:
                sigma = [s * scale_factor if s is not None else s for s in sigma]
            elif sigma is not None:
                sigma *= scale_factor
    elif image.dtype == np.float16:
        image = image.astype(np.float32)
    return (image, sigma)

def _rescale_sigma_rgb2ycbcr(sigmas):
    if False:
        while True:
            i = 10
    'Convert user-provided noise standard deviations to YCbCr space.\n\n    Notes\n    -----\n    If R, G, B are linearly independent random variables and a1, a2, a3 are\n    scalars, then random variable C:\n        C = a1 * R + a2 * G + a3 * B\n    has variance, var_C, given by:\n        var_C = a1**2 * var_R + a2**2 * var_G + a3**2 * var_B\n    '
    if sigmas[0] is None:
        return sigmas
    sigmas = np.asarray(sigmas)
    rgv_variances = sigmas * sigmas
    for i in range(3):
        scalars = ycbcr_from_rgb[i, :]
        var_channel = np.sum(scalars * scalars * rgv_variances)
        sigmas[i] = np.sqrt(var_channel)
    return sigmas

@utils.channel_as_last_axis()
def denoise_wavelet(image, sigma=None, wavelet='db1', mode='soft', wavelet_levels=None, convert2ycbcr=False, method='BayesShrink', rescale_sigma=True, *, channel_axis=None):
    if False:
        while True:
            i = 10
    'Perform wavelet denoising on an image.\n\n    Parameters\n    ----------\n    image : ndarray (M[, N[, ...P]][, C]) of ints, uints or floats\n        Input data to be denoised. `image` can be of any numeric type,\n        but it is cast into an ndarray of floats for the computation\n        of the denoised image.\n    sigma : float or list, optional\n        The noise standard deviation used when computing the wavelet detail\n        coefficient threshold(s). When None (default), the noise standard\n        deviation is estimated via the method in [2]_.\n    wavelet : string, optional\n        The type of wavelet to perform and can be any of the options\n        ``pywt.wavelist`` outputs. The default is `\'db1\'`. For example,\n        ``wavelet`` can be any of ``{\'db2\', \'haar\', \'sym9\'}`` and many more.\n    mode : {\'soft\', \'hard\'}, optional\n        An optional argument to choose the type of denoising performed. It\n        noted that choosing soft thresholding given additive noise finds the\n        best approximation of the original image.\n    wavelet_levels : int or None, optional\n        The number of wavelet decomposition levels to use.  The default is\n        three less than the maximum number of possible decomposition levels.\n    convert2ycbcr : bool, optional\n        If True and channel_axis is set, do the wavelet denoising in the YCbCr\n        colorspace instead of the RGB color space. This typically results in\n        better performance for RGB images.\n    method : {\'BayesShrink\', \'VisuShrink\'}, optional\n        Thresholding method to be used. The currently supported methods are\n        "BayesShrink" [1]_ and "VisuShrink" [2]_. Defaults to "BayesShrink".\n    rescale_sigma : bool, optional\n        If False, no rescaling of the user-provided ``sigma`` will be\n        performed. The default of ``True`` rescales sigma appropriately if the\n        image is rescaled internally.\n\n        .. versionadded:: 0.16\n           ``rescale_sigma`` was introduced in 0.16\n    channel_axis : int or None, optional\n        If ``None``, the image is assumed to be grayscale (single-channel).\n        Otherwise, this parameter indicates which axis of the array corresponds\n        to channels.\n\n        .. versionadded:: 0.19\n           ``channel_axis`` was added in 0.19.\n\n    Returns\n    -------\n    out : ndarray\n        Denoised image.\n\n    Notes\n    -----\n    The wavelet domain is a sparse representation of the image, and can be\n    thought of similarly to the frequency domain of the Fourier transform.\n    Sparse representations have most values zero or near-zero and truly random\n    noise is (usually) represented by many small values in the wavelet domain.\n    Setting all values below some threshold to 0 reduces the noise in the\n    image, but larger thresholds also decrease the detail present in the image.\n\n    If the input is 3D, this function performs wavelet denoising on each color\n    plane separately.\n\n    .. versionchanged:: 0.16\n       For floating point inputs, the original input range is maintained and\n       there is no clipping applied to the output. Other input types will be\n       converted to a floating point value in the range [-1, 1] or [0, 1]\n       depending on the input image range. Unless ``rescale_sigma = False``,\n       any internal rescaling applied to the ``image`` will also be applied\n       to ``sigma`` to maintain the same relative amplitude.\n\n    Many wavelet coefficient thresholding approaches have been proposed. By\n    default, ``denoise_wavelet`` applies BayesShrink, which is an adaptive\n    thresholding method that computes separate thresholds for each wavelet\n    sub-band as described in [1]_.\n\n    If ``method == "VisuShrink"``, a single "universal threshold" is applied to\n    all wavelet detail coefficients as described in [2]_. This threshold\n    is designed to remove all Gaussian noise at a given ``sigma`` with high\n    probability, but tends to produce images that appear overly smooth.\n\n    Although any of the wavelets from ``PyWavelets`` can be selected, the\n    thresholding methods assume an orthogonal wavelet transform and may not\n    choose the threshold appropriately for biorthogonal wavelets. Orthogonal\n    wavelets are desirable because white noise in the input remains white noise\n    in the subbands. Biorthogonal wavelets lead to colored noise in the\n    subbands. Additionally, the orthogonal wavelets in PyWavelets are\n    orthonormal so that noise variance in the subbands remains identical to the\n    noise variance of the input. Example orthogonal wavelets are the Daubechies\n    (e.g. \'db2\') or symmlet (e.g. \'sym2\') families.\n\n    References\n    ----------\n    .. [1] Chang, S. Grace, Bin Yu, and Martin Vetterli. "Adaptive wavelet\n           thresholding for image denoising and compression." Image Processing,\n           IEEE Transactions on 9.9 (2000): 1532-1546.\n           :DOI:`10.1109/83.862633`\n    .. [2] D. L. Donoho and I. M. Johnstone. "Ideal spatial adaptation\n           by wavelet shrinkage." Biometrika 81.3 (1994): 425-455.\n           :DOI:`10.1093/biomet/81.3.425`\n\n    Examples\n    --------\n    >>> import pytest\n    >>> _ = pytest.importorskip(\'pywt\')\n    >>> from skimage import color, data\n    >>> img = img_as_float(data.astronaut())\n    >>> img = color.rgb2gray(img)\n    >>> rng = np.random.default_rng()\n    >>> img += 0.1 * rng.standard_normal(img.shape)\n    >>> img = np.clip(img, 0, 1)\n    >>> denoised_img = denoise_wavelet(img, sigma=0.1, rescale_sigma=True)\n\n    '
    multichannel = channel_axis is not None
    if method not in ['BayesShrink', 'VisuShrink']:
        raise ValueError(f'Invalid method: {method}. The currently supported methods are "BayesShrink" and "VisuShrink".')
    clip_output = image.dtype.kind != 'f'
    if convert2ycbcr and (not multichannel):
        raise ValueError('convert2ycbcr requires channel_axis to be set')
    (image, sigma) = _scale_sigma_and_image_consistently(image, sigma, multichannel, rescale_sigma)
    if multichannel:
        if convert2ycbcr:
            out = color.rgb2ycbcr(image)
            if rescale_sigma:
                sigma = _rescale_sigma_rgb2ycbcr(sigma)
            for i in range(3):
                (_min, _max) = (out[..., i].min(), out[..., i].max())
                scale_factor = _max - _min
                if scale_factor == 0:
                    continue
                channel = out[..., i] - _min
                channel /= scale_factor
                sigma_channel = sigma[i]
                if sigma_channel is not None:
                    sigma_channel /= scale_factor
                out[..., i] = denoise_wavelet(channel, wavelet=wavelet, method=method, sigma=sigma_channel, mode=mode, wavelet_levels=wavelet_levels, rescale_sigma=rescale_sigma)
                out[..., i] = out[..., i] * scale_factor
                out[..., i] += _min
            out = color.ycbcr2rgb(out)
        else:
            out = np.empty_like(image)
            for c in range(image.shape[-1]):
                out[..., c] = _wavelet_threshold(image[..., c], wavelet=wavelet, method=method, sigma=sigma[c], mode=mode, wavelet_levels=wavelet_levels)
    else:
        out = _wavelet_threshold(image, wavelet=wavelet, method=method, sigma=sigma, mode=mode, wavelet_levels=wavelet_levels)
    if clip_output:
        clip_range = (-1, 1) if image.min() < 0 else (0, 1)
        out = np.clip(out, *clip_range, out=out)
    return out

def estimate_sigma(image, average_sigmas=False, *, channel_axis=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Robust wavelet-based estimator of the (Gaussian) noise standard deviation.\n\n    Parameters\n    ----------\n    image : ndarray\n        Image for which to estimate the noise standard deviation.\n    average_sigmas : bool, optional\n        If true, average the channel estimates of `sigma`.  Otherwise return\n        a list of sigmas corresponding to each channel.\n    channel_axis : int or None, optional\n        If ``None``, the image is assumed to be grayscale (single-channel).\n        Otherwise, this parameter indicates which axis of the array corresponds\n        to channels.\n\n        .. versionadded:: 0.19\n           ``channel_axis`` was added in 0.19.\n\n    Returns\n    -------\n    sigma : float or list\n        Estimated noise standard deviation(s).  If `multichannel` is True and\n        `average_sigmas` is False, a separate noise estimate for each channel\n        is returned.  Otherwise, the average of the individual channel\n        estimates is returned.\n\n    Notes\n    -----\n    This function assumes the noise follows a Gaussian distribution. The\n    estimation algorithm is based on the median absolute deviation of the\n    wavelet detail coefficients as described in section 4.2 of [1]_.\n\n    References\n    ----------\n    .. [1] D. L. Donoho and I. M. Johnstone. "Ideal spatial adaptation\n       by wavelet shrinkage." Biometrika 81.3 (1994): 425-455.\n       :DOI:`10.1093/biomet/81.3.425`\n\n    Examples\n    --------\n    >>> import pytest\n    >>> _ = pytest.importorskip(\'pywt\')\n    >>> import skimage.data\n    >>> from skimage import img_as_float\n    >>> img = img_as_float(skimage.data.camera())\n    >>> sigma = 0.1\n    >>> rng = np.random.default_rng()\n    >>> img = img + sigma * rng.standard_normal(img.shape)\n    >>> sigma_hat = estimate_sigma(img, channel_axis=None)\n    '
    try:
        import pywt
    except ImportError:
        raise ImportError('PyWavelets is not installed. Please ensure it is installed in order to use this function.')
    if channel_axis is not None:
        channel_axis = channel_axis % image.ndim
        _at = functools.partial(utils.slice_at_axis, axis=channel_axis)
        nchannels = image.shape[channel_axis]
        sigmas = [estimate_sigma(image[_at(c)], channel_axis=None) for c in range(nchannels)]
        if average_sigmas:
            sigmas = np.mean(sigmas)
        return sigmas
    elif image.shape[-1] <= 4:
        msg = f'image is size {image.shape[-1]} on the last axis, but channel_axis is None. If this is a color image, please set channel_axis=-1 for proper noise estimation.'
        warn(msg)
    coeffs = pywt.dwtn(image, wavelet='db2')
    detail_coeffs = coeffs['d' * image.ndim]
    return _sigma_est_dwt(detail_coeffs, distribution='Gaussian')