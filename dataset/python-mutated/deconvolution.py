"""Implementations restoration functions"""
import numpy as np
from scipy.signal import convolve
from .._shared.utils import _supported_float_type, deprecate_kwarg
from . import uft

def wiener(image, psf, balance, reg=None, is_real=True, clip=True):
    if False:
        for i in range(10):
            print('nop')
    'Wiener-Hunt deconvolution\n\n    Return the deconvolution with a Wiener-Hunt approach (i.e. with\n    Fourier diagonalisation).\n\n    Parameters\n    ----------\n    image : ndarray\n       Input degraded image (can be n-dimensional).\n    psf : ndarray\n       Point Spread Function. This is assumed to be the impulse\n       response (input image space) if the data-type is real, or the\n       transfer function (Fourier space) if the data-type is\n       complex. There is no constraints on the shape of the impulse\n       response. The transfer function must be of shape\n       `(N1, N2, ..., ND)` if `is_real is True`,\n       `(N1, N2, ..., ND // 2 + 1)` otherwise (see `np.fft.rfftn`).\n    balance : float\n       The regularisation parameter value that tunes the balance\n       between the data adequacy that improve frequency restoration\n       and the prior adequacy that reduce frequency restoration (to\n       avoid noise artifacts).\n    reg : ndarray, optional\n       The regularisation operator. The Laplacian by default. It can\n       be an impulse response or a transfer function, as for the\n       psf. Shape constraint is the same as for the `psf` parameter.\n    is_real : boolean, optional\n       True by default. Specify if ``psf`` and ``reg`` are provided\n       with hermitian hypothesis, that is only half of the frequency\n       plane is provided (due to the redundancy of Fourier transform\n       of real signal). It\'s apply only if ``psf`` and/or ``reg`` are\n       provided as transfer function.  For the hermitian property see\n       ``uft`` module or ``np.fft.rfftn``.\n    clip : boolean, optional\n       True by default. If True, pixel values of the result above 1 or\n       under -1 are thresholded for skimage pipeline compatibility.\n\n    Returns\n    -------\n    im_deconv : (M, N) ndarray\n       The deconvolved image.\n\n    Examples\n    --------\n    >>> from skimage import color, data, restoration\n    >>> img = color.rgb2gray(data.astronaut())\n    >>> from scipy.signal import convolve2d\n    >>> psf = np.ones((5, 5)) / 25\n    >>> img = convolve2d(img, psf, \'same\')\n    >>> rng = np.random.default_rng()\n    >>> img += 0.1 * img.std() * rng.standard_normal(img.shape)\n    >>> deconvolved_img = restoration.wiener(img, psf, 0.1)\n\n    Notes\n    -----\n    This function applies the Wiener filter to a noisy and degraded\n    image by an impulse response (or PSF). If the data model is\n\n    .. math:: y = Hx + n\n\n    where :math:`n` is noise, :math:`H` the PSF and :math:`x` the\n    unknown original image, the Wiener filter is\n\n    .. math::\n       \\hat x = F^\\dagger \\left( |\\Lambda_H|^2 + \\lambda |\\Lambda_D|^2 \\right)^{-1}\n       \\Lambda_H^\\dagger F y\n\n    where :math:`F` and :math:`F^\\dagger` are the Fourier and inverse\n    Fourier transforms respectively, :math:`\\Lambda_H` the transfer\n    function (or the Fourier transform of the PSF, see [Hunt] below)\n    and :math:`\\Lambda_D` the filter to penalize the restored image\n    frequencies (Laplacian by default, that is penalization of high\n    frequency). The parameter :math:`\\lambda` tunes the balance\n    between the data (that tends to increase high frequency, even\n    those coming from noise), and the regularization.\n\n    These methods are then specific to a prior model. Consequently,\n    the application or the true image nature must correspond to the\n    prior model. By default, the prior model (Laplacian) introduce\n    image smoothness or pixel correlation. It can also be interpreted\n    as high-frequency penalization to compensate the instability of\n    the solution with respect to the data (sometimes called noise\n    amplification or "explosive" solution).\n\n    Finally, the use of Fourier space implies a circulant property of\n    :math:`H`, see [2]_.\n\n    References\n    ----------\n    .. [1] François Orieux, Jean-François Giovannelli, and Thomas\n           Rodet, "Bayesian estimation of regularization and point\n           spread function parameters for Wiener-Hunt deconvolution",\n           J. Opt. Soc. Am. A 27, 1593-1607 (2010)\n\n           https://www.osapublishing.org/josaa/abstract.cfm?URI=josaa-27-7-1593\n\n           https://hal.archives-ouvertes.fr/hal-00674508\n\n    .. [2] B. R. Hunt "A matrix theory proof of the discrete\n           convolution theorem", IEEE Trans. on Audio and\n           Electroacoustics, vol. au-19, no. 4, pp. 285-288, dec. 1971\n    '
    if reg is None:
        (reg, _) = uft.laplacian(image.ndim, image.shape, is_real=is_real)
    if not np.iscomplexobj(reg):
        reg = uft.ir2tf(reg, image.shape, is_real=is_real)
    float_type = _supported_float_type(image.dtype)
    image = image.astype(float_type, copy=False)
    psf = psf.real.astype(float_type, copy=False)
    reg = reg.real.astype(float_type, copy=False)
    if psf.shape != reg.shape:
        trans_func = uft.ir2tf(psf, image.shape, is_real=is_real)
    else:
        trans_func = psf
    wiener_filter = np.conj(trans_func) / (np.abs(trans_func) ** 2 + balance * np.abs(reg) ** 2)
    if is_real:
        deconv = uft.uirfftn(wiener_filter * uft.urfftn(image), shape=image.shape)
    else:
        deconv = uft.uifftn(wiener_filter * uft.ufftn(image))
    if clip:
        deconv[deconv > 1] = 1
        deconv[deconv < -1] = -1
    return deconv

@deprecate_kwarg({'random_state': 'rng'}, deprecated_version='0.21', removed_version='0.23')
def unsupervised_wiener(image, psf, reg=None, user_params=None, is_real=True, clip=True, *, rng=None):
    if False:
        for i in range(10):
            print('nop')
    'Unsupervised Wiener-Hunt deconvolution.\n\n    Return the deconvolution with a Wiener-Hunt approach, where the\n    hyperparameters are automatically estimated. The algorithm is a\n    stochastic iterative process (Gibbs sampler) described in the\n    reference below. See also ``wiener`` function.\n\n    Parameters\n    ----------\n    image : (M, N) ndarray\n       The input degraded image.\n    psf : ndarray\n       The impulse response (input image\'s space) or the transfer\n       function (Fourier space). Both are accepted. The transfer\n       function is automatically recognized as being complex\n       (``np.iscomplexobj(psf)``).\n    reg : ndarray, optional\n       The regularisation operator. The Laplacian by default. It can\n       be an impulse response or a transfer function, as for the psf.\n    user_params : dict, optional\n       Dictionary of parameters for the Gibbs sampler. See below.\n    clip : boolean, optional\n       True by default. If true, pixel values of the result above 1 or\n       under -1 are thresholded for skimage pipeline compatibility.\n    rng : {`numpy.random.Generator`, int}, optional\n        Pseudo-random number generator.\n        By default, a PCG64 generator is used (see :func:`numpy.random.default_rng`).\n        If `rng` is an int, it is used to seed the generator.\n\n        .. versionadded:: 0.19\n\n    Returns\n    -------\n    x_postmean : (M, N) ndarray\n       The deconvolved image (the posterior mean).\n    chains : dict\n       The keys ``noise`` and ``prior`` contain the chain list of\n       noise and prior precision respectively.\n\n    Other parameters\n    ----------------\n    The keys of ``user_params`` are:\n\n    threshold : float\n       The stopping criterion: the norm of the difference between to\n       successive approximated solution (empirical mean of object\n       samples, see Notes section). 1e-4 by default.\n    burnin : int\n       The number of sample to ignore to start computation of the\n       mean. 15 by default.\n    min_num_iter : int\n       The minimum number of iterations. 30 by default.\n    max_num_iter : int\n       The maximum number of iterations if ``threshold`` is not\n       satisfied. 200 by default.\n    callback : callable (None by default)\n       A user provided callable to which is passed, if the function\n       exists, the current image sample for whatever purpose. The user\n       can store the sample, or compute other moments than the\n       mean. It has no influence on the algorithm execution and is\n       only for inspection.\n\n    Examples\n    --------\n    >>> from skimage import color, data, restoration\n    >>> img = color.rgb2gray(data.astronaut())\n    >>> from scipy.signal import convolve2d\n    >>> psf = np.ones((5, 5)) / 25\n    >>> img = convolve2d(img, psf, \'same\')\n    >>> rng = np.random.default_rng()\n    >>> img += 0.1 * img.std() * rng.standard_normal(img.shape)\n    >>> deconvolved_img = restoration.unsupervised_wiener(img, psf)\n\n    Notes\n    -----\n    The estimated image is design as the posterior mean of a\n    probability law (from a Bayesian analysis). The mean is defined as\n    a sum over all the possible images weighted by their respective\n    probability. Given the size of the problem, the exact sum is not\n    tractable. This algorithm use of MCMC to draw image under the\n    posterior law. The practical idea is to only draw highly probable\n    images since they have the biggest contribution to the mean. At the\n    opposite, the less probable images are drawn less often since\n    their contribution is low. Finally, the empirical mean of these\n    samples give us an estimation of the mean, and an exact\n    computation with an infinite sample set.\n\n    References\n    ----------\n    .. [1] François Orieux, Jean-François Giovannelli, and Thomas\n           Rodet, "Bayesian estimation of regularization and point\n           spread function parameters for Wiener-Hunt deconvolution",\n           J. Opt. Soc. Am. A 27, 1593-1607 (2010)\n\n           https://www.osapublishing.org/josaa/abstract.cfm?URI=josaa-27-7-1593\n\n           https://hal.archives-ouvertes.fr/hal-00674508\n    '
    params = {'threshold': 0.0001, 'max_num_iter': 200, 'min_num_iter': 30, 'burnin': 15, 'callback': None}
    params.update(user_params or {})
    if reg is None:
        (reg, _) = uft.laplacian(image.ndim, image.shape, is_real=is_real)
    if not np.iscomplexobj(reg):
        reg = uft.ir2tf(reg, image.shape, is_real=is_real)
    float_type = _supported_float_type(image.dtype)
    image = image.astype(float_type, copy=False)
    psf = psf.real.astype(float_type, copy=False)
    reg = reg.real.astype(float_type, copy=False)
    if psf.shape != reg.shape:
        trans_fct = uft.ir2tf(psf, image.shape, is_real=is_real)
    else:
        trans_fct = psf
    x_postmean = np.zeros(trans_fct.shape, dtype=float_type)
    prev_x_postmean = np.zeros(trans_fct.shape, dtype=float_type)
    delta = np.nan
    (gn_chain, gx_chain) = ([1], [1])
    areg2 = np.abs(reg) ** 2
    atf2 = np.abs(trans_fct) ** 2
    if is_real:
        data_spectrum = uft.urfft2(image)
    else:
        data_spectrum = uft.ufft2(image)
    rng = np.random.default_rng(rng)
    for iteration in range(params['max_num_iter']):
        precision = gn_chain[-1] * atf2 + gx_chain[-1] * areg2
        _rand1 = rng.standard_normal(data_spectrum.shape)
        _rand1 = _rand1.astype(float_type, copy=False)
        _rand2 = rng.standard_normal(data_spectrum.shape)
        _rand2 = _rand2.astype(float_type, copy=False)
        excursion = np.sqrt(0.5 / precision) * (_rand1 + 1j * _rand2)
        wiener_filter = gn_chain[-1] * np.conj(trans_fct) / precision
        x_sample = wiener_filter * data_spectrum + excursion
        if params['callback']:
            params['callback'](x_sample)
        gn_chain.append(rng.gamma(image.size / 2, 2 / uft.image_quad_norm(data_spectrum - x_sample * trans_fct)))
        gx_chain.append(rng.gamma((image.size - 1) / 2, 2 / uft.image_quad_norm(x_sample * reg)))
        if iteration > params['burnin']:
            x_postmean = prev_x_postmean + x_sample
        if iteration > params['burnin'] + 1:
            current = x_postmean / (iteration - params['burnin'])
            previous = prev_x_postmean / (iteration - params['burnin'] - 1)
            delta = np.sum(np.abs(current - previous)) / np.sum(np.abs(x_postmean)) / (iteration - params['burnin'])
        prev_x_postmean = x_postmean
        if iteration > params['min_num_iter'] and delta < params['threshold']:
            break
    x_postmean = x_postmean / (iteration - params['burnin'])
    if is_real:
        x_postmean = uft.uirfft2(x_postmean, shape=image.shape)
    else:
        x_postmean = uft.uifft2(x_postmean)
    if clip:
        x_postmean[x_postmean > 1] = 1
        x_postmean[x_postmean < -1] = -1
    return (x_postmean, {'noise': gn_chain, 'prior': gx_chain})

def richardson_lucy(image, psf, num_iter=50, clip=True, filter_epsilon=None):
    if False:
        while True:
            i = 10
    "Richardson-Lucy deconvolution.\n\n    Parameters\n    ----------\n    image : ndarray\n       Input degraded image (can be n-dimensional).\n    psf : ndarray\n       The point spread function.\n    num_iter : int, optional\n       Number of iterations. This parameter plays the role of\n       regularisation.\n    clip : boolean, optional\n       True by default. If true, pixel value of the result above 1 or\n       under -1 are thresholded for skimage pipeline compatibility.\n    filter_epsilon: float, optional\n       Value below which intermediate results become 0 to avoid division\n       by small numbers.\n\n    Returns\n    -------\n    im_deconv : ndarray\n       The deconvolved image.\n\n    Examples\n    --------\n    >>> from skimage import img_as_float, data, restoration\n    >>> camera = img_as_float(data.camera())\n    >>> from scipy.signal import convolve2d\n    >>> psf = np.ones((5, 5)) / 25\n    >>> camera = convolve2d(camera, psf, 'same')\n    >>> rng = np.random.default_rng()\n    >>> camera += 0.1 * camera.std() * rng.standard_normal(camera.shape)\n    >>> deconvolved = restoration.richardson_lucy(camera, psf, 5)\n\n    References\n    ----------\n    .. [1] https://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution\n    "
    float_type = _supported_float_type(image.dtype)
    image = image.astype(float_type, copy=False)
    psf = psf.astype(float_type, copy=False)
    im_deconv = np.full(image.shape, 0.5, dtype=float_type)
    psf_mirror = np.flip(psf)
    eps = 1e-12
    for _ in range(num_iter):
        conv = convolve(im_deconv, psf, mode='same') + eps
        if filter_epsilon:
            relative_blur = np.where(conv < filter_epsilon, 0, image / conv)
        else:
            relative_blur = image / conv
        im_deconv *= convolve(relative_blur, psf_mirror, mode='same')
    if clip:
        im_deconv[im_deconv > 1] = 1
        im_deconv[im_deconv < -1] = -1
    return im_deconv