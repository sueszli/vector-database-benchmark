"""
Ridge filters.

Ridge filters can be used to detect continuous edges, such as vessels,
neurites, wrinkles, rivers, and other tube-like structures. The present
class of ridge filters relies on the eigenvalues of the Hessian matrix of
image intensities to detect tube-like structures where the intensity changes
perpendicular but not along the structure.
"""
from warnings import warn
import numpy as np
from scipy import linalg
from .._shared.utils import _supported_float_type, check_nD
from ..feature.corner import hessian_matrix, hessian_matrix_eigvals

def meijering(image, sigmas=range(1, 10, 2), alpha=None, black_ridges=True, mode='reflect', cval=0):
    if False:
        return 10
    "\n    Filter an image with the Meijering neuriteness filter.\n\n    This filter can be used to detect continuous ridges, e.g. neurites,\n    wrinkles, rivers. It can be used to calculate the fraction of the\n    whole image containing such objects.\n\n    Calculates the eigenvalues of the Hessian to compute the similarity of\n    an image region to neurites, according to the method described in [1]_.\n\n    Parameters\n    ----------\n    image : (M, N[, ...]) ndarray\n        Array with input image data.\n    sigmas : iterable of floats, optional\n        Sigmas used as scales of filter\n    alpha : float, optional\n        Shaping filter constant, that selects maximally flat elongated\n        features.  The default, None, selects the optimal value -1/(ndim+1).\n    black_ridges : boolean, optional\n        When True (the default), the filter detects black ridges; when\n        False, it detects white ridges.\n    mode : {'constant', 'reflect', 'wrap', 'nearest', 'mirror'}, optional\n        How to handle values outside the image borders.\n    cval : float, optional\n        Used in conjunction with mode 'constant', the value outside\n        the image boundaries.\n\n    Returns\n    -------\n    out : (M, N[, ...]) ndarray\n        Filtered image (maximum of pixels across all scales).\n\n    See also\n    --------\n    sato\n    frangi\n    hessian\n\n    References\n    ----------\n    .. [1] Meijering, E., Jacob, M., Sarria, J. C., Steiner, P., Hirling, H.,\n        Unser, M. (2004). Design and validation of a tool for neurite tracing\n        and analysis in fluorescence microscopy images. Cytometry Part A,\n        58(2), 167-176.\n        :DOI:`10.1002/cyto.a.20022`\n    "
    image = image.astype(_supported_float_type(image.dtype), copy=False)
    if not black_ridges:
        image = -image
    if alpha is None:
        alpha = 1 / (image.ndim + 1)
    mtx = linalg.circulant([1, *[alpha] * (image.ndim - 1)]).astype(image.dtype)
    filtered_max = np.zeros_like(image)
    for sigma in sigmas:
        eigvals = hessian_matrix_eigvals(hessian_matrix(image, sigma, mode=mode, cval=cval, use_gaussian_derivatives=True))
        vals = np.tensordot(mtx, eigvals, 1)
        vals = np.take_along_axis(vals, abs(vals).argmax(0)[None], 0).squeeze(0)
        vals = np.maximum(vals, 0)
        max_val = vals.max()
        if max_val > 0:
            vals /= max_val
        filtered_max = np.maximum(filtered_max, vals)
    return filtered_max

def sato(image, sigmas=range(1, 10, 2), black_ridges=True, mode='reflect', cval=0):
    if False:
        i = 10
        return i + 15
    "\n    Filter an image with the Sato tubeness filter.\n\n    This filter can be used to detect continuous ridges, e.g. tubes,\n    wrinkles, rivers. It can be used to calculate the fraction of the\n    whole image containing such objects.\n\n    Defined only for 2-D and 3-D images. Calculates the eigenvalues of the\n    Hessian to compute the similarity of an image region to tubes, according to\n    the method described in [1]_.\n\n    Parameters\n    ----------\n    image : (M, N[, P]) ndarray\n        Array with input image data.\n    sigmas : iterable of floats, optional\n        Sigmas used as scales of filter.\n    black_ridges : boolean, optional\n        When True (the default), the filter detects black ridges; when\n        False, it detects white ridges.\n    mode : {'constant', 'reflect', 'wrap', 'nearest', 'mirror'}, optional\n        How to handle values outside the image borders.\n    cval : float, optional\n        Used in conjunction with mode 'constant', the value outside\n        the image boundaries.\n\n    Returns\n    -------\n    out : (M, N[, P]) ndarray\n        Filtered image (maximum of pixels across all scales).\n\n    See also\n    --------\n    meijering\n    frangi\n    hessian\n\n    References\n    ----------\n    .. [1] Sato, Y., Nakajima, S., Shiraga, N., Atsumi, H., Yoshida, S.,\n        Koller, T., ..., Kikinis, R. (1998). Three-dimensional multi-scale line\n        filter for segmentation and visualization of curvilinear structures in\n        medical images. Medical image analysis, 2(2), 143-168.\n        :DOI:`10.1016/S1361-8415(98)80009-1`\n    "
    check_nD(image, [2, 3])
    image = image.astype(_supported_float_type(image.dtype), copy=False)
    if not black_ridges:
        image = -image
    filtered_max = np.zeros_like(image)
    for sigma in sigmas:
        eigvals = hessian_matrix_eigvals(hessian_matrix(image, sigma, mode=mode, cval=cval, use_gaussian_derivatives=True))
        eigvals = eigvals[:-1]
        vals = sigma ** 2 * np.prod(np.maximum(eigvals, 0), 0) ** (1 / len(eigvals))
        filtered_max = np.maximum(filtered_max, vals)
    return filtered_max

def frangi(image, sigmas=range(1, 10, 2), scale_range=None, scale_step=None, alpha=0.5, beta=0.5, gamma=None, black_ridges=True, mode='reflect', cval=0):
    if False:
        for i in range(10):
            print('nop')
    "\n    Filter an image with the Frangi vesselness filter.\n\n    This filter can be used to detect continuous ridges, e.g. vessels,\n    wrinkles, rivers. It can be used to calculate the fraction of the\n    whole image containing such objects.\n\n    Defined only for 2-D and 3-D images. Calculates the eigenvalues of the\n    Hessian to compute the similarity of an image region to vessels, according\n    to the method described in [1]_.\n\n    Parameters\n    ----------\n    image : (M, N[, P]) ndarray\n        Array with input image data.\n    sigmas : iterable of floats, optional\n        Sigmas used as scales of filter, i.e.,\n        np.arange(scale_range[0], scale_range[1], scale_step)\n    scale_range : 2-tuple of floats, optional\n        The range of sigmas used.\n    scale_step : float, optional\n        Step size between sigmas.\n    alpha : float, optional\n        Frangi correction constant that adjusts the filter's\n        sensitivity to deviation from a plate-like structure.\n    beta : float, optional\n        Frangi correction constant that adjusts the filter's\n        sensitivity to deviation from a blob-like structure.\n    gamma : float, optional\n        Frangi correction constant that adjusts the filter's\n        sensitivity to areas of high variance/texture/structure.\n        The default, None, uses half of the maximum Hessian norm.\n    black_ridges : boolean, optional\n        When True (the default), the filter detects black ridges; when\n        False, it detects white ridges.\n    mode : {'constant', 'reflect', 'wrap', 'nearest', 'mirror'}, optional\n        How to handle values outside the image borders.\n    cval : float, optional\n        Used in conjunction with mode 'constant', the value outside\n        the image boundaries.\n\n    Returns\n    -------\n    out : (M, N[, P]) ndarray\n        Filtered image (maximum of pixels across all scales).\n\n    Notes\n    -----\n    Earlier versions of this filter were implemented by Marc Schrijver,\n    (November 2001), D. J. Kroon, University of Twente (May 2009) [2]_, and\n    D. G. Ellis (January 2017) [3]_.\n\n    See also\n    --------\n    meijering\n    sato\n    hessian\n\n    References\n    ----------\n    .. [1] Frangi, A. F., Niessen, W. J., Vincken, K. L., & Viergever, M. A.\n        (1998,). Multiscale vessel enhancement filtering. In International\n        Conference on Medical Image Computing and Computer-Assisted\n        Intervention (pp. 130-137). Springer Berlin Heidelberg.\n        :DOI:`10.1007/BFb0056195`\n    .. [2] Kroon, D. J.: Hessian based Frangi vesselness filter.\n    .. [3] Ellis, D. G.: https://github.com/ellisdg/frangi3d/tree/master/frangi\n    "
    if scale_range is not None and scale_step is not None:
        warn('Use keyword parameter `sigmas` instead of `scale_range` and `scale_range` which will be removed in version 0.17.', stacklevel=2)
        sigmas = np.arange(scale_range[0], scale_range[1], scale_step)
    check_nD(image, [2, 3])
    image = image.astype(_supported_float_type(image.dtype), copy=False)
    if not black_ridges:
        image = -image
    filtered_max = np.zeros_like(image)
    for sigma in sigmas:
        eigvals = hessian_matrix_eigvals(hessian_matrix(image, sigma, mode=mode, cval=cval, use_gaussian_derivatives=True))
        eigvals = np.take_along_axis(eigvals, abs(eigvals).argsort(0), 0)
        lambda1 = eigvals[0]
        if image.ndim == 2:
            (lambda2,) = np.maximum(eigvals[1:], 1e-10)
            r_a = np.inf
            r_b = abs(lambda1) / lambda2
        else:
            (lambda2, lambda3) = np.maximum(eigvals[1:], 1e-10)
            r_a = lambda2 / lambda3
            r_b = abs(lambda1) / np.sqrt(lambda2 * lambda3)
        s = np.sqrt((eigvals ** 2).sum(0))
        if gamma is None:
            gamma = s.max() / 2
            if gamma == 0:
                gamma = 1
        vals = 1.0 - np.exp(-r_a ** 2 / (2 * alpha ** 2))
        vals *= np.exp(-r_b ** 2 / (2 * beta ** 2))
        vals *= 1.0 - np.exp(-s ** 2 / (2 * gamma ** 2))
        filtered_max = np.maximum(filtered_max, vals)
    return filtered_max

def hessian(image, sigmas=range(1, 10, 2), scale_range=None, scale_step=None, alpha=0.5, beta=0.5, gamma=15, black_ridges=True, mode='reflect', cval=0):
    if False:
        return 10
    "Filter an image with the Hybrid Hessian filter.\n\n    This filter can be used to detect continuous edges, e.g. vessels,\n    wrinkles, rivers. It can be used to calculate the fraction of the whole\n    image containing such objects.\n\n    Defined only for 2-D and 3-D images. Almost equal to Frangi filter, but\n    uses alternative method of smoothing. Refer to [1]_ to find the differences\n    between Frangi and Hessian filters.\n\n    Parameters\n    ----------\n    image : (M, N[, P]) ndarray\n        Array with input image data.\n    sigmas : iterable of floats, optional\n        Sigmas used as scales of filter, i.e.,\n        np.arange(scale_range[0], scale_range[1], scale_step)\n    scale_range : 2-tuple of floats, optional\n        The range of sigmas used.\n    scale_step : float, optional\n        Step size between sigmas.\n    beta : float, optional\n        Frangi correction constant that adjusts the filter's\n        sensitivity to deviation from a blob-like structure.\n    gamma : float, optional\n        Frangi correction constant that adjusts the filter's\n        sensitivity to areas of high variance/texture/structure.\n    black_ridges : boolean, optional\n        When True (the default), the filter detects black ridges; when\n        False, it detects white ridges.\n    mode : {'constant', 'reflect', 'wrap', 'nearest', 'mirror'}, optional\n        How to handle values outside the image borders.\n    cval : float, optional\n        Used in conjunction with mode 'constant', the value outside\n        the image boundaries.\n\n    Returns\n    -------\n    out : (M, N[, P]) ndarray\n        Filtered image (maximum of pixels across all scales).\n\n    Notes\n    -----\n    Written by Marc Schrijver (November 2001)\n    Re-Written by D. J. Kroon University of Twente (May 2009) [2]_\n\n    See also\n    --------\n    meijering\n    sato\n    frangi\n\n    References\n    ----------\n    .. [1] Ng, C. C., Yap, M. H., Costen, N., & Li, B. (2014,). Automatic\n        wrinkle detection using hybrid Hessian filter. In Asian Conference on\n        Computer Vision (pp. 609-622). Springer International Publishing.\n        :DOI:`10.1007/978-3-319-16811-1_40`\n    .. [2] Kroon, D. J.: Hessian based Frangi vesselness filter.\n    "
    filtered = frangi(image, sigmas=sigmas, scale_range=scale_range, scale_step=scale_step, alpha=alpha, beta=beta, gamma=gamma, black_ridges=black_ridges, mode=mode, cval=cval)
    filtered[filtered <= 0] = 1
    return filtered