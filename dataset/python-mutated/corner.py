import functools
import math
from itertools import combinations_with_replacement
import numpy as np
from scipy import ndimage as ndi
from scipy import spatial, stats
from .._shared.filters import gaussian
from .._shared.utils import _supported_float_type, safe_as_int, warn
from ..transform import integral_image
from ..util import img_as_float
from ._hessian_det_appx import _hessian_matrix_det
from .corner_cy import _corner_fast, _corner_moravec, _corner_orientations
from .peak import peak_local_max
from .util import _prepare_grayscale_input_2D, _prepare_grayscale_input_nD

def _compute_derivatives(image, mode='constant', cval=0):
    if False:
        print('Hello World!')
    "Compute derivatives in axis directions using the Sobel operator.\n\n    Parameters\n    ----------\n    image : ndarray\n        Input image.\n    mode : {'constant', 'reflect', 'wrap', 'nearest', 'mirror'}, optional\n        How to handle values outside the image borders.\n    cval : float, optional\n        Used in conjunction with mode 'constant', the value outside\n        the image boundaries.\n\n    Returns\n    -------\n    derivatives : list of ndarray\n        Derivatives in each axis direction.\n\n    "
    derivatives = [ndi.sobel(image, axis=i, mode=mode, cval=cval) for i in range(image.ndim)]
    return derivatives

def structure_tensor(image, sigma=1, mode='constant', cval=0, order='rc'):
    if False:
        for i in range(10):
            print('nop')
    "Compute structure tensor using sum of squared differences.\n\n    The (2-dimensional) structure tensor A is defined as::\n\n        A = [Arr Arc]\n            [Arc Acc]\n\n    which is approximated by the weighted sum of squared differences in a local\n    window around each pixel in the image. This formula can be extended to a\n    larger number of dimensions (see [1]_).\n\n    Parameters\n    ----------\n    image : ndarray\n        Input image.\n    sigma : float or array-like of float, optional\n        Standard deviation used for the Gaussian kernel, which is used as a\n        weighting function for the local summation of squared differences.\n        If sigma is an iterable, its length must be equal to `image.ndim` and\n        each element is used for the Gaussian kernel applied along its\n        respective axis.\n    mode : {'constant', 'reflect', 'wrap', 'nearest', 'mirror'}, optional\n        How to handle values outside the image borders.\n    cval : float, optional\n        Used in conjunction with mode 'constant', the value outside\n        the image boundaries.\n    order : {'rc', 'xy'}, optional\n        NOTE: 'xy' is only an option for 2D images, higher dimensions must\n        always use 'rc' order. This parameter allows for the use of reverse or\n        forward order of the image axes in gradient computation. 'rc' indicates\n        the use of the first axis initially (Arr, Arc, Acc), whilst 'xy'\n        indicates the usage of the last axis initially (Axx, Axy, Ayy).\n\n    Returns\n    -------\n    A_elems : list of ndarray\n        Upper-diagonal elements of the structure tensor for each pixel in the\n        input image.\n\n    Examples\n    --------\n    >>> from skimage.feature import structure_tensor\n    >>> square = np.zeros((5, 5))\n    >>> square[2, 2] = 1\n    >>> Arr, Arc, Acc = structure_tensor(square, sigma=0.1, order='rc')\n    >>> Acc\n    array([[0., 0., 0., 0., 0.],\n           [0., 1., 0., 1., 0.],\n           [0., 4., 0., 4., 0.],\n           [0., 1., 0., 1., 0.],\n           [0., 0., 0., 0., 0.]])\n\n    See also\n    --------\n    structure_tensor_eigenvalues\n\n    References\n    ----------\n    .. [1] https://en.wikipedia.org/wiki/Structure_tensor\n    "
    if order == 'xy' and image.ndim > 2:
        raise ValueError('Only "rc" order is supported for dim > 2.')
    if order not in ['rc', 'xy']:
        raise ValueError(f'order {order} is invalid. Must be either "rc" or "xy"')
    if not np.isscalar(sigma):
        sigma = tuple(sigma)
        if len(sigma) != image.ndim:
            raise ValueError('sigma must have as many elements as image has axes')
    image = _prepare_grayscale_input_nD(image)
    derivatives = _compute_derivatives(image, mode=mode, cval=cval)
    if order == 'xy':
        derivatives = reversed(derivatives)
    A_elems = [gaussian(der0 * der1, sigma, mode=mode, cval=cval) for (der0, der1) in combinations_with_replacement(derivatives, 2)]
    return A_elems

def _hessian_matrix_with_gaussian(image, sigma=1, mode='reflect', cval=0, order='rc'):
    if False:
        while True:
            i = 10
    "Compute the Hessian via convolutions with Gaussian derivatives.\n\n    In 2D, the Hessian matrix is defined as:\n        H = [Hrr Hrc]\n            [Hrc Hcc]\n\n    which is computed by convolving the image with the second derivatives\n    of the Gaussian kernel in the respective r- and c-directions.\n\n    The implementation here also supports n-dimensional data.\n\n    Parameters\n    ----------\n    image : ndarray\n        Input image.\n    sigma : float or sequence of float, optional\n        Standard deviation used for the Gaussian kernel, which sets the\n        amount of smoothing in terms of pixel-distances. It is\n        advised to not choose a sigma much less than 1.0, otherwise\n        aliasing artifacts may occur.\n    mode : {'constant', 'reflect', 'wrap', 'nearest', 'mirror'}, optional\n        How to handle values outside the image borders.\n    cval : float, optional\n        Used in conjunction with mode 'constant', the value outside\n        the image boundaries.\n    order : {'rc', 'xy'}, optional\n        This parameter allows for the use of reverse or forward order of\n        the image axes in gradient computation. 'rc' indicates the use of\n        the first axis initially (Hrr, Hrc, Hcc), whilst 'xy' indicates the\n        usage of the last axis initially (Hxx, Hxy, Hyy)\n\n    Returns\n    -------\n    H_elems : list of ndarray\n        Upper-diagonal elements of the hessian matrix for each pixel in the\n        input image. In 2D, this will be a three element list containing [Hrr,\n        Hrc, Hcc]. In nD, the list will contain ``(n**2 + n) / 2`` arrays.\n\n    "
    image = img_as_float(image)
    float_dtype = _supported_float_type(image.dtype)
    image = image.astype(float_dtype, copy=False)
    if image.ndim > 2 and order == 'xy':
        raise ValueError("order='xy' is only supported for 2D images.")
    if order not in ['rc', 'xy']:
        raise ValueError(f'unrecognized order: {order}')
    if np.isscalar(sigma):
        sigma = (sigma,) * image.ndim
    truncate = 8 if all((s > 1 for s in sigma)) else 100
    sq1_2 = 1 / math.sqrt(2)
    sigma_scaled = tuple((sq1_2 * s for s in sigma))
    common_kwargs = dict(sigma=sigma_scaled, mode=mode, cval=cval, truncate=truncate)
    gaussian_ = functools.partial(ndi.gaussian_filter, **common_kwargs)
    ndim = image.ndim
    orders = tuple(([0] * d + [1] + [0] * (ndim - d - 1) for d in range(ndim)))
    gradients = [gaussian_(image, order=orders[d]) for d in range(ndim)]
    axes = range(ndim)
    if order == 'xy':
        axes = reversed(axes)
    H_elems = [gaussian_(gradients[ax0], order=orders[ax1]) for (ax0, ax1) in combinations_with_replacement(axes, 2)]
    return H_elems

def hessian_matrix(image, sigma=1, mode='constant', cval=0, order='rc', use_gaussian_derivatives=None):
    if False:
        i = 10
        return i + 15
    "Compute the Hessian matrix.\n\n    In 2D, the Hessian matrix is defined as::\n\n        H = [Hrr Hrc]\n            [Hrc Hcc]\n\n    which is computed by convolving the image with the second derivatives\n    of the Gaussian kernel in the respective r- and c-directions.\n\n    The implementation here also supports n-dimensional data.\n\n    Parameters\n    ----------\n    image : ndarray\n        Input image.\n    sigma : float\n        Standard deviation used for the Gaussian kernel, which is used as\n        weighting function for the auto-correlation matrix.\n    mode : {'constant', 'reflect', 'wrap', 'nearest', 'mirror'}, optional\n        How to handle values outside the image borders.\n    cval : float, optional\n        Used in conjunction with mode 'constant', the value outside\n        the image boundaries.\n    order : {'rc', 'xy'}, optional\n        For 2D images, this parameter allows for the use of reverse or forward\n        order of the image axes in gradient computation. 'rc' indicates the use\n        of the first axis initially (Hrr, Hrc, Hcc), whilst 'xy' indicates the\n        usage of the last axis initially (Hxx, Hxy, Hyy). Images with higher\n        dimension must always use 'rc' order.\n    use_gaussian_derivatives : boolean, optional\n        Indicates whether the Hessian is computed by convolving with Gaussian\n        derivatives, or by a simple finite-difference operation.\n\n    Returns\n    -------\n    H_elems : list of ndarray\n        Upper-diagonal elements of the hessian matrix for each pixel in the\n        input image. In 2D, this will be a three element list containing [Hrr,\n        Hrc, Hcc]. In nD, the list will contain ``(n**2 + n) / 2`` arrays.\n\n\n    Notes\n    -----\n    The distributive property of derivatives and convolutions allows us to\n    restate the derivative of an image, I, smoothed with a Gaussian kernel, G,\n    as the convolution of the image with the derivative of G.\n\n    .. math::\n\n        \\frac{\\partial }{\\partial x_i}(I * G) =\n        I * \\left( \\frac{\\partial }{\\partial x_i} G \\right)\n\n    When ``use_gaussian_derivatives`` is ``True``, this property is used to\n    compute the second order derivatives that make up the Hessian matrix.\n\n    When ``use_gaussian_derivatives`` is ``False``, simple finite differences\n    on a Gaussian-smoothed image are used instead.\n\n    Examples\n    --------\n    >>> from skimage.feature import hessian_matrix\n    >>> square = np.zeros((5, 5))\n    >>> square[2, 2] = 4\n    >>> Hrr, Hrc, Hcc = hessian_matrix(square, sigma=0.1, order='rc',\n    ...                                use_gaussian_derivatives=False)\n    >>> Hrc\n    array([[ 0.,  0.,  0.,  0.,  0.],\n           [ 0.,  1.,  0., -1.,  0.],\n           [ 0.,  0.,  0.,  0.,  0.],\n           [ 0., -1.,  0.,  1.,  0.],\n           [ 0.,  0.,  0.,  0.,  0.]])\n\n    "
    image = img_as_float(image)
    float_dtype = _supported_float_type(image.dtype)
    image = image.astype(float_dtype, copy=False)
    if image.ndim > 2 and order == 'xy':
        raise ValueError("order='xy' is only supported for 2D images.")
    if order not in ['rc', 'xy']:
        raise ValueError(f'unrecognized order: {order}')
    if use_gaussian_derivatives is None:
        use_gaussian_derivatives = False
        warn('use_gaussian_derivatives currently defaults to False, but will change to True in a future version. Please specify this argument explicitly to maintain the current behavior', category=FutureWarning, stacklevel=2)
    if use_gaussian_derivatives:
        return _hessian_matrix_with_gaussian(image, sigma=sigma, mode=mode, cval=cval, order=order)
    gaussian_filtered = gaussian(image, sigma=sigma, mode=mode, cval=cval)
    gradients = np.gradient(gaussian_filtered)
    axes = range(image.ndim)
    if order == 'xy':
        axes = reversed(axes)
    H_elems = [np.gradient(gradients[ax0], axis=ax1) for (ax0, ax1) in combinations_with_replacement(axes, 2)]
    return H_elems

def hessian_matrix_det(image, sigma=1, approximate=True):
    if False:
        while True:
            i = 10
    'Compute the approximate Hessian Determinant over an image.\n\n    The 2D approximate method uses box filters over integral images to\n    compute the approximate Hessian Determinant.\n\n    Parameters\n    ----------\n    image : ndarray\n        The image over which to compute the Hessian Determinant.\n    sigma : float, optional\n        Standard deviation of the Gaussian kernel used for the Hessian\n        matrix.\n    approximate : bool, optional\n        If ``True`` and the image is 2D, use a much faster approximate\n        computation. This argument has no effect on 3D and higher images.\n\n    Returns\n    -------\n    out : array\n        The array of the Determinant of Hessians.\n\n    References\n    ----------\n    .. [1] Herbert Bay, Andreas Ess, Tinne Tuytelaars, Luc Van Gool,\n           "SURF: Speeded Up Robust Features"\n           ftp://ftp.vision.ee.ethz.ch/publications/articles/eth_biwi_00517.pdf\n\n    Notes\n    -----\n    For 2D images when ``approximate=True``, the running time of this method\n    only depends on size of the image. It is independent of `sigma` as one\n    would expect. The downside is that the result for `sigma` less than `3`\n    is not accurate, i.e., not similar to the result obtained if someone\n    computed the Hessian and took its determinant.\n    '
    image = img_as_float(image)
    float_dtype = _supported_float_type(image.dtype)
    image = image.astype(float_dtype, copy=False)
    if image.ndim == 2 and approximate:
        integral = integral_image(image)
        return np.array(_hessian_matrix_det(integral, sigma))
    else:
        hessian_mat_array = _symmetric_image(hessian_matrix(image, sigma, use_gaussian_derivatives=False))
        return np.linalg.det(hessian_mat_array)

def _symmetric_compute_eigenvalues(S_elems):
    if False:
        while True:
            i = 10
    'Compute eigenvalues from the upper-diagonal entries of a symmetric\n    matrix.\n\n    Parameters\n    ----------\n    S_elems : list of ndarray\n        The upper-diagonal elements of the matrix, as returned by\n        `hessian_matrix` or `structure_tensor`.\n\n    Returns\n    -------\n    eigs : ndarray\n        The eigenvalues of the matrix, in decreasing order. The eigenvalues are\n        the leading dimension. That is, ``eigs[i, j, k]`` contains the\n        ith-largest eigenvalue at position (j, k).\n    '
    if len(S_elems) == 3:
        (M00, M01, M11) = S_elems
        eigs = np.empty((2, *M00.shape), M00.dtype)
        eigs[:] = (M00 + M11) / 2
        hsqrtdet = np.sqrt(M01 ** 2 + ((M00 - M11) / 2) ** 2)
        eigs[0] += hsqrtdet
        eigs[1] -= hsqrtdet
        return eigs
    else:
        matrices = _symmetric_image(S_elems)
        eigs = np.linalg.eigvalsh(matrices)[..., ::-1]
        leading_axes = tuple(range(eigs.ndim - 1))
        return np.transpose(eigs, (eigs.ndim - 1,) + leading_axes)

def _symmetric_image(S_elems):
    if False:
        i = 10
        return i + 15
    'Convert the upper-diagonal elements of a matrix to the full\n    symmetric matrix.\n\n    Parameters\n    ----------\n    S_elems : list of array\n        The upper-diagonal elements of the matrix, as returned by\n        `hessian_matrix` or `structure_tensor`.\n\n    Returns\n    -------\n    image : array\n        An array of shape ``(M, N[, ...], image.ndim, image.ndim)``,\n        containing the matrix corresponding to each coordinate.\n    '
    image = S_elems[0]
    symmetric_image = np.zeros(image.shape + (image.ndim, image.ndim), dtype=S_elems[0].dtype)
    for (idx, (row, col)) in enumerate(combinations_with_replacement(range(image.ndim), 2)):
        symmetric_image[..., row, col] = S_elems[idx]
        symmetric_image[..., col, row] = S_elems[idx]
    return symmetric_image

def structure_tensor_eigenvalues(A_elems):
    if False:
        print('Hello World!')
    "Compute eigenvalues of structure tensor.\n\n    Parameters\n    ----------\n    A_elems : list of ndarray\n        The upper-diagonal elements of the structure tensor, as returned\n        by `structure_tensor`.\n\n    Returns\n    -------\n    ndarray\n        The eigenvalues of the structure tensor, in decreasing order. The\n        eigenvalues are the leading dimension. That is, the coordinate\n        [i, j, k] corresponds to the ith-largest eigenvalue at position (j, k).\n\n    Examples\n    --------\n    >>> from skimage.feature import structure_tensor\n    >>> from skimage.feature import structure_tensor_eigenvalues\n    >>> square = np.zeros((5, 5))\n    >>> square[2, 2] = 1\n    >>> A_elems = structure_tensor(square, sigma=0.1, order='rc')\n    >>> structure_tensor_eigenvalues(A_elems)[0]\n    array([[0., 0., 0., 0., 0.],\n           [0., 2., 4., 2., 0.],\n           [0., 4., 0., 4., 0.],\n           [0., 2., 4., 2., 0.],\n           [0., 0., 0., 0., 0.]])\n\n    See also\n    --------\n    structure_tensor\n    "
    return _symmetric_compute_eigenvalues(A_elems)

def hessian_matrix_eigvals(H_elems):
    if False:
        print('Hello World!')
    "Compute eigenvalues of Hessian matrix.\n\n    Parameters\n    ----------\n    H_elems : list of ndarray\n        The upper-diagonal elements of the Hessian matrix, as returned\n        by `hessian_matrix`.\n\n    Returns\n    -------\n    eigs : ndarray\n        The eigenvalues of the Hessian matrix, in decreasing order. The\n        eigenvalues are the leading dimension. That is, ``eigs[i, j, k]``\n        contains the ith-largest eigenvalue at position (j, k).\n\n    Examples\n    --------\n    >>> from skimage.feature import hessian_matrix, hessian_matrix_eigvals\n    >>> square = np.zeros((5, 5))\n    >>> square[2, 2] = 4\n    >>> H_elems = hessian_matrix(square, sigma=0.1, order='rc',\n    ...                          use_gaussian_derivatives=False)\n    >>> hessian_matrix_eigvals(H_elems)[0]\n    array([[ 0.,  0.,  2.,  0.,  0.],\n           [ 0.,  1.,  0.,  1.,  0.],\n           [ 2.,  0., -2.,  0.,  2.],\n           [ 0.,  1.,  0.,  1.,  0.],\n           [ 0.,  0.,  2.,  0.,  0.]])\n    "
    return _symmetric_compute_eigenvalues(H_elems)

def shape_index(image, sigma=1, mode='constant', cval=0):
    if False:
        i = 10
        return i + 15
    'Compute the shape index.\n\n    The shape index, as defined by Koenderink & van Doorn [1]_, is a\n    single valued measure of local curvature, assuming the image as a 3D plane\n    with intensities representing heights.\n\n    It is derived from the eigenvalues of the Hessian, and its\n    value ranges from -1 to 1 (and is undefined (=NaN) in *flat* regions),\n    with following ranges representing following shapes:\n\n    .. table:: Ranges of the shape index and corresponding shapes.\n\n      ===================  =============\n      Interval (s in ...)  Shape\n      ===================  =============\n      [  -1, -7/8)         Spherical cup\n      [-7/8, -5/8)         Through\n      [-5/8, -3/8)         Rut\n      [-3/8, -1/8)         Saddle rut\n      [-1/8, +1/8)         Saddle\n      [+1/8, +3/8)         Saddle ridge\n      [+3/8, +5/8)         Ridge\n      [+5/8, +7/8)         Dome\n      [+7/8,   +1]         Spherical cap\n      ===================  =============\n\n    Parameters\n    ----------\n    image : (M, N) ndarray\n        Input image.\n    sigma : float, optional\n        Standard deviation used for the Gaussian kernel, which is used for\n        smoothing the input data before Hessian eigen value calculation.\n    mode : {\'constant\', \'reflect\', \'wrap\', \'nearest\', \'mirror\'}, optional\n        How to handle values outside the image borders\n    cval : float, optional\n        Used in conjunction with mode \'constant\', the value outside\n        the image boundaries.\n\n    Returns\n    -------\n    s : ndarray\n        Shape index\n\n    References\n    ----------\n    .. [1] Koenderink, J. J. & van Doorn, A. J.,\n           "Surface shape and curvature scales",\n           Image and Vision Computing, 1992, 10, 557-564.\n           :DOI:`10.1016/0262-8856(92)90076-F`\n\n    Examples\n    --------\n    >>> from skimage.feature import shape_index\n    >>> square = np.zeros((5, 5))\n    >>> square[2, 2] = 4\n    >>> s = shape_index(square, sigma=0.1)\n    >>> s\n    array([[ nan,  nan, -0.5,  nan,  nan],\n           [ nan, -0. ,  nan, -0. ,  nan],\n           [-0.5,  nan, -1. ,  nan, -0.5],\n           [ nan, -0. ,  nan, -0. ,  nan],\n           [ nan,  nan, -0.5,  nan,  nan]])\n    '
    H = hessian_matrix(image, sigma=sigma, mode=mode, cval=cval, order='rc', use_gaussian_derivatives=False)
    (l1, l2) = hessian_matrix_eigvals(H)
    with np.errstate(divide='ignore', invalid='ignore'):
        return 2.0 / np.pi * np.arctan((l2 + l1) / (l2 - l1))

def corner_kitchen_rosenfeld(image, mode='constant', cval=0):
    if False:
        return 10
    "Compute Kitchen and Rosenfeld corner measure response image.\n\n    The corner measure is calculated as follows::\n\n        (imxx * imy**2 + imyy * imx**2 - 2 * imxy * imx * imy)\n            / (imx**2 + imy**2)\n\n    Where imx and imy are the first and imxx, imxy, imyy the second\n    derivatives.\n\n    Parameters\n    ----------\n    image : (M, N) ndarray\n        Input image.\n    mode : {'constant', 'reflect', 'wrap', 'nearest', 'mirror'}, optional\n        How to handle values outside the image borders.\n    cval : float, optional\n        Used in conjunction with mode 'constant', the value outside\n        the image boundaries.\n\n    Returns\n    -------\n    response : ndarray\n        Kitchen and Rosenfeld response image.\n\n    References\n    ----------\n    .. [1] Kitchen, L., & Rosenfeld, A. (1982). Gray-level corner detection.\n           Pattern recognition letters, 1(2), 95-102.\n           :DOI:`10.1016/0167-8655(82)90020-4`\n    "
    float_dtype = _supported_float_type(image.dtype)
    image = image.astype(float_dtype, copy=False)
    (imy, imx) = _compute_derivatives(image, mode=mode, cval=cval)
    (imxy, imxx) = _compute_derivatives(imx, mode=mode, cval=cval)
    (imyy, imyx) = _compute_derivatives(imy, mode=mode, cval=cval)
    numerator = imxx * imy ** 2 + imyy * imx ** 2 - 2 * imxy * imx * imy
    denominator = imx ** 2 + imy ** 2
    response = np.zeros_like(image, dtype=float_dtype)
    mask = denominator != 0
    response[mask] = numerator[mask] / denominator[mask]
    return response

def corner_harris(image, method='k', k=0.05, eps=1e-06, sigma=1):
    if False:
        print('Hello World!')
    "Compute Harris corner measure response image.\n\n    This corner detector uses information from the auto-correlation matrix A::\n\n        A = [(imx**2)   (imx*imy)] = [Axx Axy]\n            [(imx*imy)   (imy**2)]   [Axy Ayy]\n\n    Where imx and imy are first derivatives, averaged with a gaussian filter.\n    The corner measure is then defined as::\n\n        det(A) - k * trace(A)**2\n\n    or::\n\n        2 * det(A) / (trace(A) + eps)\n\n    Parameters\n    ----------\n    image : (M, N) ndarray\n        Input image.\n    method : {'k', 'eps'}, optional\n        Method to compute the response image from the auto-correlation matrix.\n    k : float, optional\n        Sensitivity factor to separate corners from edges, typically in range\n        `[0, 0.2]`. Small values of k result in detection of sharp corners.\n    eps : float, optional\n        Normalisation factor (Noble's corner measure).\n    sigma : float, optional\n        Standard deviation used for the Gaussian kernel, which is used as\n        weighting function for the auto-correlation matrix.\n\n    Returns\n    -------\n    response : ndarray\n        Harris response image.\n\n    References\n    ----------\n    .. [1] https://en.wikipedia.org/wiki/Corner_detection\n\n    Examples\n    --------\n    >>> from skimage.feature import corner_harris, corner_peaks\n    >>> square = np.zeros([10, 10])\n    >>> square[2:8, 2:8] = 1\n    >>> square.astype(int)\n    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],\n           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],\n           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],\n           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],\n           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],\n           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])\n    >>> corner_peaks(corner_harris(square), min_distance=1)\n    array([[2, 2],\n           [2, 7],\n           [7, 2],\n           [7, 7]])\n\n    "
    (Arr, Arc, Acc) = structure_tensor(image, sigma, order='rc')
    detA = Arr * Acc - Arc ** 2
    traceA = Arr + Acc
    if method == 'k':
        response = detA - k * traceA ** 2
    else:
        response = 2 * detA / (traceA + eps)
    return response

def corner_shi_tomasi(image, sigma=1):
    if False:
        i = 10
        return i + 15
    'Compute Shi-Tomasi (Kanade-Tomasi) corner measure response image.\n\n    This corner detector uses information from the auto-correlation matrix A::\n\n        A = [(imx**2)   (imx*imy)] = [Axx Axy]\n            [(imx*imy)   (imy**2)]   [Axy Ayy]\n\n    Where imx and imy are first derivatives, averaged with a gaussian filter.\n    The corner measure is then defined as the smaller eigenvalue of A::\n\n        ((Axx + Ayy) - sqrt((Axx - Ayy)**2 + 4 * Axy**2)) / 2\n\n    Parameters\n    ----------\n    image : (M, N) ndarray\n        Input image.\n    sigma : float, optional\n        Standard deviation used for the Gaussian kernel, which is used as\n        weighting function for the auto-correlation matrix.\n\n    Returns\n    -------\n    response : ndarray\n        Shi-Tomasi response image.\n\n    References\n    ----------\n    .. [1] https://en.wikipedia.org/wiki/Corner_detection\n\n    Examples\n    --------\n    >>> from skimage.feature import corner_shi_tomasi, corner_peaks\n    >>> square = np.zeros([10, 10])\n    >>> square[2:8, 2:8] = 1\n    >>> square.astype(int)\n    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],\n           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],\n           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],\n           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],\n           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],\n           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])\n    >>> corner_peaks(corner_shi_tomasi(square), min_distance=1)\n    array([[2, 2],\n           [2, 7],\n           [7, 2],\n           [7, 7]])\n\n    '
    (Arr, Arc, Acc) = structure_tensor(image, sigma, order='rc')
    response = (Arr + Acc - np.sqrt((Arr - Acc) ** 2 + 4 * Arc ** 2)) / 2
    return response

def corner_foerstner(image, sigma=1):
    if False:
        print('Hello World!')
    'Compute Foerstner corner measure response image.\n\n    This corner detector uses information from the auto-correlation matrix A::\n\n        A = [(imx**2)   (imx*imy)] = [Axx Axy]\n            [(imx*imy)   (imy**2)]   [Axy Ayy]\n\n    Where imx and imy are first derivatives, averaged with a gaussian filter.\n    The corner measure is then defined as::\n\n        w = det(A) / trace(A)           (size of error ellipse)\n        q = 4 * det(A) / trace(A)**2    (roundness of error ellipse)\n\n    Parameters\n    ----------\n    image : (M, N) ndarray\n        Input image.\n    sigma : float, optional\n        Standard deviation used for the Gaussian kernel, which is used as\n        weighting function for the auto-correlation matrix.\n\n    Returns\n    -------\n    w : ndarray\n        Error ellipse sizes.\n    q : ndarray\n        Roundness of error ellipse.\n\n    References\n    ----------\n    .. [1] Förstner, W., & Gülch, E. (1987, June). A fast operator for\n           detection and precise location of distinct points, corners and\n           centres of circular features. In Proc. ISPRS intercommission\n           conference on fast processing of photogrammetric data (pp. 281-305).\n           https://cseweb.ucsd.edu/classes/sp02/cse252/foerstner/foerstner.pdf\n    .. [2] https://en.wikipedia.org/wiki/Corner_detection\n\n    Examples\n    --------\n    >>> from skimage.feature import corner_foerstner, corner_peaks\n    >>> square = np.zeros([10, 10])\n    >>> square[2:8, 2:8] = 1\n    >>> square.astype(int)\n    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],\n           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],\n           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],\n           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],\n           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],\n           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])\n    >>> w, q = corner_foerstner(square)\n    >>> accuracy_thresh = 0.5\n    >>> roundness_thresh = 0.3\n    >>> foerstner = (q > roundness_thresh) * (w > accuracy_thresh) * w\n    >>> corner_peaks(foerstner, min_distance=1)\n    array([[2, 2],\n           [2, 7],\n           [7, 2],\n           [7, 7]])\n\n    '
    (Arr, Arc, Acc) = structure_tensor(image, sigma, order='rc')
    detA = Arr * Acc - Arc ** 2
    traceA = Arr + Acc
    w = np.zeros_like(image, dtype=detA.dtype)
    q = np.zeros_like(w)
    mask = traceA != 0
    w[mask] = detA[mask] / traceA[mask]
    q[mask] = 4 * detA[mask] / traceA[mask] ** 2
    return (w, q)

def corner_fast(image, n=12, threshold=0.15):
    if False:
        while True:
            i = 10
    'Extract FAST corners for a given image.\n\n    Parameters\n    ----------\n    image : (M, N) ndarray\n        Input image.\n    n : int, optional\n        Minimum number of consecutive pixels out of 16 pixels on the circle\n        that should all be either brighter or darker w.r.t testpixel.\n        A point c on the circle is darker w.r.t test pixel p if\n        `Ic < Ip - threshold` and brighter if `Ic > Ip + threshold`. Also\n        stands for the n in `FAST-n` corner detector.\n    threshold : float, optional\n        Threshold used in deciding whether the pixels on the circle are\n        brighter, darker or similar w.r.t. the test pixel. Decrease the\n        threshold when more corners are desired and vice-versa.\n\n    Returns\n    -------\n    response : ndarray\n        FAST corner response image.\n\n    References\n    ----------\n    .. [1] Rosten, E., & Drummond, T. (2006, May). Machine learning for\n           high-speed corner detection. In European conference on computer\n           vision (pp. 430-443). Springer, Berlin, Heidelberg.\n           :DOI:`10.1007/11744023_34`\n           http://www.edwardrosten.com/work/rosten_2006_machine.pdf\n    .. [2] Wikipedia, "Features from accelerated segment test",\n           https://en.wikipedia.org/wiki/Features_from_accelerated_segment_test\n\n    Examples\n    --------\n    >>> from skimage.feature import corner_fast, corner_peaks\n    >>> square = np.zeros((12, 12))\n    >>> square[3:9, 3:9] = 1\n    >>> square.astype(int)\n    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],\n           [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],\n           [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],\n           [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],\n           [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],\n           [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])\n    >>> corner_peaks(corner_fast(square, 9), min_distance=1)\n    array([[3, 3],\n           [3, 8],\n           [8, 3],\n           [8, 8]])\n\n    '
    image = _prepare_grayscale_input_2D(image)
    image = np.ascontiguousarray(image)
    response = _corner_fast(image, n, threshold)
    return response

def corner_subpix(image, corners, window_size=11, alpha=0.99):
    if False:
        print('Hello World!')
    'Determine subpixel position of corners.\n\n    A statistical test decides whether the corner is defined as the\n    intersection of two edges or a single peak. Depending on the classification\n    result, the subpixel corner location is determined based on the local\n    covariance of the grey-values. If the significance level for either\n    statistical test is not sufficient, the corner cannot be classified, and\n    the output subpixel position is set to NaN.\n\n    Parameters\n    ----------\n    image : (M, N) ndarray\n        Input image.\n    corners : (K, 2) ndarray\n        Corner coordinates `(row, col)`.\n    window_size : int, optional\n        Search window size for subpixel estimation.\n    alpha : float, optional\n        Significance level for corner classification.\n\n    Returns\n    -------\n    positions : (K, 2) ndarray\n        Subpixel corner positions. NaN for "not classified" corners.\n\n    References\n    ----------\n    .. [1] Förstner, W., & Gülch, E. (1987, June). A fast operator for\n           detection and precise location of distinct points, corners and\n           centres of circular features. In Proc. ISPRS intercommission\n           conference on fast processing of photogrammetric data (pp. 281-305).\n           https://cseweb.ucsd.edu/classes/sp02/cse252/foerstner/foerstner.pdf\n    .. [2] https://en.wikipedia.org/wiki/Corner_detection\n\n    Examples\n    --------\n    >>> from skimage.feature import corner_harris, corner_peaks, corner_subpix\n    >>> img = np.zeros((10, 10))\n    >>> img[:5, :5] = 1\n    >>> img[5:, 5:] = 1\n    >>> img.astype(int)\n    array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],\n           [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],\n           [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],\n           [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],\n           [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],\n           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],\n           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],\n           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],\n           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]])\n    >>> coords = corner_peaks(corner_harris(img), min_distance=2)\n    >>> coords_subpix = corner_subpix(img, coords, window_size=7)\n    >>> coords_subpix\n    array([[4.5, 4.5]])\n\n    '
    wext = (window_size - 1) // 2
    float_dtype = _supported_float_type(image.dtype)
    image = image.astype(float_dtype, copy=False)
    image = np.pad(image, pad_width=wext, mode='constant', constant_values=0)
    corners = safe_as_int(corners + wext)
    N_dot = np.zeros((2, 2), dtype=float_dtype)
    N_edge = np.zeros((2, 2), dtype=float_dtype)
    b_dot = np.zeros((2,), dtype=float_dtype)
    b_edge = np.zeros((2,), dtype=float_dtype)
    redundancy = window_size ** 2 - 2
    t_crit_dot = stats.f.isf(1 - alpha, redundancy, redundancy)
    t_crit_edge = stats.f.isf(alpha, redundancy, redundancy)
    (y, x) = np.mgrid[-wext:wext + 1, -wext:wext + 1]
    corners_subpix = np.zeros_like(corners, dtype=float_dtype)
    for (i, (y0, x0)) in enumerate(corners):
        miny = y0 - wext - 1
        maxy = y0 + wext + 2
        minx = x0 - wext - 1
        maxx = x0 + wext + 2
        window = image[miny:maxy, minx:maxx]
        (winy, winx) = _compute_derivatives(window, mode='constant', cval=0)
        winx_winx = (winx * winx)[1:-1, 1:-1]
        winx_winy = (winx * winy)[1:-1, 1:-1]
        winy_winy = (winy * winy)[1:-1, 1:-1]
        Axx = np.sum(winx_winx)
        Axy = np.sum(winx_winy)
        Ayy = np.sum(winy_winy)
        bxx_x = np.sum(winx_winx * x)
        bxx_y = np.sum(winx_winx * y)
        bxy_x = np.sum(winx_winy * x)
        bxy_y = np.sum(winx_winy * y)
        byy_x = np.sum(winy_winy * x)
        byy_y = np.sum(winy_winy * y)
        N_dot[0, 0] = Axx
        N_dot[0, 1] = N_dot[1, 0] = -Axy
        N_dot[1, 1] = Ayy
        N_edge[0, 0] = Ayy
        N_edge[0, 1] = N_edge[1, 0] = Axy
        N_edge[1, 1] = Axx
        b_dot[:] = (bxx_y - bxy_x, byy_x - bxy_y)
        b_edge[:] = (byy_y + bxy_x, bxx_x + bxy_y)
        try:
            est_dot = np.linalg.solve(N_dot, b_dot)
            est_edge = np.linalg.solve(N_edge, b_edge)
        except np.linalg.LinAlgError:
            corners_subpix[i, :] = (np.nan, np.nan)
            continue
        ry_dot = y - est_dot[0]
        rx_dot = x - est_dot[1]
        ry_edge = y - est_edge[0]
        rx_edge = x - est_edge[1]
        rxx_dot = rx_dot * rx_dot
        rxy_dot = rx_dot * ry_dot
        ryy_dot = ry_dot * ry_dot
        rxx_edge = rx_edge * rx_edge
        rxy_edge = rx_edge * ry_edge
        ryy_edge = ry_edge * ry_edge
        var_dot = np.sum(winx_winx * ryy_dot - 2 * winx_winy * rxy_dot + winy_winy * rxx_dot)
        var_edge = np.sum(winy_winy * ryy_edge + 2 * winx_winy * rxy_edge + winx_winx * rxx_edge)
        if var_dot < np.spacing(1) and var_edge < np.spacing(1):
            t = np.nan
        elif var_dot == 0:
            t = np.inf
        else:
            t = var_edge / var_dot
        corner_class = int(t < t_crit_edge) - int(t > t_crit_dot)
        if corner_class == -1:
            corners_subpix[i, :] = (y0 + est_dot[0], x0 + est_dot[1])
        elif corner_class == 0:
            corners_subpix[i, :] = (np.nan, np.nan)
        elif corner_class == 1:
            corners_subpix[i, :] = (y0 + est_edge[0], x0 + est_edge[1])
    corners_subpix -= wext
    return corners_subpix

def corner_peaks(image, min_distance=1, threshold_abs=None, threshold_rel=None, exclude_border=True, indices=True, num_peaks=np.inf, footprint=None, labels=None, *, num_peaks_per_label=np.inf, p_norm=np.inf):
    if False:
        while True:
            i = 10
    'Find peaks in corner measure response image.\n\n    This differs from `skimage.feature.peak_local_max` in that it suppresses\n    multiple connected peaks with the same accumulator value.\n\n    Parameters\n    ----------\n    image : (M, N) ndarray\n        Input image.\n    min_distance : int, optional\n        The minimal allowed distance separating peaks.\n    * : *\n        See :py:meth:`skimage.feature.peak_local_max`.\n    p_norm : float\n        Which Minkowski p-norm to use. Should be in the range [1, inf].\n        A finite large p may cause a ValueError if overflow can occur.\n        ``inf`` corresponds to the Chebyshev distance and 2 to the\n        Euclidean distance.\n\n    Returns\n    -------\n    output : ndarray or ndarray of bools\n\n        * If `indices = True`  : (row, column, ...) coordinates of peaks.\n        * If `indices = False` : Boolean array shaped like `image`, with peaks\n          represented by True values.\n\n    See also\n    --------\n    skimage.feature.peak_local_max\n\n    Notes\n    -----\n    .. versionchanged:: 0.18\n        The default value of `threshold_rel` has changed to None, which\n        corresponds to letting `skimage.feature.peak_local_max` decide on the\n        default. This is equivalent to `threshold_rel=0`.\n\n    The `num_peaks` limit is applied before suppression of connected peaks.\n    To limit the number of peaks after suppression, set `num_peaks=np.inf` and\n    post-process the output of this function.\n\n    Examples\n    --------\n    >>> from skimage.feature import peak_local_max\n    >>> response = np.zeros((5, 5))\n    >>> response[2:4, 2:4] = 1\n    >>> response\n    array([[0., 0., 0., 0., 0.],\n           [0., 0., 0., 0., 0.],\n           [0., 0., 1., 1., 0.],\n           [0., 0., 1., 1., 0.],\n           [0., 0., 0., 0., 0.]])\n    >>> peak_local_max(response)\n    array([[2, 2],\n           [2, 3],\n           [3, 2],\n           [3, 3]])\n    >>> corner_peaks(response)\n    array([[2, 2]])\n\n    '
    if np.isinf(num_peaks):
        num_peaks = None
    coords = peak_local_max(image, min_distance=min_distance, threshold_abs=threshold_abs, threshold_rel=threshold_rel, exclude_border=exclude_border, num_peaks=np.inf, footprint=footprint, labels=labels, num_peaks_per_label=num_peaks_per_label)
    if len(coords):
        tree = spatial.cKDTree(coords)
        rejected_peaks_indices = set()
        for (idx, point) in enumerate(coords):
            if idx not in rejected_peaks_indices:
                candidates = tree.query_ball_point(point, r=min_distance, p=p_norm)
                candidates.remove(idx)
                rejected_peaks_indices.update(candidates)
        coords = np.delete(coords, tuple(rejected_peaks_indices), axis=0)[:num_peaks]
    if indices:
        return coords
    peaks = np.zeros_like(image, dtype=bool)
    peaks[tuple(coords.T)] = True
    return peaks

def corner_moravec(image, window_size=1):
    if False:
        while True:
            i = 10
    'Compute Moravec corner measure response image.\n\n    This is one of the simplest corner detectors and is comparatively fast but\n    has several limitations (e.g. not rotation invariant).\n\n    Parameters\n    ----------\n    image : (M, N) ndarray\n        Input image.\n    window_size : int, optional\n        Window size.\n\n    Returns\n    -------\n    response : ndarray\n        Moravec response image.\n\n    References\n    ----------\n    .. [1] https://en.wikipedia.org/wiki/Corner_detection\n\n    Examples\n    --------\n    >>> from skimage.feature import corner_moravec\n    >>> square = np.zeros([7, 7])\n    >>> square[3, 3] = 1\n    >>> square.astype(int)\n    array([[0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 1, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0]])\n    >>> corner_moravec(square).astype(int)\n    array([[0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 1, 1, 1, 0, 0],\n           [0, 0, 1, 2, 1, 0, 0],\n           [0, 0, 1, 1, 1, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0]])\n    '
    image = img_as_float(image)
    float_dtype = _supported_float_type(image.dtype)
    image = image.astype(float_dtype, copy=False)
    return _corner_moravec(np.ascontiguousarray(image), window_size)

def corner_orientations(image, corners, mask):
    if False:
        print('Hello World!')
    'Compute the orientation of corners.\n\n    The orientation of corners is computed using the first order central moment\n    i.e. the center of mass approach. The corner orientation is the angle of\n    the vector from the corner coordinate to the intensity centroid in the\n    local neighborhood around the corner calculated using first order central\n    moment.\n\n    Parameters\n    ----------\n    image : (M, N) array\n        Input grayscale image.\n    corners : (K, 2) array\n        Corner coordinates as ``(row, col)``.\n    mask : 2D array\n        Mask defining the local neighborhood of the corner used for the\n        calculation of the central moment.\n\n    Returns\n    -------\n    orientations : (K, 1) array\n        Orientations of corners in the range [-pi, pi].\n\n    References\n    ----------\n    .. [1] Ethan Rublee, Vincent Rabaud, Kurt Konolige and Gary Bradski\n          "ORB : An efficient alternative to SIFT and SURF"\n          http://www.vision.cs.chubu.ac.jp/CV-R/pdf/Rublee_iccv2011.pdf\n    .. [2] Paul L. Rosin, "Measuring Corner Properties"\n          http://users.cs.cf.ac.uk/Paul.Rosin/corner2.pdf\n\n    Examples\n    --------\n    >>> from skimage.morphology import octagon\n    >>> from skimage.feature import (corner_fast, corner_peaks,\n    ...                              corner_orientations)\n    >>> square = np.zeros((12, 12))\n    >>> square[3:9, 3:9] = 1\n    >>> square.astype(int)\n    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],\n           [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],\n           [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],\n           [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],\n           [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],\n           [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])\n    >>> corners = corner_peaks(corner_fast(square, 9), min_distance=1)\n    >>> corners\n    array([[3, 3],\n           [3, 8],\n           [8, 3],\n           [8, 8]])\n    >>> orientations = corner_orientations(square, corners, octagon(3, 2))\n    >>> np.rad2deg(orientations)\n    array([  45.,  135.,  -45., -135.])\n\n    '
    image = _prepare_grayscale_input_2D(image)
    return _corner_orientations(image, corners, mask)