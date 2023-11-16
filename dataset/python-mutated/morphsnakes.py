from itertools import cycle
import numpy as np
from scipy import ndimage as ndi
from .._shared.utils import check_nD
__all__ = ['morphological_chan_vese', 'morphological_geodesic_active_contour', 'inverse_gaussian_gradient', 'disk_level_set', 'checkerboard_level_set']

class _fcycle:

    def __init__(self, iterable):
        if False:
            i = 10
            return i + 15
        'Call functions from the iterable each time it is called.'
        self.funcs = cycle(iterable)

    def __call__(self, *args, **kwargs):
        if False:
            return 10
        f = next(self.funcs)
        return f(*args, **kwargs)
_P2 = [np.eye(3), np.array([[0, 1, 0]] * 3), np.flipud(np.eye(3)), np.rot90([[0, 1, 0]] * 3)]
_P3 = [np.zeros((3, 3, 3)) for i in range(9)]
_P3[0][:, :, 1] = 1
_P3[1][:, 1, :] = 1
_P3[2][1, :, :] = 1
_P3[3][:, [0, 1, 2], [0, 1, 2]] = 1
_P3[4][:, [0, 1, 2], [2, 1, 0]] = 1
_P3[5][[0, 1, 2], :, [0, 1, 2]] = 1
_P3[6][[0, 1, 2], :, [2, 1, 0]] = 1
_P3[7][[0, 1, 2], [0, 1, 2], :] = 1
_P3[8][[0, 1, 2], [2, 1, 0], :] = 1

def sup_inf(u):
    if False:
        print('Hello World!')
    'SI operator.'
    if np.ndim(u) == 2:
        P = _P2
    elif np.ndim(u) == 3:
        P = _P3
    else:
        raise ValueError('u has an invalid number of dimensions (should be 2 or 3)')
    erosions = []
    for P_i in P:
        erosions.append(ndi.binary_erosion(u, P_i).astype(np.int8))
    return np.stack(erosions, axis=0).max(0)

def inf_sup(u):
    if False:
        for i in range(10):
            print('nop')
    'IS operator.'
    if np.ndim(u) == 2:
        P = _P2
    elif np.ndim(u) == 3:
        P = _P3
    else:
        raise ValueError('u has an invalid number of dimensions (should be 2 or 3)')
    dilations = []
    for P_i in P:
        dilations.append(ndi.binary_dilation(u, P_i).astype(np.int8))
    return np.stack(dilations, axis=0).min(0)
_curvop = _fcycle([lambda u: sup_inf(inf_sup(u)), lambda u: inf_sup(sup_inf(u))])

def _check_input(image, init_level_set):
    if False:
        for i in range(10):
            print('nop')
    'Check that shapes of `image` and `init_level_set` match.'
    check_nD(image, [2, 3])
    if len(image.shape) != len(init_level_set.shape):
        raise ValueError('The dimensions of the initial level set do not match the dimensions of the image.')

def _init_level_set(init_level_set, image_shape):
    if False:
        print('Hello World!')
    'Auxiliary function for initializing level sets with a string.\n\n    If `init_level_set` is not a string, it is returned as is.\n    '
    if isinstance(init_level_set, str):
        if init_level_set == 'checkerboard':
            res = checkerboard_level_set(image_shape)
        elif init_level_set == 'disk':
            res = disk_level_set(image_shape)
        else:
            raise ValueError("`init_level_set` not in ['checkerboard', 'disk']")
    else:
        res = init_level_set
    return res

def disk_level_set(image_shape, *, center=None, radius=None):
    if False:
        for i in range(10):
            print('nop')
    'Create a disk level set with binary values.\n\n    Parameters\n    ----------\n    image_shape : tuple of positive integers\n        Shape of the image\n    center : tuple of positive integers, optional\n        Coordinates of the center of the disk given in (row, column). If not\n        given, it defaults to the center of the image.\n    radius : float, optional\n        Radius of the disk. If not given, it is set to the 75% of the\n        smallest image dimension.\n\n    Returns\n    -------\n    out : array with shape `image_shape`\n        Binary level set of the disk with the given `radius` and `center`.\n\n    See Also\n    --------\n    checkerboard_level_set\n    '
    if center is None:
        center = tuple((i // 2 for i in image_shape))
    if radius is None:
        radius = min(image_shape) * 3.0 / 8.0
    grid = np.mgrid[[slice(i) for i in image_shape]]
    grid = (grid.T - center).T
    phi = radius - np.sqrt(np.sum(grid ** 2, 0))
    res = np.int8(phi > 0)
    return res

def checkerboard_level_set(image_shape, square_size=5):
    if False:
        return 10
    'Create a checkerboard level set with binary values.\n\n    Parameters\n    ----------\n    image_shape : tuple of positive integers\n        Shape of the image.\n    square_size : int, optional\n        Size of the squares of the checkerboard. It defaults to 5.\n\n    Returns\n    -------\n    out : array with shape `image_shape`\n        Binary level set of the checkerboard.\n\n    See Also\n    --------\n    disk_level_set\n    '
    grid = np.mgrid[[slice(i) for i in image_shape]]
    grid = grid // square_size
    grid = grid & 1
    checkerboard = np.bitwise_xor.reduce(grid, axis=0)
    res = np.int8(checkerboard)
    return res

def inverse_gaussian_gradient(image, alpha=100.0, sigma=5.0):
    if False:
        for i in range(10):
            print('nop')
    'Inverse of gradient magnitude.\n\n    Compute the magnitude of the gradients in the image and then inverts the\n    result in the range [0, 1]. Flat areas are assigned values close to 1,\n    while areas close to borders are assigned values close to 0.\n\n    This function or a similar one defined by the user should be applied over\n    the image as a preprocessing step before calling\n    `morphological_geodesic_active_contour`.\n\n    Parameters\n    ----------\n    image : (M, N) or (L, M, N) array\n        Grayscale image or volume.\n    alpha : float, optional\n        Controls the steepness of the inversion. A larger value will make the\n        transition between the flat areas and border areas steeper in the\n        resulting array.\n    sigma : float, optional\n        Standard deviation of the Gaussian filter applied over the image.\n\n    Returns\n    -------\n    gimage : (M, N) or (L, M, N) array\n        Preprocessed image (or volume) suitable for\n        `morphological_geodesic_active_contour`.\n    '
    gradnorm = ndi.gaussian_gradient_magnitude(image, sigma, mode='nearest')
    return 1.0 / np.sqrt(1.0 + alpha * gradnorm)

def morphological_chan_vese(image, num_iter, init_level_set='checkerboard', smoothing=1, lambda1=1, lambda2=1, iter_callback=lambda x: None):
    if False:
        print('Hello World!')
    "Morphological Active Contours without Edges (MorphACWE)\n\n    Active contours without edges implemented with morphological operators. It\n    can be used to segment objects in images and volumes without well defined\n    borders. It is required that the inside of the object looks different on\n    average than the outside (i.e., the inner area of the object should be\n    darker or lighter than the outer area on average).\n\n    Parameters\n    ----------\n    image : (M, N) or (L, M, N) array\n        Grayscale image or volume to be segmented.\n    num_iter : uint\n        Number of num_iter to run\n    init_level_set : str, (M, N) array, or (L, M, N) array\n        Initial level set. If an array is given, it will be binarized and used\n        as the initial level set. If a string is given, it defines the method\n        to generate a reasonable initial level set with the shape of the\n        `image`. Accepted values are 'checkerboard' and 'disk'. See the\n        documentation of `checkerboard_level_set` and `disk_level_set`\n        respectively for details about how these level sets are created.\n    smoothing : uint, optional\n        Number of times the smoothing operator is applied per iteration.\n        Reasonable values are around 1-4. Larger values lead to smoother\n        segmentations.\n    lambda1 : float, optional\n        Weight parameter for the outer region. If `lambda1` is larger than\n        `lambda2`, the outer region will contain a larger range of values than\n        the inner region.\n    lambda2 : float, optional\n        Weight parameter for the inner region. If `lambda2` is larger than\n        `lambda1`, the inner region will contain a larger range of values than\n        the outer region.\n    iter_callback : function, optional\n        If given, this function is called once per iteration with the current\n        level set as the only argument. This is useful for debugging or for\n        plotting intermediate results during the evolution.\n\n    Returns\n    -------\n    out : (M, N) or (L, M, N) array\n        Final segmentation (i.e., the final level set)\n\n    See Also\n    --------\n    disk_level_set, checkerboard_level_set\n\n    Notes\n    -----\n    This is a version of the Chan-Vese algorithm that uses morphological\n    operators instead of solving a partial differential equation (PDE) for the\n    evolution of the contour. The set of morphological operators used in this\n    algorithm are proved to be infinitesimally equivalent to the Chan-Vese PDE\n    (see [1]_). However, morphological operators are do not suffer from the\n    numerical stability issues typically found in PDEs (it is not necessary to\n    find the right time step for the evolution), and are computationally\n    faster.\n\n    The algorithm and its theoretical derivation are described in [1]_.\n\n    References\n    ----------\n    .. [1] A Morphological Approach to Curvature-based Evolution of Curves and\n           Surfaces, Pablo Márquez-Neila, Luis Baumela, Luis Álvarez. In IEEE\n           Transactions on Pattern Analysis and Machine Intelligence (PAMI),\n           2014, :DOI:`10.1109/TPAMI.2013.106`\n    "
    init_level_set = _init_level_set(init_level_set, image.shape)
    _check_input(image, init_level_set)
    u = np.int8(init_level_set > 0)
    iter_callback(u)
    for _ in range(num_iter):
        c0 = (image * (1 - u)).sum() / float((1 - u).sum() + 1e-08)
        c1 = (image * u).sum() / float(u.sum() + 1e-08)
        du = np.gradient(u)
        abs_du = np.abs(du).sum(0)
        aux = abs_du * (lambda1 * (image - c1) ** 2 - lambda2 * (image - c0) ** 2)
        u[aux < 0] = 1
        u[aux > 0] = 0
        for _ in range(smoothing):
            u = _curvop(u)
        iter_callback(u)
    return u

def morphological_geodesic_active_contour(gimage, num_iter, init_level_set='disk', smoothing=1, threshold='auto', balloon=0, iter_callback=lambda x: None):
    if False:
        i = 10
        return i + 15
    "Morphological Geodesic Active Contours (MorphGAC).\n\n    Geodesic active contours implemented with morphological operators. It can\n    be used to segment objects with visible but noisy, cluttered, broken\n    borders.\n\n    Parameters\n    ----------\n    gimage : (M, N) or (L, M, N) array\n        Preprocessed image or volume to be segmented. This is very rarely the\n        original image. Instead, this is usually a preprocessed version of the\n        original image that enhances and highlights the borders (or other\n        structures) of the object to segment.\n        :func:`morphological_geodesic_active_contour` will try to stop the contour\n        evolution in areas where `gimage` is small. See\n        :func:`inverse_gaussian_gradient` as an example function to\n        perform this preprocessing. Note that the quality of\n        :func:`morphological_geodesic_active_contour` might greatly depend on this\n        preprocessing.\n    num_iter : uint\n        Number of num_iter to run.\n    init_level_set : str, (M, N) array, or (L, M, N) array\n        Initial level set. If an array is given, it will be binarized and used\n        as the initial level set. If a string is given, it defines the method\n        to generate a reasonable initial level set with the shape of the\n        `image`. Accepted values are 'checkerboard' and 'disk'. See the\n        documentation of `checkerboard_level_set` and `disk_level_set`\n        respectively for details about how these level sets are created.\n    smoothing : uint, optional\n        Number of times the smoothing operator is applied per iteration.\n        Reasonable values are around 1-4. Larger values lead to smoother\n        segmentations.\n    threshold : float, optional\n        Areas of the image with a value smaller than this threshold will be\n        considered borders. The evolution of the contour will stop in these\n        areas.\n    balloon : float, optional\n        Balloon force to guide the contour in non-informative areas of the\n        image, i.e., areas where the gradient of the image is too small to push\n        the contour towards a border. A negative value will shrink the contour,\n        while a positive value will expand the contour in these areas. Setting\n        this to zero will disable the balloon force.\n    iter_callback : function, optional\n        If given, this function is called once per iteration with the current\n        level set as the only argument. This is useful for debugging or for\n        plotting intermediate results during the evolution.\n\n    Returns\n    -------\n    out : (M, N) or (L, M, N) array\n        Final segmentation (i.e., the final level set)\n\n    See Also\n    --------\n    inverse_gaussian_gradient, disk_level_set, checkerboard_level_set\n\n    Notes\n    -----\n    This is a version of the Geodesic Active Contours (GAC) algorithm that uses\n    morphological operators instead of solving partial differential equations\n    (PDEs) for the evolution of the contour. The set of morphological operators\n    used in this algorithm are proved to be infinitesimally equivalent to the\n    GAC PDEs (see [1]_). However, morphological operators are do not suffer\n    from the numerical stability issues typically found in PDEs (e.g., it is\n    not necessary to find the right time step for the evolution), and are\n    computationally faster.\n\n    The algorithm and its theoretical derivation are described in [1]_.\n\n    References\n    ----------\n    .. [1] A Morphological Approach to Curvature-based Evolution of Curves and\n           Surfaces, Pablo Márquez-Neila, Luis Baumela, Luis Álvarez. In IEEE\n           Transactions on Pattern Analysis and Machine Intelligence (PAMI),\n           2014, :DOI:`10.1109/TPAMI.2013.106`\n    "
    image = gimage
    init_level_set = _init_level_set(init_level_set, image.shape)
    _check_input(image, init_level_set)
    if threshold == 'auto':
        threshold = np.percentile(image, 40)
    structure = np.ones((3,) * len(image.shape), dtype=np.int8)
    dimage = np.gradient(image)
    if balloon != 0:
        threshold_mask_balloon = image > threshold / np.abs(balloon)
    u = np.int8(init_level_set > 0)
    iter_callback(u)
    for _ in range(num_iter):
        if balloon > 0:
            aux = ndi.binary_dilation(u, structure)
        elif balloon < 0:
            aux = ndi.binary_erosion(u, structure)
        if balloon != 0:
            u[threshold_mask_balloon] = aux[threshold_mask_balloon]
        aux = np.zeros_like(image)
        du = np.gradient(u)
        for (el1, el2) in zip(dimage, du):
            aux += el1 * el2
        u[aux > 0] = 1
        u[aux < 0] = 0
        for _ in range(smoothing):
            u = _curvop(u)
        iter_callback(u)
    return u