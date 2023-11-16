"""extrema.py - local minima and maxima

This module provides functions to find local maxima and minima of an image.
Here, local maxima (minima) are defined as connected sets of pixels with equal
gray level which is strictly greater (smaller) than the gray level of all
pixels in direct neighborhood of the connected set. In addition, the module
provides the related functions h-maxima and h-minima.

Soille, P. (2003). Morphological Image Analysis: Principles and Applications
(2nd ed.), Chapter 6. Springer-Verlag New York, Inc.
"""
import numpy as np
from .._shared.utils import warn
from ..util import dtype_limits, invert, crop
from . import grayreconstruct, _util
from ._extrema_cy import _local_maxima

def _add_constant_clip(image, const_value):
    if False:
        return 10
    'Add constant to the image while handling overflow issues gracefully.'
    (min_dtype, max_dtype) = dtype_limits(image, clip_negative=False)
    if const_value > max_dtype - min_dtype:
        raise ValueError('The added constant is not compatiblewith the image data type.')
    result = image + const_value
    result[image > max_dtype - const_value] = max_dtype
    return result

def _subtract_constant_clip(image, const_value):
    if False:
        for i in range(10):
            print('nop')
    'Subtract constant from image while handling underflow issues.'
    (min_dtype, max_dtype) = dtype_limits(image, clip_negative=False)
    if const_value > max_dtype - min_dtype:
        raise ValueError('The subtracted constant is not compatiblewith the image data type.')
    result = image - const_value
    result[image < const_value + min_dtype] = min_dtype
    return result

def h_maxima(image, h, footprint=None):
    if False:
        print('Hello World!')
    'Determine all maxima of the image with height >= h.\n\n    The local maxima are defined as connected sets of pixels with equal\n    gray level strictly greater than the gray level of all pixels in direct\n    neighborhood of the set.\n\n    A local maximum M of height h is a local maximum for which\n    there is at least one path joining M with an equal or higher local maximum\n    on which the minimal value is f(M) - h (i.e. the values along the path\n    are not decreasing by more than h with respect to the maximum\'s value)\n    and no path to an equal or higher local maximum for which the minimal\n    value is greater.\n\n    The global maxima of the image are also found by this function.\n\n    Parameters\n    ----------\n    image : ndarray\n        The input image for which the maxima are to be calculated.\n    h : unsigned integer\n        The minimal height of all extracted maxima.\n    footprint : ndarray, optional\n        The neighborhood expressed as an n-D array of 1\'s and 0\'s.\n        Default is the ball of radius 1 according to the maximum norm\n        (i.e. a 3x3 square for 2D images, a 3x3x3 cube for 3D images, etc.)\n\n    Returns\n    -------\n    h_max : ndarray\n        The local maxima of height >= h and the global maxima.\n        The resulting image is a binary image, where pixels belonging to\n        the determined maxima take value 1, the others take value 0.\n\n    See Also\n    --------\n    skimage.morphology.h_minima\n    skimage.morphology.local_maxima\n    skimage.morphology.local_minima\n\n    References\n    ----------\n    .. [1] Soille, P., "Morphological Image Analysis: Principles and\n           Applications" (Chapter 6), 2nd edition (2003), ISBN 3540429883.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from skimage.morphology import extrema\n\n    We create an image (quadratic function with a maximum in the center and\n    4 additional constant maxima.\n    The heights of the maxima are: 1, 21, 41, 61, 81\n\n    >>> w = 10\n    >>> x, y = np.mgrid[0:w,0:w]\n    >>> f = 20 - 0.2*((x - w/2)**2 + (y-w/2)**2)\n    >>> f[2:4,2:4] = 40; f[2:4,7:9] = 60; f[7:9,2:4] = 80; f[7:9,7:9] = 100\n    >>> f = f.astype(int)\n\n    We can calculate all maxima with a height of at least 40:\n\n    >>> maxima = extrema.h_maxima(f, 40)\n\n    The resulting image will contain 3 local maxima.\n    '
    if h > np.ptp(image):
        return np.zeros(image.shape, dtype=np.uint8)
    if np.issubdtype(type(h), np.floating) and np.issubdtype(image.dtype, np.integer):
        if h % 1 != 0:
            warn('possible precision loss converting image to floating point. To silence this warning, ensure image and h have same data type.', stacklevel=2)
            image = image.astype(float)
        else:
            h = image.dtype.type(h)
    if h == 0:
        raise ValueError('h = 0 is ambiguous, use local_maxima() instead?')
    if np.issubdtype(image.dtype, np.floating):
        resolution = 2 * np.finfo(image.dtype).resolution * np.abs(image)
        shifted_img = image - h - resolution
    else:
        shifted_img = _subtract_constant_clip(image, h)
    rec_img = grayreconstruct.reconstruction(shifted_img, image, method='dilation', footprint=footprint)
    residue_img = image - rec_img
    return (residue_img >= h).astype(np.uint8)

def h_minima(image, h, footprint=None):
    if False:
        i = 10
        return i + 15
    'Determine all minima of the image with depth >= h.\n\n    The local minima are defined as connected sets of pixels with equal\n    gray level strictly smaller than the gray levels of all pixels in direct\n    neighborhood of the set.\n\n    A local minimum M of depth h is a local minimum for which\n    there is at least one path joining M with an equal or lower local minimum\n    on which the maximal value is f(M) + h (i.e. the values along the path\n    are not increasing by more than h with respect to the minimum\'s value)\n    and no path to an equal or lower local minimum for which the maximal\n    value is smaller.\n\n    The global minima of the image are also found by this function.\n\n    Parameters\n    ----------\n    image : ndarray\n        The input image for which the minima are to be calculated.\n    h : unsigned integer\n        The minimal depth of all extracted minima.\n    footprint : ndarray, optional\n        The neighborhood expressed as an n-D array of 1\'s and 0\'s.\n        Default is the ball of radius 1 according to the maximum norm\n        (i.e. a 3x3 square for 2D images, a 3x3x3 cube for 3D images, etc.)\n\n    Returns\n    -------\n    h_min : ndarray\n        The local minima of depth >= h and the global minima.\n        The resulting image is a binary image, where pixels belonging to\n        the determined minima take value 1, the others take value 0.\n\n    See Also\n    --------\n    skimage.morphology.h_maxima\n    skimage.morphology.local_maxima\n    skimage.morphology.local_minima\n\n    References\n    ----------\n    .. [1] Soille, P., "Morphological Image Analysis: Principles and\n           Applications" (Chapter 6), 2nd edition (2003), ISBN 3540429883.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from skimage.morphology import extrema\n\n    We create an image (quadratic function with a minimum in the center and\n    4 additional constant maxima.\n    The depth of the minima are: 1, 21, 41, 61, 81\n\n    >>> w = 10\n    >>> x, y = np.mgrid[0:w,0:w]\n    >>> f = 180 + 0.2*((x - w/2)**2 + (y-w/2)**2)\n    >>> f[2:4,2:4] = 160; f[2:4,7:9] = 140; f[7:9,2:4] = 120; f[7:9,7:9] = 100\n    >>> f = f.astype(int)\n\n    We can calculate all minima with a depth of at least 40:\n\n    >>> minima = extrema.h_minima(f, 40)\n\n    The resulting image will contain 3 local minima.\n    '
    if h > np.ptp(image):
        return np.zeros(image.shape, dtype=np.uint8)
    if np.issubdtype(type(h), np.floating) and np.issubdtype(image.dtype, np.integer):
        if h % 1 != 0:
            warn('possible precision loss converting image to floating point. To silence this warning, ensure image and h have same data type.', stacklevel=2)
            image = image.astype(float)
        else:
            h = image.dtype.type(h)
    if h == 0:
        raise ValueError('h = 0 is ambiguous, use local_minima() instead?')
    if np.issubdtype(image.dtype, np.floating):
        resolution = 2 * np.finfo(image.dtype).resolution * np.abs(image)
        shifted_img = image + h + resolution
    else:
        shifted_img = _add_constant_clip(image, h)
    rec_img = grayreconstruct.reconstruction(shifted_img, image, method='erosion', footprint=footprint)
    residue_img = rec_img - image
    return (residue_img >= h).astype(np.uint8)

def local_maxima(image, footprint=None, connectivity=None, indices=False, allow_borders=True):
    if False:
        i = 10
        return i + 15
    "Find local maxima of n-dimensional array.\n\n    The local maxima are defined as connected sets of pixels with equal gray\n    level (plateaus) strictly greater than the gray levels of all pixels in the\n    neighborhood.\n\n    Parameters\n    ----------\n    image : ndarray\n        An n-dimensional array.\n    footprint : ndarray, optional\n        The footprint (structuring element) used to determine the neighborhood\n        of each evaluated pixel (``True`` denotes a connected pixel). It must\n        be a boolean array and have the same number of dimensions as `image`.\n        If neither `footprint` nor `connectivity` are given, all adjacent\n        pixels are considered as part of the neighborhood.\n    connectivity : int, optional\n        A number used to determine the neighborhood of each evaluated pixel.\n        Adjacent pixels whose squared distance from the center is less than or\n        equal to `connectivity` are considered neighbors. Ignored if\n        `footprint` is not None.\n    indices : bool, optional\n        If True, the output will be a tuple of one-dimensional arrays\n        representing the indices of local maxima in each dimension. If False,\n        the output will be a boolean array with the same shape as `image`.\n    allow_borders : bool, optional\n        If true, plateaus that touch the image border are valid maxima.\n\n    Returns\n    -------\n    maxima : ndarray or tuple[ndarray]\n        If `indices` is false, a boolean array with the same shape as `image`\n        is returned with ``True`` indicating the position of local maxima\n        (``False`` otherwise). If `indices` is true, a tuple of one-dimensional\n        arrays containing the coordinates (indices) of all found maxima.\n\n    Warns\n    -----\n    UserWarning\n        If `allow_borders` is false and any dimension of the given `image` is\n        shorter than 3 samples, maxima can't exist and a warning is shown.\n\n    See Also\n    --------\n    skimage.morphology.local_minima\n    skimage.morphology.h_maxima\n    skimage.morphology.h_minima\n\n    Notes\n    -----\n    This function operates on the following ideas:\n\n    1. Make a first pass over the image's last dimension and flag candidates\n       for local maxima by comparing pixels in only one direction.\n       If the pixels aren't connected in the last dimension all pixels are\n       flagged as candidates instead.\n\n    For each candidate:\n\n    2. Perform a flood-fill to find all connected pixels that have the same\n       gray value and are part of the plateau.\n    3. Consider the connected neighborhood of a plateau: if no bordering sample\n       has a higher gray level, mark the plateau as a definite local maximum.\n\n    Examples\n    --------\n    >>> from skimage.morphology import local_maxima\n    >>> image = np.zeros((4, 7), dtype=int)\n    >>> image[1:3, 1:3] = 1\n    >>> image[3, 0] = 1\n    >>> image[1:3, 4:6] = 2\n    >>> image[3, 6] = 3\n    >>> image\n    array([[0, 0, 0, 0, 0, 0, 0],\n           [0, 1, 1, 0, 2, 2, 0],\n           [0, 1, 1, 0, 2, 2, 0],\n           [1, 0, 0, 0, 0, 0, 3]])\n\n    Find local maxima by comparing to all neighboring pixels (maximal\n    connectivity):\n\n    >>> local_maxima(image)\n    array([[False, False, False, False, False, False, False],\n           [False,  True,  True, False, False, False, False],\n           [False,  True,  True, False, False, False, False],\n           [ True, False, False, False, False, False,  True]])\n    >>> local_maxima(image, indices=True)\n    (array([1, 1, 2, 2, 3, 3]), array([1, 2, 1, 2, 0, 6]))\n\n    Find local maxima without comparing to diagonal pixels (connectivity 1):\n\n    >>> local_maxima(image, connectivity=1)\n    array([[False, False, False, False, False, False, False],\n           [False,  True,  True, False,  True,  True, False],\n           [False,  True,  True, False,  True,  True, False],\n           [ True, False, False, False, False, False,  True]])\n\n    and exclude maxima that border the image edge:\n\n    >>> local_maxima(image, connectivity=1, allow_borders=False)\n    array([[False, False, False, False, False, False, False],\n           [False,  True,  True, False,  True,  True, False],\n           [False,  True,  True, False,  True,  True, False],\n           [False, False, False, False, False, False, False]])\n    "
    image = np.asarray(image, order='C')
    if image.size == 0:
        if indices:
            return np.nonzero(image)
        else:
            return np.zeros(image.shape, dtype=bool)
    if allow_borders:
        image = np.pad(image, 1, mode='constant', constant_values=image.min())
    flags = np.zeros(image.shape, dtype=np.uint8)
    _util._set_border_values(flags, value=3)
    if any((s < 3 for s in image.shape)):
        warn("maxima can't exist for an image with any dimension smaller 3 if borders aren't allowed", stacklevel=3)
    else:
        footprint = _util._resolve_neighborhood(footprint, connectivity, image.ndim)
        neighbor_offsets = _util._offsets_to_raveled_neighbors(image.shape, footprint, center=(1,) * image.ndim)
        try:
            _local_maxima(image.ravel(), flags.ravel(), neighbor_offsets)
        except TypeError:
            if image.dtype == np.float16:
                raise TypeError('dtype of `image` is float16 which is not supported, try upcasting to float32')
            else:
                raise
    if allow_borders:
        flags = crop(flags, 1)
    else:
        _util._set_border_values(flags, value=0)
    if indices:
        return np.nonzero(flags)
    else:
        return flags.view(bool)

def local_minima(image, footprint=None, connectivity=None, indices=False, allow_borders=True):
    if False:
        return 10
    "Find local minima of n-dimensional array.\n\n    The local minima are defined as connected sets of pixels with equal gray\n    level (plateaus) strictly smaller than the gray levels of all pixels in the\n    neighborhood.\n\n    Parameters\n    ----------\n    image : ndarray\n        An n-dimensional array.\n    footprint : ndarray, optional\n        The footprint (structuring element) used to determine the neighborhood\n        of each evaluated pixel (``True`` denotes a connected pixel). It must\n        be a boolean array and have the same number of dimensions as `image`.\n        If neither `footprint` nor `connectivity` are given, all adjacent\n        pixels are considered as part of the neighborhood.\n    connectivity : int, optional\n        A number used to determine the neighborhood of each evaluated pixel.\n        Adjacent pixels whose squared distance from the center is less than or\n        equal to `connectivity` are considered neighbors. Ignored if\n        `footprint` is not None.\n    indices : bool, optional\n        If True, the output will be a tuple of one-dimensional arrays\n        representing the indices of local minima in each dimension. If False,\n        the output will be a boolean array with the same shape as `image`.\n    allow_borders : bool, optional\n        If true, plateaus that touch the image border are valid minima.\n\n    Returns\n    -------\n    minima : ndarray or tuple[ndarray]\n        If `indices` is false, a boolean array with the same shape as `image`\n        is returned with ``True`` indicating the position of local minima\n        (``False`` otherwise). If `indices` is true, a tuple of one-dimensional\n        arrays containing the coordinates (indices) of all found minima.\n\n    See Also\n    --------\n    skimage.morphology.local_maxima\n    skimage.morphology.h_maxima\n    skimage.morphology.h_minima\n\n    Notes\n    -----\n    This function operates on the following ideas:\n\n    1. Make a first pass over the image's last dimension and flag candidates\n       for local minima by comparing pixels in only one direction.\n       If the pixels aren't connected in the last dimension all pixels are\n       flagged as candidates instead.\n\n    For each candidate:\n\n    2. Perform a flood-fill to find all connected pixels that have the same\n       gray value and are part of the plateau.\n    3. Consider the connected neighborhood of a plateau: if no bordering sample\n       has a smaller gray level, mark the plateau as a definite local minimum.\n\n    Examples\n    --------\n    >>> from skimage.morphology import local_minima\n    >>> image = np.zeros((4, 7), dtype=int)\n    >>> image[1:3, 1:3] = -1\n    >>> image[3, 0] = -1\n    >>> image[1:3, 4:6] = -2\n    >>> image[3, 6] = -3\n    >>> image\n    array([[ 0,  0,  0,  0,  0,  0,  0],\n           [ 0, -1, -1,  0, -2, -2,  0],\n           [ 0, -1, -1,  0, -2, -2,  0],\n           [-1,  0,  0,  0,  0,  0, -3]])\n\n    Find local minima by comparing to all neighboring pixels (maximal\n    connectivity):\n\n    >>> local_minima(image)\n    array([[False, False, False, False, False, False, False],\n           [False,  True,  True, False, False, False, False],\n           [False,  True,  True, False, False, False, False],\n           [ True, False, False, False, False, False,  True]])\n    >>> local_minima(image, indices=True)\n    (array([1, 1, 2, 2, 3, 3]), array([1, 2, 1, 2, 0, 6]))\n\n    Find local minima without comparing to diagonal pixels (connectivity 1):\n\n    >>> local_minima(image, connectivity=1)\n    array([[False, False, False, False, False, False, False],\n           [False,  True,  True, False,  True,  True, False],\n           [False,  True,  True, False,  True,  True, False],\n           [ True, False, False, False, False, False,  True]])\n\n    and exclude minima that border the image edge:\n\n    >>> local_minima(image, connectivity=1, allow_borders=False)\n    array([[False, False, False, False, False, False, False],\n           [False,  True,  True, False,  True,  True, False],\n           [False,  True,  True, False,  True,  True, False],\n           [False, False, False, False, False, False, False]])\n    "
    return local_maxima(image=invert(image), footprint=footprint, connectivity=connectivity, indices=indices, allow_borders=allow_borders)