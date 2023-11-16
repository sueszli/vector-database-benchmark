"""
Grayscale morphological operations
"""
import warnings
import numpy as np
from scipy import ndimage as ndi
from .footprints import _footprint_is_sequence, mirror_footprint, pad_footprint
from .misc import default_footprint
from .._shared.utils import DEPRECATED
__all__ = ['erosion', 'dilation', 'opening', 'closing', 'white_tophat', 'black_tophat']

def _iterate_gray_func(gray_func, image, footprints, out, mode, cval):
    if False:
        for i in range(10):
            print('nop')
    'Helper to call `gray_func` for each footprint in a sequence.\n\n    `gray_func` is a morphology function that accepts `footprint`, `output`,\n    `mode` and `cval` keyword arguments (e.g. `scipy.ndimage.grey_erosion`).\n    '
    (fp, num_iter) = footprints[0]
    gray_func(image, footprint=fp, output=out, mode=mode, cval=cval)
    for _ in range(1, num_iter):
        gray_func(out.copy(), footprint=fp, output=out, mode=mode, cval=cval)
    for (fp, num_iter) in footprints[1:]:
        for _ in range(num_iter):
            gray_func(out.copy(), footprint=fp, output=out, mode=mode, cval=cval)
    return out

def _shift_footprint(footprint, shift_x, shift_y):
    if False:
        return 10
    'Shift the binary image `footprint` in the left and/or up.\n\n    This only affects 2D footprints with even number of rows\n    or columns.\n\n    Parameters\n    ----------\n    footprint : 2D array, shape (M, N)\n        The input footprint.\n    shift_x, shift_y : bool or None\n        Whether to move `footprint` along each axis. If ``None``, the\n        array is not modified along that dimension.\n\n    Returns\n    -------\n    out : 2D array, shape (M + int(shift_x), N + int(shift_y))\n        The shifted footprint.\n    '
    footprint = np.asarray(footprint)
    if footprint.ndim != 2:
        return footprint
    (m, n) = footprint.shape
    if m % 2 == 0:
        extra_row = np.zeros((1, n), footprint.dtype)
        if shift_x:
            footprint = np.vstack((footprint, extra_row))
        else:
            footprint = np.vstack((extra_row, footprint))
        m += 1
    if n % 2 == 0:
        extra_col = np.zeros((m, 1), footprint.dtype)
        if shift_y:
            footprint = np.hstack((footprint, extra_col))
        else:
            footprint = np.hstack((extra_col, footprint))
    return footprint

def _shift_footprints(footprint, shift_x, shift_y):
    if False:
        while True:
            i = 10
    "Shifts the footprints, whether it's a single array or a sequence.\n\n    See `_shift_footprint`, which is called for each array in the sequence.\n    "
    if shift_x is DEPRECATED and shift_y is DEPRECATED:
        return footprint
    warning_msg = 'The parameters `shift_x` and `shift_y` are deprecated since v0.23 and will be removed in v0.26. Use `pad_footprint` or modify the footprintmanually instead.'
    warnings.warn(warning_msg, FutureWarning, stacklevel=4)
    if _footprint_is_sequence(footprint):
        return tuple(((_shift_footprint(fp, shift_x, shift_y), n) for (fp, n) in footprint))
    return _shift_footprint(footprint, shift_x, shift_y)

def _min_max_to_constant_mode(dtype, mode, cval):
    if False:
        i = 10
        return i + 15
    "Replace 'max' and 'min' with appropriate 'cval' and 'constant' mode."
    if mode == 'max':
        mode = 'constant'
        if np.issubdtype(dtype, bool):
            cval = True
        elif np.issubdtype(dtype, np.integer):
            cval = np.iinfo(dtype).max
        else:
            cval = np.inf
    elif mode == 'min':
        mode = 'constant'
        if np.issubdtype(dtype, bool):
            cval = False
        elif np.issubdtype(dtype, np.integer):
            cval = np.iinfo(dtype).min
        else:
            cval = -np.inf
    return (mode, cval)
_SUPPORTED_MODES = {'reflect', 'constant', 'nearest', 'mirror', 'wrap', 'max', 'min', 'ignore'}

@default_footprint
def erosion(image, footprint=None, out=None, shift_x=DEPRECATED, shift_y=DEPRECATED, *, mode='reflect', cval=0.0):
    if False:
        while True:
            i = 10
    "Return grayscale morphological erosion of an image.\n\n    Morphological erosion sets a pixel at (i,j) to the minimum over all pixels\n    in the neighborhood centered at (i,j). Erosion shrinks bright regions and\n    enlarges dark regions.\n\n    Parameters\n    ----------\n    image : ndarray\n        Image array.\n    footprint : ndarray or tuple, optional\n        The neighborhood expressed as a 2-D array of 1's and 0's.\n        If None, use a cross-shaped footprint (connectivity=1). The footprint\n        can also be provided as a sequence of smaller footprints as described\n        in the notes below.\n    out : ndarrays, optional\n        The array to store the result of the morphology. If None is\n        passed, a new array will be allocated.\n    mode : str, optional\n        The `mode` parameter determines how the array borders are handled.\n        Valid modes are: 'reflect', 'constant', 'nearest', 'mirror', 'wrap',\n        'max', 'min', or 'ignore'.\n        If 'max' or 'ignore', pixels outside the image domain are assumed\n        to be the maximum for the image's dtype, which causes them to not\n        influence the result. Default is 'reflect'.\n    cval : scalar, optional\n        Value to fill past edges of input if `mode` is 'constant'. Default\n        is 0.0.\n\n        .. versionadded:: 0.23\n            `mode` and `cval` were added in 0.23.\n\n    Returns\n    -------\n    eroded : array, same shape as `image`\n        The result of the morphological erosion.\n\n    Other Parameters\n    ----------------\n    shift_x, shift_y : DEPRECATED\n\n        .. deprecated:: 0.23\n\n    Notes\n    -----\n    For ``uint8`` (and ``uint16`` up to a certain bit-depth) data, the\n    lower algorithm complexity makes the :func:`skimage.filters.rank.minimum`\n    function more efficient for larger images and footprints.\n\n    The footprint can also be a provided as a sequence of 2-tuples where the\n    first element of each 2-tuple is a footprint ndarray and the second element\n    is an integer describing the number of times it should be iterated. For\n    example ``footprint=[(np.ones((9, 1)), 1), (np.ones((1, 9)), 1)]``\n    would apply a 9x1 footprint followed by a 1x9 footprint resulting in a net\n    effect that is the same as ``footprint=np.ones((9, 9))``, but with lower\n    computational cost. Most of the builtin footprints such as\n    :func:`skimage.morphology.disk` provide an option to automatically generate\n    a footprint sequence of this type.\n\n    For even-sized footprints, :func:`skimage.morphology.binary_erosion` and\n    this function produce an output that differs: one is shifted by one pixel\n    compared to the other.\n\n    Examples\n    --------\n    >>> # Erosion shrinks bright regions\n    >>> import numpy as np\n    >>> from skimage.morphology import square\n    >>> bright_square = np.array([[0, 0, 0, 0, 0],\n    ...                           [0, 1, 1, 1, 0],\n    ...                           [0, 1, 1, 1, 0],\n    ...                           [0, 1, 1, 1, 0],\n    ...                           [0, 0, 0, 0, 0]], dtype=np.uint8)\n    >>> erosion(bright_square, square(3))\n    array([[0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0],\n           [0, 0, 1, 0, 0],\n           [0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0]], dtype=uint8)\n\n    "
    if out is None:
        out = np.empty_like(image)
    if mode not in _SUPPORTED_MODES:
        raise ValueError(f'unsupported mode, got {mode!r}')
    if mode == 'ignore':
        mode = 'max'
    (mode, cval) = _min_max_to_constant_mode(image.dtype, mode, cval)
    footprint = _shift_footprints(footprint, shift_x, shift_y)
    footprint = pad_footprint(footprint, pad_end=False)
    if not _footprint_is_sequence(footprint):
        footprint = [(footprint, 1)]
    out = _iterate_gray_func(gray_func=ndi.grey_erosion, image=image, footprints=footprint, out=out, mode=mode, cval=cval)
    return out

@default_footprint
def dilation(image, footprint=None, out=None, shift_x=DEPRECATED, shift_y=DEPRECATED, *, mode='reflect', cval=0.0):
    if False:
        print('Hello World!')
    "Return grayscale morphological dilation of an image.\n\n    Morphological dilation sets the value of a pixel to the maximum over all\n    pixel values within a local neighborhood centered about it. The values\n    where the footprint is 1 define this neighborhood.\n    Dilation enlarges bright regions and shrinks dark regions.\n\n    Parameters\n    ----------\n    image : ndarray\n        Image array.\n    footprint : ndarray or tuple, optional\n        The neighborhood expressed as a 2-D array of 1's and 0's.\n        If None, use a cross-shaped footprint (connectivity=1). The footprint\n        can also be provided as a sequence of smaller footprints as described\n        in the notes below.\n    out : ndarray, optional\n        The array to store the result of the morphology. If None is\n        passed, a new array will be allocated.\n    mode : str, optional\n        The `mode` parameter determines how the array borders are handled.\n        Valid modes are: 'reflect', 'constant', 'nearest', 'mirror', 'wrap',\n        'max', 'min', or 'ignore'.\n        If 'min' or 'ignore', pixels outside the image domain are assumed\n        to be the maximum for the image's dtype, which causes them to not\n        influence the result. Default is 'reflect'.\n    cval : scalar, optional\n        Value to fill past edges of input if `mode` is 'constant'. Default\n        is 0.0.\n\n        .. versionadded:: 0.23\n            `mode` and `cval` were added in 0.23.\n\n    Returns\n    -------\n    dilated : uint8 array, same shape and type as `image`\n        The result of the morphological dilation.\n\n    Other Parameters\n    ----------------\n    shift_x, shift_y : DEPRECATED\n\n        .. deprecated:: 0.23\n\n    Notes\n    -----\n    For ``uint8`` (and ``uint16`` up to a certain bit-depth) data, the lower\n    algorithm complexity makes the :func:`skimage.filters.rank.maximum`\n    function more efficient for larger images and footprints.\n\n    The footprint can also be a provided as a sequence of 2-tuples where the\n    first element of each 2-tuple is a footprint ndarray and the second element\n    is an integer describing the number of times it should be iterated. For\n    example ``footprint=[(np.ones((9, 1)), 1), (np.ones((1, 9)), 1)]``\n    would apply a 9x1 footprint followed by a 1x9 footprint resulting in a net\n    effect that is the same as ``footprint=np.ones((9, 9))``, but with lower\n    computational cost. Most of the builtin footprints such as\n    :func:`skimage.morphology.disk` provide an option to automatically generate\n    a footprint sequence of this type.\n\n    For non-symmetric footprints, :func:`skimage.morphology.binary_dilation`\n    and :func:`skimage.morphology.dilation` produce an output that differs:\n    `binary_dilation` mirrors the footprint, whereas `dilation` does not.\n\n    Examples\n    --------\n    >>> # Dilation enlarges bright regions\n    >>> import numpy as np\n    >>> from skimage.morphology import square\n    >>> bright_pixel = np.array([[0, 0, 0, 0, 0],\n    ...                          [0, 0, 0, 0, 0],\n    ...                          [0, 0, 1, 0, 0],\n    ...                          [0, 0, 0, 0, 0],\n    ...                          [0, 0, 0, 0, 0]], dtype=np.uint8)\n    >>> dilation(bright_pixel, square(3))\n    array([[0, 0, 0, 0, 0],\n           [0, 1, 1, 1, 0],\n           [0, 1, 1, 1, 0],\n           [0, 1, 1, 1, 0],\n           [0, 0, 0, 0, 0]], dtype=uint8)\n\n    "
    if out is None:
        out = np.empty_like(image)
    if mode not in _SUPPORTED_MODES:
        raise ValueError(f'unsupported mode, got {mode!r}')
    if mode == 'ignore':
        mode = 'min'
    (mode, cval) = _min_max_to_constant_mode(image.dtype, mode, cval)
    footprint = _shift_footprints(footprint, shift_x, shift_y)
    footprint = pad_footprint(footprint, pad_end=False)
    footprint = mirror_footprint(footprint)
    if not _footprint_is_sequence(footprint):
        footprint = [(footprint, 1)]
    out = _iterate_gray_func(gray_func=ndi.grey_dilation, image=image, footprints=footprint, out=out, mode=mode, cval=cval)
    return out

@default_footprint
def opening(image, footprint=None, out=None, *, mode='reflect', cval=0.0):
    if False:
        print('Hello World!')
    'Return grayscale morphological opening of an image.\n\n    The morphological opening of an image is defined as an erosion followed by\n    a dilation. Opening can remove small bright spots (i.e. "salt") and connect\n    small dark cracks. This tends to "open" up (dark) gaps between (bright)\n    features.\n\n    Parameters\n    ----------\n    image : ndarray\n        Image array.\n    footprint : ndarray or tuple, optional\n        The neighborhood expressed as a 2-D array of 1\'s and 0\'s.\n        If None, use a cross-shaped footprint (connectivity=1). The footprint\n        can also be provided as a sequence of smaller footprints as described\n        in the notes below.\n    out : ndarray, optional\n        The array to store the result of the morphology. If None\n        is passed, a new array will be allocated.\n    mode : str, optional\n        The `mode` parameter determines how the array borders are handled.\n        Valid modes are: \'reflect\', \'constant\', \'nearest\', \'mirror\', \'wrap\',\n        \'max\', \'min\', or \'ignore\'.\n        If \'ignore\', pixels outside the image domain are assumed\n        to be the maximum for the image\'s dtype in the erosion, and minimum\n        in the dilation, which causes them to not influence the result.\n        Default is \'reflect\'.\n    cval : scalar, optional\n        Value to fill past edges of input if `mode` is \'constant\'. Default\n        is 0.0.\n\n        .. versionadded:: 0.23\n            `mode` and `cval` were added in 0.23.\n\n    Returns\n    -------\n    opening : array, same shape and type as `image`\n        The result of the morphological opening.\n\n    Notes\n    -----\n    The footprint can also be a provided as a sequence of 2-tuples where the\n    first element of each 2-tuple is a footprint ndarray and the second element\n    is an integer describing the number of times it should be iterated. For\n    example ``footprint=[(np.ones((9, 1)), 1), (np.ones((1, 9)), 1)]``\n    would apply a 9x1 footprint followed by a 1x9 footprint resulting in a net\n    effect that is the same as ``footprint=np.ones((9, 9))``, but with lower\n    computational cost. Most of the builtin footprints such as\n    :func:`skimage.morphology.disk` provide an option to automatically generate\n    a footprint sequence of this type.\n\n    Examples\n    --------\n    >>> # Open up gap between two bright regions (but also shrink regions)\n    >>> import numpy as np\n    >>> from skimage.morphology import square\n    >>> bad_connection = np.array([[1, 0, 0, 0, 1],\n    ...                            [1, 1, 0, 1, 1],\n    ...                            [1, 1, 1, 1, 1],\n    ...                            [1, 1, 0, 1, 1],\n    ...                            [1, 0, 0, 0, 1]], dtype=np.uint8)\n    >>> opening(bad_connection, square(3))\n    array([[0, 0, 0, 0, 0],\n           [1, 1, 0, 1, 1],\n           [1, 1, 0, 1, 1],\n           [1, 1, 0, 1, 1],\n           [0, 0, 0, 0, 0]], dtype=uint8)\n\n    '
    footprint = pad_footprint(footprint, pad_end=False)
    eroded = erosion(image, footprint, mode=mode, cval=cval)
    out = dilation(eroded, mirror_footprint(footprint), out=out, mode=mode, cval=cval)
    return out

@default_footprint
def closing(image, footprint=None, out=None, *, mode='reflect', cval=0.0):
    if False:
        print('Hello World!')
    'Return grayscale morphological closing of an image.\n\n    The morphological closing of an image is defined as a dilation followed by\n    an erosion. Closing can remove small dark spots (i.e. "pepper") and connect\n    small bright cracks. This tends to "close" up (dark) gaps between (bright)\n    features.\n\n    Parameters\n    ----------\n    image : ndarray\n        Image array.\n    footprint : ndarray or tuple, optional\n        The neighborhood expressed as a 2-D array of 1\'s and 0\'s.\n        If None, use a cross-shaped footprint (connectivity=1). The footprint\n        can also be provided as a sequence of smaller footprints as described\n        in the notes below.\n    out : ndarray, optional\n        The array to store the result of the morphology. If None,\n        a new array will be allocated.\n    mode : str, optional\n        The `mode` parameter determines how the array borders are handled.\n        Valid modes are: \'reflect\', \'constant\', \'nearest\', \'mirror\', \'wrap\',\n        \'max\', \'min\', or \'ignore\'.\n        If \'ignore\', pixels outside the image domain are assumed\n        to be the maximum for the image\'s dtype in the erosion, and minimum\n        in the dilation, which causes them to not influence the result.\n        Default is \'reflect\'.\n    cval : scalar, optional\n        Value to fill past edges of input if `mode` is \'constant\'. Default\n        is 0.0.\n\n        .. versionadded:: 0.23\n            `mode` and `cval` were added in 0.23.\n\n    Returns\n    -------\n    closing : array, same shape and type as `image`\n        The result of the morphological closing.\n\n    Notes\n    -----\n    The footprint can also be a provided as a sequence of 2-tuples where the\n    first element of each 2-tuple is a footprint ndarray and the second element\n    is an integer describing the number of times it should be iterated. For\n    example ``footprint=[(np.ones((9, 1)), 1), (np.ones((1, 9)), 1)]``\n    would apply a 9x1 footprint followed by a 1x9 footprint resulting in a net\n    effect that is the same as ``footprint=np.ones((9, 9))``, but with lower\n    computational cost. Most of the builtin footprints such as\n    :func:`skimage.morphology.disk` provide an option to automatically generate\n    a footprint sequence of this type.\n\n    Examples\n    --------\n    >>> # Close a gap between two bright lines\n    >>> import numpy as np\n    >>> from skimage.morphology import square\n    >>> broken_line = np.array([[0, 0, 0, 0, 0],\n    ...                         [0, 0, 0, 0, 0],\n    ...                         [1, 1, 0, 1, 1],\n    ...                         [0, 0, 0, 0, 0],\n    ...                         [0, 0, 0, 0, 0]], dtype=np.uint8)\n    >>> closing(broken_line, square(3))\n    array([[0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0],\n           [1, 1, 1, 1, 1],\n           [0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0]], dtype=uint8)\n\n    '
    footprint = pad_footprint(footprint, pad_end=False)
    dilated = dilation(image, footprint, mode=mode, cval=cval)
    out = erosion(dilated, mirror_footprint(footprint), out=out, mode=mode, cval=cval)
    return out

@default_footprint
def white_tophat(image, footprint=None, out=None, *, mode='reflect', cval=0.0):
    if False:
        i = 10
        return i + 15
    "Return white top hat of an image.\n\n    The white top hat of an image is defined as the image minus its\n    morphological opening. This operation returns the bright spots of the image\n    that are smaller than the footprint.\n\n    Parameters\n    ----------\n    image : ndarray\n        Image array.\n    footprint : ndarray or tuple, optional\n        The neighborhood expressed as a 2-D array of 1's and 0's.\n        If None, use a cross-shaped footprint (connectivity=1). The footprint\n        can also be provided as a sequence of smaller footprints as described\n        in the notes below.\n    out : ndarray, optional\n        The array to store the result of the morphology. If None\n        is passed, a new array will be allocated.\n    mode : str, optional\n        The `mode` parameter determines how the array borders are handled.\n        Valid modes are: 'reflect', 'constant', 'nearest', 'mirror', 'wrap',\n        'max', 'min', or 'ignore'. See :func:`skimage.morphology.opening`.\n        Default is 'reflect'.\n    cval : scalar, optional\n        Value to fill past edges of input if `mode` is 'constant'. Default\n        is 0.0.\n\n        .. versionadded:: 0.23\n            `mode` and `cval` were added in 0.23.\n\n    Returns\n    -------\n    out : array, same shape and type as `image`\n        The result of the morphological white top hat.\n\n    Notes\n    -----\n    The footprint can also be a provided as a sequence of 2-tuples where the\n    first element of each 2-tuple is a footprint ndarray and the second element\n    is an integer describing the number of times it should be iterated. For\n    example ``footprint=[(np.ones((9, 1)), 1), (np.ones((1, 9)), 1)]``\n    would apply a 9x1 footprint followed by a 1x9 footprint resulting in a net\n    effect that is the same as ``footprint=np.ones((9, 9))``, but with lower\n    computational cost. Most of the builtin footprints such as\n    :func:`skimage.morphology.disk` provide an option to automatically generate\n    a footprint sequence of this type.\n\n    See Also\n    --------\n    black_tophat\n\n    References\n    ----------\n    .. [1] https://en.wikipedia.org/wiki/Top-hat_transform\n\n    Examples\n    --------\n    >>> # Subtract gray background from bright peak\n    >>> import numpy as np\n    >>> from skimage.morphology import square\n    >>> bright_on_gray = np.array([[2, 3, 3, 3, 2],\n    ...                            [3, 4, 5, 4, 3],\n    ...                            [3, 5, 9, 5, 3],\n    ...                            [3, 4, 5, 4, 3],\n    ...                            [2, 3, 3, 3, 2]], dtype=np.uint8)\n    >>> white_tophat(bright_on_gray, square(3))\n    array([[0, 0, 0, 0, 0],\n           [0, 0, 1, 0, 0],\n           [0, 1, 5, 1, 0],\n           [0, 0, 1, 0, 0],\n           [0, 0, 0, 0, 0]], dtype=uint8)\n\n    "
    if out is image:
        opened = opening(image, footprint, mode=mode, cval=cval)
        if np.issubdtype(opened.dtype, bool):
            np.logical_xor(out, opened, out=out)
        else:
            out -= opened
        return out
    out = opening(image, footprint, out=out, mode=mode, cval=cval)
    if np.issubdtype(out.dtype, bool):
        np.logical_xor(image, out, out=out)
    else:
        np.subtract(image, out, out=out)
    return out

@default_footprint
def black_tophat(image, footprint=None, out=None, *, mode='reflect', cval=0.0):
    if False:
        print('Hello World!')
    "Return black top hat of an image.\n\n    The black top hat of an image is defined as its morphological closing minus\n    the original image. This operation returns the dark spots of the image that\n    are smaller than the footprint. Note that dark spots in the\n    original image are bright spots after the black top hat.\n\n    Parameters\n    ----------\n    image : ndarray\n        Image array.\n    footprint : ndarray or tuple, optional\n        The neighborhood expressed as a 2-D array of 1's and 0's.\n        If None, use a cross-shaped footprint (connectivity=1). The footprint\n        can also be provided as a sequence of smaller footprints as described\n        in the notes below.\n    out : ndarray, optional\n        The array to store the result of the morphology. If None\n        is passed, a new array will be allocated.\n    mode : str, optional\n        The `mode` parameter determines how the array borders are handled.\n        Valid modes are: 'reflect', 'constant', 'nearest', 'mirror', 'wrap',\n        'max', 'min', or 'ignore'. See :func:`skimage.morphology.closing`.\n        Default is 'reflect'.\n    cval : scalar, optional\n        Value to fill past edges of input if `mode` is 'constant'. Default\n        is 0.0.\n\n        .. versionadded:: 0.23\n            `mode` and `cval` were added in 0.23.\n\n    Returns\n    -------\n    out : array, same shape and type as `image`\n        The result of the morphological black top hat.\n\n    Notes\n    -----\n    The footprint can also be a provided as a sequence of 2-tuples where the\n    first element of each 2-tuple is a footprint ndarray and the second element\n    is an integer describing the number of times it should be iterated. For\n    example ``footprint=[(np.ones((9, 1)), 1), (np.ones((1, 9)), 1)]``\n    would apply a 9x1 footprint followed by a 1x9 footprint resulting in a net\n    effect that is the same as ``footprint=np.ones((9, 9))``, but with lower\n    computational cost. Most of the builtin footprints such as\n    :func:`skimage.morphology.disk` provide an option to automatically generate\n    a footprint sequence of this type.\n\n    See Also\n    --------\n    white_tophat\n\n    References\n    ----------\n    .. [1] https://en.wikipedia.org/wiki/Top-hat_transform\n\n    Examples\n    --------\n    >>> # Change dark peak to bright peak and subtract background\n    >>> import numpy as np\n    >>> from skimage.morphology import square\n    >>> dark_on_gray = np.array([[7, 6, 6, 6, 7],\n    ...                          [6, 5, 4, 5, 6],\n    ...                          [6, 4, 0, 4, 6],\n    ...                          [6, 5, 4, 5, 6],\n    ...                          [7, 6, 6, 6, 7]], dtype=np.uint8)\n    >>> black_tophat(dark_on_gray, square(3))\n    array([[0, 0, 0, 0, 0],\n           [0, 0, 1, 0, 0],\n           [0, 1, 5, 1, 0],\n           [0, 0, 1, 0, 0],\n           [0, 0, 0, 0, 0]], dtype=uint8)\n\n    "
    if out is image:
        closed = closing(image, footprint, mode=mode, cval=cval)
        if np.issubdtype(closed.dtype, bool):
            np.logical_xor(closed, out, out=out)
        else:
            np.subtract(closed, out, out=out)
        return out
    out = closing(image, footprint, out=out, mode=mode, cval=cval)
    if np.issubdtype(out.dtype, np.bool_):
        np.logical_xor(out, image, out=out)
    else:
        out -= image
    return out