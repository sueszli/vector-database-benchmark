"""
Algorithms for computing the skeleton of a binary image
"""
import numpy as np
from scipy import ndimage as ndi
from .._shared.utils import check_nD, deprecate_kwarg
from ..util import crop, img_as_ubyte
from ._skeletonize_3d_cy import _compute_thin_image
from ._skeletonize_cy import _fast_skeletonize, _skeletonize_loop, _table_lookup_index

def skeletonize(image, *, method=None):
    if False:
        return 10
    "Compute the skeleton of a binary image.\n\n    Thinning is used to reduce each connected component in a binary image\n    to a single-pixel wide skeleton.\n\n    Parameters\n    ----------\n    image : ndarray, 2D or 3D\n        An image containing the objects to be skeletonized. Zeros\n        represent background, nonzero values are foreground.\n    method : {'zhang', 'lee'}, optional\n        Which algorithm to use. Zhang's algorithm [Zha84]_ only works for\n        2D images, and is the default for 2D. Lee's algorithm [Lee94]_\n        works for 2D or 3D images and is the default for 3D.\n\n    Returns\n    -------\n    skeleton : ndarray\n        The thinned image.\n\n    See Also\n    --------\n    medial_axis\n\n    References\n    ----------\n    .. [Lee94] T.-C. Lee, R.L. Kashyap and C.-N. Chu, Building skeleton models\n           via 3-D medial surface/axis thinning algorithms.\n           Computer Vision, Graphics, and Image Processing, 56(6):462-478, 1994.\n\n    .. [Zha84] A fast parallel algorithm for thinning digital patterns,\n           T. Y. Zhang and C. Y. Suen, Communications of the ACM,\n           March 1984, Volume 27, Number 3.\n\n    Examples\n    --------\n    >>> X, Y = np.ogrid[0:9, 0:9]\n    >>> ellipse = (1./3 * (X - 4)**2 + (Y - 4)**2 < 3**2).astype(np.uint8)\n    >>> ellipse\n    array([[0, 0, 0, 1, 1, 1, 0, 0, 0],\n           [0, 0, 1, 1, 1, 1, 1, 0, 0],\n           [0, 0, 1, 1, 1, 1, 1, 0, 0],\n           [0, 0, 1, 1, 1, 1, 1, 0, 0],\n           [0, 0, 1, 1, 1, 1, 1, 0, 0],\n           [0, 0, 1, 1, 1, 1, 1, 0, 0],\n           [0, 0, 1, 1, 1, 1, 1, 0, 0],\n           [0, 0, 1, 1, 1, 1, 1, 0, 0],\n           [0, 0, 0, 1, 1, 1, 0, 0, 0]], dtype=uint8)\n    >>> skel = skeletonize(ellipse)\n    >>> skel.astype(np.uint8)\n    array([[0, 0, 0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 1, 0, 0, 0, 0],\n           [0, 0, 0, 0, 1, 0, 0, 0, 0],\n           [0, 0, 0, 0, 1, 0, 0, 0, 0],\n           [0, 0, 0, 0, 1, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8)\n\n    "
    if method not in {'zhang', 'lee', None}:
        raise ValueError(f'skeletonize method should be either "lee" or "zhang", got {method}.')
    if image.ndim == 2 and (method is None or method == 'zhang'):
        skeleton = skeletonize_2d(image.astype(bool, copy=False))
    elif image.ndim == 3 and method == 'zhang':
        raise ValueError('skeletonize method "zhang" only works for 2D images.')
    elif image.ndim == 3 or (image.ndim == 2 and method == 'lee'):
        skeleton = skeletonize_3d(image)
    else:
        raise ValueError(f'skeletonize requires a 2D or 3D image as input, got {image.ndim}D.')
    return skeleton

def skeletonize_2d(image):
    if False:
        i = 10
        return i + 15
    'Return the skeleton of a 2D binary image.\n\n    Thinning is used to reduce each connected component in a binary image\n    to a single-pixel wide skeleton.\n\n    Parameters\n    ----------\n    image : numpy.ndarray\n        A binary image containing the objects to be skeletonized. \'1\'\n        represents foreground, and \'0\' represents background. It\n        also accepts arrays of boolean values where True is foreground.\n\n    Returns\n    -------\n    skeleton : ndarray\n        A matrix containing the thinned image.\n\n    See Also\n    --------\n    medial_axis\n\n    Notes\n    -----\n    The algorithm [Zha84]_ works by making successive passes of the image,\n    removing pixels on object borders. This continues until no\n    more pixels can be removed.  The image is correlated with a\n    mask that assigns each pixel a number in the range [0...255]\n    corresponding to each possible pattern of its 8 neighboring\n    pixels. A look up table is then used to assign the pixels a\n    value of 0, 1, 2 or 3, which are selectively removed during\n    the iterations.\n\n    Note that this algorithm will give different results than a\n    medial axis transform, which is also often referred to as\n    "skeletonization".\n\n    References\n    ----------\n    .. [Zha84] A fast parallel algorithm for thinning digital patterns,\n           T. Y. Zhang and C. Y. Suen, Communications of the ACM,\n           March 1984, Volume 27, Number 3.\n\n    Examples\n    --------\n    >>> X, Y = np.ogrid[0:9, 0:9]\n    >>> ellipse = (1./3 * (X - 4)**2 + (Y - 4)**2 < 3**2).astype(np.uint8)\n    >>> ellipse\n    array([[0, 0, 0, 1, 1, 1, 0, 0, 0],\n           [0, 0, 1, 1, 1, 1, 1, 0, 0],\n           [0, 0, 1, 1, 1, 1, 1, 0, 0],\n           [0, 0, 1, 1, 1, 1, 1, 0, 0],\n           [0, 0, 1, 1, 1, 1, 1, 0, 0],\n           [0, 0, 1, 1, 1, 1, 1, 0, 0],\n           [0, 0, 1, 1, 1, 1, 1, 0, 0],\n           [0, 0, 1, 1, 1, 1, 1, 0, 0],\n           [0, 0, 0, 1, 1, 1, 0, 0, 0]], dtype=uint8)\n    >>> skel = skeletonize(ellipse)\n    >>> skel.astype(np.uint8)\n    array([[0, 0, 0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 1, 0, 0, 0, 0],\n           [0, 0, 0, 0, 1, 0, 0, 0, 0],\n           [0, 0, 0, 0, 1, 0, 0, 0, 0],\n           [0, 0, 0, 0, 1, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8)\n\n    '
    if image.ndim != 2:
        raise ValueError("Zhang's skeletonize method requires a 2D array")
    return _fast_skeletonize(image)

def _generate_thin_luts():
    if False:
        i = 10
        return i + 15
    'generate LUTs for thinning algorithm (for reference)'

    def nabe(n):
        if False:
            i = 10
            return i + 15
        return np.array([n >> i & 1 for i in range(0, 9)]).astype(bool)

    def G1(n):
        if False:
            return 10
        s = 0
        bits = nabe(n)
        for i in (0, 2, 4, 6):
            if not bits[i] and (bits[i + 1] or bits[(i + 2) % 8]):
                s += 1
        return s == 1
    g1_lut = np.array([G1(n) for n in range(256)])

    def G2(n):
        if False:
            while True:
                i = 10
        (n1, n2) = (0, 0)
        bits = nabe(n)
        for k in (1, 3, 5, 7):
            if bits[k] or bits[k - 1]:
                n1 += 1
            if bits[k] or bits[(k + 1) % 8]:
                n2 += 1
        return min(n1, n2) in [2, 3]
    g2_lut = np.array([G2(n) for n in range(256)])
    g12_lut = g1_lut & g2_lut

    def G3(n):
        if False:
            print('Hello World!')
        bits = nabe(n)
        return not ((bits[1] or bits[2] or (not bits[7])) and bits[0])

    def G3p(n):
        if False:
            i = 10
            return i + 15
        bits = nabe(n)
        return not ((bits[5] or bits[6] or (not bits[3])) and bits[4])
    g3_lut = np.array([G3(n) for n in range(256)])
    g3p_lut = np.array([G3p(n) for n in range(256)])
    g123_lut = g12_lut & g3_lut
    g123p_lut = g12_lut & g3p_lut
    return (g123_lut, g123p_lut)
G123_LUT = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0], dtype=bool)
G123P_LUT = np.array([0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=bool)

def thin(image, max_num_iter=None):
    if False:
        print('Hello World!')
    '\n    Perform morphological thinning of a binary image.\n\n    Parameters\n    ----------\n    image : binary (M, N) ndarray\n        The image to be thinned.\n    max_num_iter : int, number of iterations, optional\n        Regardless of the value of this parameter, the thinned image\n        is returned immediately if an iteration produces no change.\n        If this parameter is specified it thus sets an upper bound on\n        the number of iterations performed.\n\n    Returns\n    -------\n    out : ndarray of bool\n        Thinned image.\n\n    See Also\n    --------\n    skeletonize, medial_axis\n\n    Notes\n    -----\n    This algorithm [1]_ works by making multiple passes over the image,\n    removing pixels matching a set of criteria designed to thin\n    connected regions while preserving eight-connected components and\n    2 x 2 squares [2]_. In each of the two sub-iterations the algorithm\n    correlates the intermediate skeleton image with a neighborhood mask,\n    then looks up each neighborhood in a lookup table indicating whether\n    the central pixel should be deleted in that sub-iteration.\n\n    References\n    ----------\n    .. [1] Z. Guo and R. W. Hall, "Parallel thinning with\n           two-subiteration algorithms," Comm. ACM, vol. 32, no. 3,\n           pp. 359-373, 1989. :DOI:`10.1145/62065.62074`\n    .. [2] Lam, L., Seong-Whan Lee, and Ching Y. Suen, "Thinning\n           Methodologies-A Comprehensive Survey," IEEE Transactions on\n           Pattern Analysis and Machine Intelligence, Vol 14, No. 9,\n           p. 879, 1992. :DOI:`10.1109/34.161346`\n\n    Examples\n    --------\n    >>> square = np.zeros((7, 7), dtype=np.uint8)\n    >>> square[1:-1, 2:-2] = 1\n    >>> square[0, 1] =  1\n    >>> square\n    array([[0, 1, 0, 0, 0, 0, 0],\n           [0, 0, 1, 1, 1, 0, 0],\n           [0, 0, 1, 1, 1, 0, 0],\n           [0, 0, 1, 1, 1, 0, 0],\n           [0, 0, 1, 1, 1, 0, 0],\n           [0, 0, 1, 1, 1, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0]], dtype=uint8)\n    >>> skel = thin(square)\n    >>> skel.astype(np.uint8)\n    array([[0, 1, 0, 0, 0, 0, 0],\n           [0, 0, 1, 0, 0, 0, 0],\n           [0, 0, 0, 1, 0, 0, 0],\n           [0, 0, 0, 1, 0, 0, 0],\n           [0, 0, 0, 1, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0]], dtype=uint8)\n    '
    check_nD(image, 2)
    skel = np.asanyarray(image, dtype=bool).astype(np.uint8)
    mask = np.array([[8, 4, 2], [16, 0, 1], [32, 64, 128]], dtype=np.uint8)
    max_num_iter = max_num_iter or np.inf
    num_iter = 0
    (n_pts_old, n_pts_new) = (np.inf, np.sum(skel))
    while n_pts_old != n_pts_new and num_iter < max_num_iter:
        n_pts_old = n_pts_new
        for lut in [G123_LUT, G123P_LUT]:
            N = ndi.correlate(skel, mask, mode='constant')
            D = np.take(lut, N)
            skel[D] = 0
        n_pts_new = np.sum(skel)
        num_iter += 1
    return skel.astype(bool)
_eight_connect = ndi.generate_binary_structure(2, 2)

@deprecate_kwarg({'random_state': 'rng'}, deprecated_version='0.21', removed_version='0.23')
def medial_axis(image, mask=None, return_distance=False, *, rng=None):
    if False:
        print('Hello World!')
    'Compute the medial axis transform of a binary image.\n\n    Parameters\n    ----------\n    image : binary ndarray, shape (M, N)\n        The image of the shape to be skeletonized.\n    mask : binary ndarray, shape (M, N), optional\n        If a mask is given, only those elements in `image` with a true\n        value in `mask` are used for computing the medial axis.\n    return_distance : bool, optional\n        If true, the distance transform is returned as well as the skeleton.\n    rng : {`numpy.random.Generator`, int}, optional\n        Pseudo-random number generator.\n        By default, a PCG64 generator is used (see :func:`numpy.random.default_rng`).\n        If `rng` is an int, it is used to seed the generator.\n\n        The PRNG determines the order in which pixels are processed for\n        tiebreaking.\n\n        .. versionadded:: 0.19\n\n    Returns\n    -------\n    out : ndarray of bools\n        Medial axis transform of the image\n    dist : ndarray of ints, optional\n        Distance transform of the image (only returned if `return_distance`\n        is True)\n\n    See Also\n    --------\n    skeletonize\n\n    Notes\n    -----\n    This algorithm computes the medial axis transform of an image\n    as the ridges of its distance transform.\n\n    The different steps of the algorithm are as follows\n     * A lookup table is used, that assigns 0 or 1 to each configuration of\n       the 3x3 binary square, whether the central pixel should be removed\n       or kept. We want a point to be removed if it has more than one neighbor\n       and if removing it does not change the number of connected components.\n\n     * The distance transform to the background is computed, as well as\n       the cornerness of the pixel.\n\n     * The foreground (value of 1) points are ordered by\n       the distance transform, then the cornerness.\n\n     * A cython function is called to reduce the image to its skeleton. It\n       processes pixels in the order determined at the previous step, and\n       removes or maintains a pixel according to the lookup table. Because\n       of the ordering, it is possible to process all pixels in only one\n       pass.\n\n    Examples\n    --------\n    >>> square = np.zeros((7, 7), dtype=np.uint8)\n    >>> square[1:-1, 2:-2] = 1\n    >>> square\n    array([[0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 1, 1, 1, 0, 0],\n           [0, 0, 1, 1, 1, 0, 0],\n           [0, 0, 1, 1, 1, 0, 0],\n           [0, 0, 1, 1, 1, 0, 0],\n           [0, 0, 1, 1, 1, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0]], dtype=uint8)\n    >>> medial_axis(square).astype(np.uint8)\n    array([[0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 1, 0, 1, 0, 0],\n           [0, 0, 0, 1, 0, 0, 0],\n           [0, 0, 0, 1, 0, 0, 0],\n           [0, 0, 0, 1, 0, 0, 0],\n           [0, 0, 1, 0, 1, 0, 0],\n           [0, 0, 0, 0, 0, 0, 0]], dtype=uint8)\n\n    '
    global _eight_connect
    if mask is None:
        masked_image = image.astype(bool)
    else:
        masked_image = image.astype(bool).copy()
        masked_image[~mask] = False
    center_is_foreground = (np.arange(512) & 2 ** 4).astype(bool)
    table = center_is_foreground & (np.array([ndi.label(_pattern_of(index), _eight_connect)[1] != ndi.label(_pattern_of(index & ~2 ** 4), _eight_connect)[1] for index in range(512)]) | np.array([np.sum(_pattern_of(index)) < 3 for index in range(512)]))
    distance = ndi.distance_transform_edt(masked_image)
    if return_distance:
        store_distance = distance.copy()
    cornerness_table = np.array([9 - np.sum(_pattern_of(index)) for index in range(512)])
    corner_score = _table_lookup(masked_image, cornerness_table)
    (i, j) = np.mgrid[0:image.shape[0], 0:image.shape[1]]
    result = masked_image.copy()
    distance = distance[result]
    i = np.ascontiguousarray(i[result], dtype=np.intp)
    j = np.ascontiguousarray(j[result], dtype=np.intp)
    result = np.ascontiguousarray(result, np.uint8)
    generator = np.random.default_rng(rng)
    tiebreaker = generator.permutation(np.arange(masked_image.sum()))
    order = np.lexsort((tiebreaker, corner_score[masked_image], distance))
    order = np.ascontiguousarray(order, dtype=np.int32)
    table = np.ascontiguousarray(table, dtype=np.uint8)
    _skeletonize_loop(result, i, j, order, table)
    result = result.astype(bool)
    if mask is not None:
        result[~mask] = image[~mask]
    if return_distance:
        return (result, store_distance)
    else:
        return result

def _pattern_of(index):
    if False:
        i = 10
        return i + 15
    '\n    Return the pattern represented by an index value\n    Byte decomposition of index\n    '
    return np.array([[index & 2 ** 0, index & 2 ** 1, index & 2 ** 2], [index & 2 ** 3, index & 2 ** 4, index & 2 ** 5], [index & 2 ** 6, index & 2 ** 7, index & 2 ** 8]], bool)

def _table_lookup(image, table):
    if False:
        print('Hello World!')
    '\n    Perform a morphological transform on an image, directed by its\n    neighbors\n\n    Parameters\n    ----------\n    image : ndarray\n        A binary image\n    table : ndarray\n        A 512-element table giving the transform of each pixel given\n        the values of that pixel and its 8-connected neighbors.\n\n    Returns\n    -------\n    result : ndarray of same shape as `image`\n        Transformed image\n\n    Notes\n    -----\n    The pixels are numbered like this::\n\n      0 1 2\n      3 4 5\n      6 7 8\n\n    The index at a pixel is the sum of 2**<pixel-number> for pixels\n    that evaluate to true.\n    '
    if image.shape[0] < 3 or image.shape[1] < 3:
        image = image.astype(bool)
        indexer = np.zeros(image.shape, int)
        indexer[1:, 1:] += image[:-1, :-1] * 2 ** 0
        indexer[1:, :] += image[:-1, :] * 2 ** 1
        indexer[1:, :-1] += image[:-1, 1:] * 2 ** 2
        indexer[:, 1:] += image[:, :-1] * 2 ** 3
        indexer[:, :] += image[:, :] * 2 ** 4
        indexer[:, :-1] += image[:, 1:] * 2 ** 5
        indexer[:-1, 1:] += image[1:, :-1] * 2 ** 6
        indexer[:-1, :] += image[1:, :] * 2 ** 7
        indexer[:-1, :-1] += image[1:, 1:] * 2 ** 8
    else:
        indexer = _table_lookup_index(np.ascontiguousarray(image, np.uint8))
    image = table[indexer]
    return image

def skeletonize_3d(image):
    if False:
        print('Hello World!')
    'Compute the skeleton of a binary image.\n\n    Thinning is used to reduce each connected component in a binary image\n    to a single-pixel wide skeleton.\n\n    Parameters\n    ----------\n    image : ndarray, 2D or 3D\n        A binary image containing the objects to be skeletonized. Zeros\n        represent background, nonzero values are foreground.\n\n    Returns\n    -------\n    skeleton : ndarray\n        The thinned image.\n\n    See Also\n    --------\n    skeletonize, medial_axis\n\n    Notes\n    -----\n    The method of [Lee94]_ uses an octree data structure to examine a 3x3x3\n    neighborhood of a pixel. The algorithm proceeds by iteratively sweeping\n    over the image, and removing pixels at each iteration until the image\n    stops changing. Each iteration consists of two steps: first, a list of\n    candidates for removal is assembled; then pixels from this list are\n    rechecked sequentially, to better preserve connectivity of the image.\n\n    The algorithm this function implements is different from the algorithms\n    used by either `skeletonize` or `medial_axis`, thus for 2D images the\n    results produced by this function are generally different.\n\n    References\n    ----------\n    .. [Lee94] T.-C. Lee, R.L. Kashyap and C.-N. Chu, Building skeleton models\n           via 3-D medial surface/axis thinning algorithms.\n           Computer Vision, Graphics, and Image Processing, 56(6):462-478, 1994.\n\n    '
    if image.ndim < 2 or image.ndim > 3:
        raise ValueError(f'skeletonize_3d can only handle 2D or 3D images; got image.ndim = {image.ndim} instead.')
    image = np.ascontiguousarray(image)
    image = img_as_ubyte(image, force_copy=False)
    image_o = image
    if image.ndim == 2:
        image_o = image[np.newaxis, ...]
    image_o = np.pad(image_o, pad_width=1, mode='constant')
    maxval = image_o.max()
    image_o[image_o != 0] = 1
    image_o = np.asarray(_compute_thin_image(image_o))
    image_o = crop(image_o, crop_width=1)
    if image.ndim == 2:
        image_o = image_o[0]
    image_o *= maxval
    return image_o