import numpy as np

def _clip(x, low, high):
    if False:
        print('Hello World!')
    'Clip coordinate between low and high values.\n\n    This method was created so that `hessian_det_appx` does not have to make\n    a Python call.\n\n    Parameters\n    ----------\n    x : int\n        Coordinate to be clipped.\n    low : int\n        The lower bound.\n    high : int\n        The higher bound.\n\n    Returns\n    -------\n    x : int\n        `x` clipped between `high` and `low`.\n    '
    assert 0 <= low <= high
    if x > high:
        return high
    elif x < low:
        return low
    else:
        return x

def _integ(img, r, c, rl, cl):
    if False:
        print('Hello World!')
    'Integrate over the 2D integral image in the given window.\n\n    This method was created so that `hessian_det_appx` does not have to make\n    a Python call.\n\n    Parameters\n    ----------\n    img : array\n        The integral image over which to integrate.\n    r : int\n        The row number of the top left corner.\n    c : int\n        The column number of the top left corner.\n    rl : int\n        The number of rows over which to integrate.\n    cl : int\n        The number of columns over which to integrate.\n\n    Returns\n    -------\n    ans : int\n        The integral over the given window.\n    '
    r = _clip(r, 0, img.shape[0] - 1)
    c = _clip(c, 0, img.shape[1] - 1)
    r2 = _clip(r + rl, 0, img.shape[0] - 1)
    c2 = _clip(c + cl, 0, img.shape[1] - 1)
    ans = img[r, c] + img[r2, c2] - img[r, c2] - img[r2, c]
    return max(0.0, ans)

def _hessian_matrix_det(img, sigma):
    if False:
        return 10
    'Compute the approximate Hessian Determinant over a 2D image.\n\n    This method uses box filters over integral images to compute the\n    approximate Hessian Determinant as described in [1]_.\n\n    Parameters\n    ----------\n    img : array\n        The integral image over which to compute Hessian Determinant.\n    sigma : float\n        Standard deviation used for the Gaussian kernel, used for the Hessian\n        matrix\n\n    Returns\n    -------\n    out : array\n        The array of the Determinant of Hessians.\n\n    References\n    ----------\n    .. [1] Herbert Bay, Andreas Ess, Tinne Tuytelaars, Luc Van Gool,\n           "SURF: Speeded Up Robust Features"\n           ftp://ftp.vision.ee.ethz.ch/publications/articles/eth_biwi_00517.pdf\n\n    Notes\n    -----\n    The running time of this method only depends on size of the image. It is\n    independent of `sigma` as one would expect. The downside is that the\n    result for `sigma` less than `3` is not accurate, i.e., not similar to\n    the result obtained if someone computed the Hessian and took its\n    determinant.\n    '
    size = int(3 * sigma)
    (height, width) = img.shape
    s2 = (size - 1) // 2
    s3 = size // 3
    w = size
    out = np.empty_like(img, dtype=np.float64)
    w_i = 1.0 / size / size
    if size % 2 == 0:
        size += 1
    for r in range(height):
        for c in range(width):
            tl = _integ(img, r - s3, c - s3, s3, s3)
            br = _integ(img, r + 1, c + 1, s3, s3)
            bl = _integ(img, r - s3, c + 1, s3, s3)
            tr = _integ(img, r + 1, c - s3, s3, s3)
            dxy = bl + tr - tl - br
            dxy = -dxy * w_i
            mid = _integ(img, r - s3 + 1, c - s2, 2 * s3 - 1, w)
            side = _integ(img, r - s3 + 1, c - s3 // 2, 2 * s3 - 1, s3)
            dxx = mid - 3 * side
            dxx = -dxx * w_i
            mid = _integ(img, r - s2, c - s3 + 1, w, 2 * s3 - 1)
            side = _integ(img, r - s3 // 2, c - s3 + 1, s3, 2 * s3 - 1)
            dyy = mid - 3 * side
            dyy = -dyy * w_i
            out[r, c] = dxx * dyy - 0.81 * (dxy * dxy)
    return out