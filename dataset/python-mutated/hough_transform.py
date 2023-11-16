import numpy as np
from scipy.spatial import cKDTree
from ._hough_transform import _hough_circle, _hough_ellipse, _hough_line
from ._hough_transform import _probabilistic_hough_line as _prob_hough_line
from .._shared.utils import deprecate_kwarg

def hough_line_peaks(hspace, angles, dists, min_distance=9, min_angle=10, threshold=None, num_peaks=np.inf):
    if False:
        i = 10
        return i + 15
    'Return peaks in a straight line Hough transform.\n\n    Identifies most prominent lines separated by a certain angle and distance\n    in a Hough transform. Non-maximum suppression with different sizes is\n    applied separately in the first (distances) and second (angles) dimension\n    of the Hough space to identify peaks.\n\n    Parameters\n    ----------\n    hspace : ndarray, shape (M, N)\n        Hough space returned by the `hough_line` function.\n    angles : array, shape (N,)\n        Angles returned by the `hough_line` function. Assumed to be continuous.\n        (`angles[-1] - angles[0] == PI`).\n    dists : array, shape (M,)\n        Distances returned by the `hough_line` function.\n    min_distance : int, optional\n        Minimum distance separating lines (maximum filter size for first\n        dimension of hough space).\n    min_angle : int, optional\n        Minimum angle separating lines (maximum filter size for second\n        dimension of hough space).\n    threshold : float, optional\n        Minimum intensity of peaks. Default is `0.5 * max(hspace)`.\n    num_peaks : int, optional\n        Maximum number of peaks. When the number of peaks exceeds `num_peaks`,\n        return `num_peaks` coordinates based on peak intensity.\n\n    Returns\n    -------\n    accum, angles, dists : tuple of array\n        Peak values in Hough space, angles and distances.\n\n    Examples\n    --------\n    >>> from skimage.transform import hough_line, hough_line_peaks\n    >>> from skimage.draw import line\n    >>> img = np.zeros((15, 15), dtype=bool)\n    >>> rr, cc = line(0, 0, 14, 14)\n    >>> img[rr, cc] = 1\n    >>> rr, cc = line(0, 14, 14, 0)\n    >>> img[cc, rr] = 1\n    >>> hspace, angles, dists = hough_line(img)\n    >>> hspace, angles, dists = hough_line_peaks(hspace, angles, dists)\n    >>> len(angles)\n    2\n\n    '
    from ..feature.peak import _prominent_peaks
    min_angle = min(min_angle, hspace.shape[1])
    (h, a, d) = _prominent_peaks(hspace, min_xdistance=min_angle, min_ydistance=min_distance, threshold=threshold, num_peaks=num_peaks)
    if a.size > 0:
        return (h, angles[a], dists[d])
    else:
        return (h, np.array([]), np.array([]))

def hough_circle(image, radius, normalize=True, full_output=False):
    if False:
        print('Hello World!')
    'Perform a circular Hough transform.\n\n    Parameters\n    ----------\n    image : ndarray, shape (M, N)\n        Input image with nonzero values representing edges.\n    radius : scalar or sequence of scalars\n        Radii at which to compute the Hough transform.\n        Floats are converted to integers.\n    normalize : boolean, optional\n        Normalize the accumulator with the number\n        of pixels used to draw the radius.\n    full_output : boolean, optional\n        Extend the output size by twice the largest\n        radius in order to detect centers outside the\n        input picture.\n\n    Returns\n    -------\n    H : ndarray, shape (radius index, M + 2R, N + 2R)\n        Hough transform accumulator for each radius.\n        R designates the larger radius if full_output is True.\n        Otherwise, R = 0.\n\n    Examples\n    --------\n    >>> from skimage.transform import hough_circle\n    >>> from skimage.draw import circle_perimeter\n    >>> img = np.zeros((100, 100), dtype=bool)\n    >>> rr, cc = circle_perimeter(25, 35, 23)\n    >>> img[rr, cc] = 1\n    >>> try_radii = np.arange(5, 50)\n    >>> res = hough_circle(img, try_radii)\n    >>> ridx, r, c = np.unravel_index(np.argmax(res), res.shape)\n    >>> r, c, try_radii[ridx]\n    (25, 35, 23)\n\n    '
    radius = np.atleast_1d(np.asarray(radius))
    return _hough_circle(image, radius.astype(np.intp), normalize=normalize, full_output=full_output)

def hough_ellipse(image, threshold=4, accuracy=1, min_size=4, max_size=None):
    if False:
        for i in range(10):
            print('nop')
    'Perform an elliptical Hough transform.\n\n    Parameters\n    ----------\n    image : (M, N) ndarray\n        Input image with nonzero values representing edges.\n    threshold : int, optional\n        Accumulator threshold value.\n    accuracy : double, optional\n        Bin size on the minor axis used in the accumulator.\n    min_size : int, optional\n        Minimal major axis length.\n    max_size : int, optional\n        Maximal minor axis length.\n        If None, the value is set to the half of the smaller\n        image dimension.\n\n    Returns\n    -------\n    result : ndarray with fields [(accumulator, yc, xc, a, b, orientation)].\n          Where ``(yc, xc)`` is the center, ``(a, b)`` the major and minor\n          axes, respectively. The `orientation` value follows\n          `skimage.draw.ellipse_perimeter` convention.\n\n    Examples\n    --------\n    >>> from skimage.transform import hough_ellipse\n    >>> from skimage.draw import ellipse_perimeter\n    >>> img = np.zeros((25, 25), dtype=np.uint8)\n    >>> rr, cc = ellipse_perimeter(10, 10, 6, 8)\n    >>> img[cc, rr] = 1\n    >>> result = hough_ellipse(img, threshold=8)\n    >>> result.tolist()\n    [(10, 10.0, 10.0, 8.0, 6.0, 0.0)]\n\n    Notes\n    -----\n    The accuracy must be chosen to produce a peak in the accumulator\n    distribution. In other words, a flat accumulator distribution with low\n    values may be caused by a too low bin size.\n\n    References\n    ----------\n    .. [1] Xie, Yonghong, and Qiang Ji. "A new efficient ellipse detection\n           method." Pattern Recognition, 2002. Proceedings. 16th International\n           Conference on. Vol. 2. IEEE, 2002\n    '
    return _hough_ellipse(image, threshold=threshold, accuracy=accuracy, min_size=min_size, max_size=max_size)

def hough_line(image, theta=None):
    if False:
        while True:
            i = 10
    'Perform a straight line Hough transform.\n\n    Parameters\n    ----------\n    image : (M, N) ndarray\n        Input image with nonzero values representing edges.\n    theta : ndarray of double, shape (K,), optional\n        Angles at which to compute the transform, in radians.\n        Defaults to a vector of 180 angles evenly spaced in the\n        range [-pi/2, pi/2).\n\n    Returns\n    -------\n    hspace : ndarray of uint64, shape (P, Q)\n        Hough transform accumulator.\n    angles : ndarray\n        Angles at which the transform is computed, in radians.\n    distances : ndarray\n        Distance values.\n\n    Notes\n    -----\n    The origin is the top left corner of the original image.\n    X and Y axis are horizontal and vertical edges respectively.\n    The distance is the minimal algebraic distance from the origin\n    to the detected line.\n    The angle accuracy can be improved by decreasing the step size in\n    the `theta` array.\n\n    Examples\n    --------\n    Generate a test image:\n\n    >>> img = np.zeros((100, 150), dtype=bool)\n    >>> img[30, :] = 1\n    >>> img[:, 65] = 1\n    >>> img[35:45, 35:50] = 1\n    >>> for i in range(90):\n    ...     img[i, i] = 1\n    >>> rng = np.random.default_rng()\n    >>> img += rng.random(img.shape) > 0.95\n\n    Apply the Hough transform:\n\n    >>> out, angles, d = hough_line(img)\n    '
    if image.ndim != 2:
        raise ValueError('The input image `image` must be 2D.')
    if theta is None:
        theta = np.linspace(-np.pi / 2, np.pi / 2, 180, endpoint=False)
    return _hough_line(image, theta=theta)

@deprecate_kwarg({'seed': 'rng'}, deprecated_version='0.21', removed_version='0.23')
def probabilistic_hough_line(image, threshold=10, line_length=50, line_gap=10, theta=None, rng=None):
    if False:
        return 10
    'Return lines from a progressive probabilistic line Hough transform.\n\n    Parameters\n    ----------\n    image : ndarray, shape (M, N)\n        Input image with nonzero values representing edges.\n    threshold : int, optional\n        Threshold\n    line_length : int, optional\n        Minimum accepted length of detected lines.\n        Increase the parameter to extract longer lines.\n    line_gap : int, optional\n        Maximum gap between pixels to still form a line.\n        Increase the parameter to merge broken lines more aggressively.\n    theta : ndarray of dtype, shape (K,), optional\n        Angles at which to compute the transform, in radians.\n        Defaults to a vector of 180 angles evenly spaced in the\n        range [-pi/2, pi/2).\n    rng : {`numpy.random.Generator`, int}, optional\n        Pseudo-random number generator.\n        By default, a PCG64 generator is used (see :func:`numpy.random.default_rng`).\n        If `rng` is an int, it is used to seed the generator.\n\n    Returns\n    -------\n    lines : list\n      List of lines identified, lines in format ((x0, y0), (x1, y1)),\n      indicating line start and end.\n\n    References\n    ----------\n    .. [1] C. Galamhos, J. Matas and J. Kittler, "Progressive probabilistic\n           Hough transform for line detection", in IEEE Computer Society\n           Conference on Computer Vision and Pattern Recognition, 1999.\n    '
    if image.ndim != 2:
        raise ValueError('The input image `image` must be 2D.')
    if theta is None:
        theta = np.linspace(-np.pi / 2, np.pi / 2, 180, endpoint=False)
    return _prob_hough_line(image, threshold=threshold, line_length=line_length, line_gap=line_gap, theta=theta, rng=rng)

def hough_circle_peaks(hspaces, radii, min_xdistance=1, min_ydistance=1, threshold=None, num_peaks=np.inf, total_num_peaks=np.inf, normalize=False):
    if False:
        while True:
            i = 10
    'Return peaks in a circle Hough transform.\n\n    Identifies most prominent circles separated by certain distances in given\n    Hough spaces. Non-maximum suppression with different sizes is applied\n    separately in the first and second dimension of the Hough space to\n    identify peaks. For circles with different radius but close in distance,\n    only the one with highest peak is kept.\n\n    Parameters\n    ----------\n    hspaces : (M, N, P) array\n        Hough spaces returned by the `hough_circle` function.\n    radii : (M,) array\n        Radii corresponding to Hough spaces.\n    min_xdistance : int, optional\n        Minimum distance separating centers in the x dimension.\n    min_ydistance : int, optional\n        Minimum distance separating centers in the y dimension.\n    threshold : float, optional\n        Minimum intensity of peaks in each Hough space.\n        Default is `0.5 * max(hspace)`.\n    num_peaks : int, optional\n        Maximum number of peaks in each Hough space. When the\n        number of peaks exceeds `num_peaks`, only `num_peaks`\n        coordinates based on peak intensity are considered for the\n        corresponding radius.\n    total_num_peaks : int, optional\n        Maximum number of peaks. When the number of peaks exceeds `num_peaks`,\n        return `num_peaks` coordinates based on peak intensity.\n    normalize : bool, optional\n        If True, normalize the accumulator by the radius to sort the prominent\n        peaks.\n\n    Returns\n    -------\n    accum, cx, cy, rad : tuple of array\n        Peak values in Hough space, x and y center coordinates and radii.\n\n    Examples\n    --------\n    >>> from skimage import transform, draw\n    >>> img = np.zeros((120, 100), dtype=int)\n    >>> radius, x_0, y_0 = (20, 99, 50)\n    >>> y, x = draw.circle_perimeter(y_0, x_0, radius)\n    >>> img[x, y] = 1\n    >>> hspaces = transform.hough_circle(img, radius)\n    >>> accum, cx, cy, rad = hough_circle_peaks(hspaces, [radius,])\n\n    Notes\n    -----\n    Circles with bigger radius have higher peaks in Hough space. If larger\n    circles are preferred over smaller ones, `normalize` should be False.\n    Otherwise, circles will be returned in the order of decreasing voting\n    number.\n    '
    from ..feature.peak import _prominent_peaks
    r = []
    cx = []
    cy = []
    accum = []
    for (rad, hp) in zip(radii, hspaces):
        (h_p, x_p, y_p) = _prominent_peaks(hp, min_xdistance=min_xdistance, min_ydistance=min_ydistance, threshold=threshold, num_peaks=num_peaks)
        r.extend((rad,) * len(h_p))
        cx.extend(x_p)
        cy.extend(y_p)
        accum.extend(h_p)
    r = np.array(r)
    cx = np.array(cx)
    cy = np.array(cy)
    accum = np.array(accum)
    if normalize:
        s = np.argsort(accum / r)
    else:
        s = np.argsort(accum)
    (accum_sorted, cx_sorted, cy_sorted, r_sorted) = (accum[s][::-1], cx[s][::-1], cy[s][::-1], r[s][::-1])
    tnp = len(accum_sorted) if total_num_peaks == np.inf else total_num_peaks
    if min_xdistance == 1 and min_ydistance == 1 or len(accum_sorted) == 0:
        return (accum_sorted[:tnp], cx_sorted[:tnp], cy_sorted[:tnp], r_sorted[:tnp])
    should_keep = label_distant_points(cx_sorted, cy_sorted, min_xdistance, min_ydistance, tnp)
    return (accum_sorted[should_keep], cx_sorted[should_keep], cy_sorted[should_keep], r_sorted[should_keep])

def label_distant_points(xs, ys, min_xdistance, min_ydistance, max_points):
    if False:
        for i in range(10):
            print('nop')
    'Keep points that are separated by certain distance in each dimension.\n\n    The first point is always accepted and all subsequent points are selected\n    so that they are distant from all their preceding ones.\n\n    Parameters\n    ----------\n    xs : array, shape (M,)\n        X coordinates of points.\n    ys : array, shape (M,)\n        Y coordinates of points.\n    min_xdistance : int\n        Minimum distance separating points in the x dimension.\n    min_ydistance : int\n        Minimum distance separating points in the y dimension.\n    max_points : int\n        Max number of distant points to keep.\n\n    Returns\n    -------\n    should_keep : array of bool\n        A mask array for distant points to keep.\n    '
    is_neighbor = np.zeros(len(xs), dtype=bool)
    coordinates = np.stack([xs, ys], axis=1)
    kd_tree = cKDTree(coordinates)
    n_pts = 0
    for i in range(len(xs)):
        if n_pts >= max_points:
            is_neighbor[i] = True
        elif not is_neighbor[i]:
            neighbors_i = kd_tree.query_ball_point((xs[i], ys[i]), np.hypot(min_xdistance, min_ydistance))
            for ni in neighbors_i:
                x_close = abs(xs[ni] - xs[i]) <= min_xdistance
                y_close = abs(ys[ni] - ys[i]) <= min_ydistance
                if x_close and y_close and (ni > i):
                    is_neighbor[ni] = True
            n_pts += 1
    should_keep = ~is_neighbor
    return should_keep