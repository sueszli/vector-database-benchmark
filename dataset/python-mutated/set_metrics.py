import warnings
import numpy as np
from scipy.spatial import cKDTree

def hausdorff_distance(image0, image1, method='standard'):
    if False:
        for i in range(10):
            print('nop')
    "Calculate the Hausdorff distance between nonzero elements of given images.\n\n    Parameters\n    ----------\n    image0, image1 : ndarray\n        Arrays where ``True`` represents a point that is included in a\n        set of points. Both arrays must have the same shape.\n    method : {'standard', 'modified'}, optional, default = 'standard'\n        The method to use for calculating the Hausdorff distance.\n        ``standard`` is the standard Hausdorff distance, while ``modified``\n        is the modified Hausdorff distance.\n\n    Returns\n    -------\n    distance : float\n        The Hausdorff distance between coordinates of nonzero pixels in\n        ``image0`` and ``image1``, using the Euclidean distance.\n\n    Notes\n    -----\n    The Hausdorff distance [1]_ is the maximum distance between any point on\n    ``image0`` and its nearest point on ``image1``, and vice-versa.\n    The Modified Hausdorff Distance (MHD) has been shown to perform better\n    than the directed Hausdorff Distance (HD) in the following work by\n    Dubuisson et al. [2]_. The function calculates forward and backward\n    mean distances and returns the largest of the two.\n\n    References\n    ----------\n    .. [1] http://en.wikipedia.org/wiki/Hausdorff_distance\n    .. [2] M. P. Dubuisson and A. K. Jain. A Modified Hausdorff distance for object\n       matching. In ICPR94, pages A:566-568, Jerusalem, Israel, 1994.\n       :DOI:`10.1109/ICPR.1994.576361`\n       http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.1.8155\n\n    Examples\n    --------\n    >>> points_a = (3, 0)\n    >>> points_b = (6, 0)\n    >>> shape = (7, 1)\n    >>> image_a = np.zeros(shape, dtype=bool)\n    >>> image_b = np.zeros(shape, dtype=bool)\n    >>> image_a[points_a] = True\n    >>> image_b[points_b] = True\n    >>> hausdorff_distance(image_a, image_b)\n    3.0\n\n    "
    if method not in ('standard', 'modified'):
        raise ValueError(f'unrecognized method {method}')
    a_points = np.transpose(np.nonzero(image0))
    b_points = np.transpose(np.nonzero(image1))
    if len(a_points) == 0:
        return 0 if len(b_points) == 0 else np.inf
    elif len(b_points) == 0:
        return np.inf
    (fwd, bwd) = (cKDTree(a_points).query(b_points, k=1)[0], cKDTree(b_points).query(a_points, k=1)[0])
    if method == 'standard':
        return max(max(fwd), max(bwd))
    elif method == 'modified':
        return max(np.mean(fwd), np.mean(bwd))

def hausdorff_pair(image0, image1):
    if False:
        while True:
            i = 10
    'Returns pair of points that are Hausdorff distance apart between nonzero\n    elements of given images.\n\n    The Hausdorff distance [1]_ is the maximum distance between any point on\n    ``image0`` and its nearest point on ``image1``, and vice-versa.\n\n    Parameters\n    ----------\n    image0, image1 : ndarray\n        Arrays where ``True`` represents a point that is included in a\n        set of points. Both arrays must have the same shape.\n\n    Returns\n    -------\n    point_a, point_b : array\n        A pair of points that have Hausdorff distance between them.\n\n    References\n    ----------\n    .. [1] http://en.wikipedia.org/wiki/Hausdorff_distance\n\n    Examples\n    --------\n    >>> points_a = (3, 0)\n    >>> points_b = (6, 0)\n    >>> shape = (7, 1)\n    >>> image_a = np.zeros(shape, dtype=bool)\n    >>> image_b = np.zeros(shape, dtype=bool)\n    >>> image_a[points_a] = True\n    >>> image_b[points_b] = True\n    >>> hausdorff_pair(image_a, image_b)\n    (array([3, 0]), array([6, 0]))\n\n    '
    a_points = np.transpose(np.nonzero(image0))
    b_points = np.transpose(np.nonzero(image1))
    if len(a_points) == 0 or len(b_points) == 0:
        warnings.warn('One or both of the images is empty.', stacklevel=2)
        return ((), ())
    (nearest_dists_from_b, nearest_a_point_indices_from_b) = cKDTree(a_points).query(b_points)
    (nearest_dists_from_a, nearest_b_point_indices_from_a) = cKDTree(b_points).query(a_points)
    max_index_from_a = nearest_dists_from_b.argmax()
    max_index_from_b = nearest_dists_from_a.argmax()
    max_dist_from_a = nearest_dists_from_b[max_index_from_a]
    max_dist_from_b = nearest_dists_from_a[max_index_from_b]
    if max_dist_from_b > max_dist_from_a:
        return (a_points[max_index_from_b], b_points[nearest_b_point_indices_from_a[max_index_from_b]])
    else:
        return (a_points[nearest_a_point_indices_from_b[max_index_from_a]], b_points[max_index_from_a])