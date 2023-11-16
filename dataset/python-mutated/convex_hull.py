"""Convex Hull."""
from itertools import product
import numpy as np
from scipy.spatial import ConvexHull, QhullError
from ..measure.pnpoly import grid_points_in_poly
from ._convex_hull import possible_hull
from ..measure._label import label
from ..util import unique_rows
from .._shared.utils import warn
__all__ = ['convex_hull_image', 'convex_hull_object']

def _offsets_diamond(ndim):
    if False:
        while True:
            i = 10
    offsets = np.zeros((2 * ndim, ndim))
    for (vertex, (axis, offset)) in enumerate(product(range(ndim), (-0.5, 0.5))):
        offsets[vertex, axis] = offset
    return offsets

def _check_coords_in_hull(gridcoords, hull_equations, tolerance):
    if False:
        while True:
            i = 10
    'Checks all the coordinates for inclusiveness in the convex hull.\n\n    Parameters\n    ----------\n    gridcoords : (M, N) ndarray\n        Coordinates of ``N`` points in ``M`` dimensions.\n    hull_equations : (M, N) ndarray\n        Hyperplane equations of the facets of the convex hull.\n    tolerance : float\n        Tolerance when determining whether a point is inside the hull. Due\n        to numerical floating point errors, a tolerance of 0 can result in\n        some points erroneously being classified as being outside the hull.\n\n    Returns\n    -------\n    coords_in_hull : ndarray of bool\n        Binary 1D ndarray representing points in n-dimensional space\n        with value ``True`` set for points inside the convex hull.\n\n    Notes\n    -----\n    Checking the inclusiveness of coordinates in a convex hull requires\n    intermediate calculations of dot products which are memory-intensive.\n    Thus, the convex hull equations are checked individually with all\n    coordinates to keep within the memory limit.\n\n    References\n    ----------\n    .. [1] https://github.com/scikit-image/scikit-image/issues/5019\n\n    '
    (ndim, n_coords) = gridcoords.shape
    n_hull_equations = hull_equations.shape[0]
    coords_in_hull = np.ones(n_coords, dtype=bool)
    dot_array = np.empty(n_coords, dtype=np.float64)
    test_ineq_temp = np.empty(n_coords, dtype=np.float64)
    coords_single_ineq = np.empty(n_coords, dtype=bool)
    for idx in range(n_hull_equations):
        np.dot(hull_equations[idx, :ndim], gridcoords, out=dot_array)
        np.add(dot_array, hull_equations[idx, ndim:], out=test_ineq_temp)
        np.less(test_ineq_temp, tolerance, out=coords_single_ineq)
        coords_in_hull *= coords_single_ineq
    return coords_in_hull

def convex_hull_image(image, offset_coordinates=True, tolerance=1e-10, include_borders=True):
    if False:
        print('Hello World!')
    'Compute the convex hull image of a binary image.\n\n    The convex hull is the set of pixels included in the smallest convex\n    polygon that surround all white pixels in the input image.\n\n    Parameters\n    ----------\n    image : array\n        Binary input image. This array is cast to bool before processing.\n    offset_coordinates : bool, optional\n        If ``True``, a pixel at coordinate, e.g., (4, 7) will be represented\n        by coordinates (3.5, 7), (4.5, 7), (4, 6.5), and (4, 7.5). This adds\n        some "extent" to a pixel when computing the hull.\n    tolerance : float, optional\n        Tolerance when determining whether a point is inside the hull. Due\n        to numerical floating point errors, a tolerance of 0 can result in\n        some points erroneously being classified as being outside the hull.\n    include_borders: bool, optional\n        If ``False``, vertices/edges are excluded from the final hull mask.\n\n    Returns\n    -------\n    hull : (M, N) array of bool\n        Binary image with pixels in convex hull set to True.\n\n    References\n    ----------\n    .. [1] https://blogs.mathworks.com/steve/2011/10/04/binary-image-convex-hull-algorithm-notes/\n\n    '
    ndim = image.ndim
    if np.count_nonzero(image) == 0:
        warn('Input image is entirely zero, no valid convex hull. Returning empty image', UserWarning)
        return np.zeros(image.shape, dtype=bool)
    if ndim == 2:
        coords = possible_hull(np.ascontiguousarray(image, dtype=np.uint8))
    else:
        coords = np.transpose(np.nonzero(image))
        if offset_coordinates:
            try:
                hull0 = ConvexHull(coords)
            except QhullError as err:
                warn(f'Failed to get convex hull image. Returning empty image, see error message below:\n{err}')
                return np.zeros(image.shape, dtype=bool)
            coords = hull0.points[hull0.vertices]
    if offset_coordinates:
        offsets = _offsets_diamond(image.ndim)
        coords = (coords[:, np.newaxis, :] + offsets).reshape(-1, ndim)
    coords = unique_rows(coords)
    try:
        hull = ConvexHull(coords)
    except QhullError as err:
        warn(f'Failed to get convex hull image. Returning empty image, see error message below:\n{err}')
        return np.zeros(image.shape, dtype=bool)
    vertices = hull.points[hull.vertices]
    if ndim == 2:
        labels = grid_points_in_poly(image.shape, vertices, binarize=False)
        mask = labels >= 1 if include_borders else labels == 1
    else:
        gridcoords = np.reshape(np.mgrid[tuple(map(slice, image.shape))], (ndim, -1))
        coords_in_hull = _check_coords_in_hull(gridcoords, hull.equations, tolerance)
        mask = np.reshape(coords_in_hull, image.shape)
    return mask

def convex_hull_object(image, *, connectivity=2):
    if False:
        print('Hello World!')
    'Compute the convex hull image of individual objects in a binary image.\n\n    The convex hull is the set of pixels included in the smallest convex\n    polygon that surround all white pixels in the input image.\n\n    Parameters\n    ----------\n    image : (M, N) ndarray\n        Binary input image.\n    connectivity : {1, 2}, int, optional\n        Determines the neighbors of each pixel. Adjacent elements\n        within a squared distance of ``connectivity`` from pixel center\n        are considered neighbors.::\n\n            1-connectivity      2-connectivity\n                  [ ]           [ ]  [ ]  [ ]\n                   |               \\  |  /\n             [ ]--[x]--[ ]      [ ]--[x]--[ ]\n                   |               /  |  \\\n                  [ ]           [ ]  [ ]  [ ]\n\n    Returns\n    -------\n    hull : ndarray of bool\n        Binary image with pixels inside convex hull set to ``True``.\n\n    Notes\n    -----\n    This function uses ``skimage.morphology.label`` to define unique objects,\n    finds the convex hull of each using ``convex_hull_image``, and combines\n    these regions with logical OR. Be aware the convex hulls of unconnected\n    objects may overlap in the result. If this is suspected, consider using\n    convex_hull_image separately on each object or adjust ``connectivity``.\n    '
    if image.ndim > 2:
        raise ValueError('Input must be a 2D image')
    if connectivity not in (1, 2):
        raise ValueError('`connectivity` must be either 1 or 2.')
    labeled_im = label(image, connectivity=connectivity, background=0)
    convex_obj = np.zeros(image.shape, dtype=bool)
    convex_img = np.zeros(image.shape, dtype=bool)
    for i in range(1, labeled_im.max() + 1):
        convex_obj = convex_hull_image(labeled_im == i)
        convex_img = np.logical_or(convex_img, convex_obj)
    return convex_img