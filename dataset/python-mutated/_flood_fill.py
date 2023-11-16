"""flood_fill.py - in place flood fill algorithm

This module provides a function to fill all equal (or within tolerance) values
connected to a given seed point with a different value.
"""
import numpy as np
from ..util import crop
from ._flood_fill_cy import _flood_fill_equal, _flood_fill_tolerance
from ._util import _offsets_to_raveled_neighbors, _resolve_neighborhood, _set_border_values

def flood_fill(image, seed_point, new_value, *, footprint=None, connectivity=None, tolerance=None, in_place=False):
    if False:
        i = 10
        return i + 15
    "Perform flood filling on an image.\n\n    Starting at a specific `seed_point`, connected points equal or within\n    `tolerance` of the seed value are found, then set to `new_value`.\n\n    Parameters\n    ----------\n    image : ndarray\n        An n-dimensional array.\n    seed_point : tuple or int\n        The point in `image` used as the starting point for the flood fill.  If\n        the image is 1D, this point may be given as an integer.\n    new_value : `image` type\n        New value to set the entire fill.  This must be chosen in agreement\n        with the dtype of `image`.\n    footprint : ndarray, optional\n        The footprint (structuring element) used to determine the neighborhood\n        of each evaluated pixel. It must contain only 1's and 0's, have the\n        same number of dimensions as `image`. If not given, all adjacent pixels\n        are considered as part of the neighborhood (fully connected).\n    connectivity : int, optional\n        A number used to determine the neighborhood of each evaluated pixel.\n        Adjacent pixels whose squared distance from the center is less than or\n        equal to `connectivity` are considered neighbors. Ignored if\n        `footprint` is not None.\n    tolerance : float or int, optional\n        If None (default), adjacent values must be strictly equal to the\n        value of `image` at `seed_point` to be filled.  This is fastest.\n        If a tolerance is provided, adjacent points with values within plus or\n        minus tolerance from the seed point are filled (inclusive).\n    in_place : bool, optional\n        If True, flood filling is applied to `image` in place.  If False, the\n        flood filled result is returned without modifying the input `image`\n        (default).\n\n    Returns\n    -------\n    filled : ndarray\n        An array with the same shape as `image` is returned, with values in\n        areas connected to and equal (or within tolerance of) the seed point\n        replaced with `new_value`.\n\n    Notes\n    -----\n    The conceptual analogy of this operation is the 'paint bucket' tool in many\n    raster graphics programs.\n\n    Examples\n    --------\n    >>> from skimage.morphology import flood_fill\n    >>> image = np.zeros((4, 7), dtype=int)\n    >>> image[1:3, 1:3] = 1\n    >>> image[3, 0] = 1\n    >>> image[1:3, 4:6] = 2\n    >>> image[3, 6] = 3\n    >>> image\n    array([[0, 0, 0, 0, 0, 0, 0],\n           [0, 1, 1, 0, 2, 2, 0],\n           [0, 1, 1, 0, 2, 2, 0],\n           [1, 0, 0, 0, 0, 0, 3]])\n\n    Fill connected ones with 5, with full connectivity (diagonals included):\n\n    >>> flood_fill(image, (1, 1), 5)\n    array([[0, 0, 0, 0, 0, 0, 0],\n           [0, 5, 5, 0, 2, 2, 0],\n           [0, 5, 5, 0, 2, 2, 0],\n           [5, 0, 0, 0, 0, 0, 3]])\n\n    Fill connected ones with 5, excluding diagonal points (connectivity 1):\n\n    >>> flood_fill(image, (1, 1), 5, connectivity=1)\n    array([[0, 0, 0, 0, 0, 0, 0],\n           [0, 5, 5, 0, 2, 2, 0],\n           [0, 5, 5, 0, 2, 2, 0],\n           [1, 0, 0, 0, 0, 0, 3]])\n\n    Fill with a tolerance:\n\n    >>> flood_fill(image, (0, 0), 5, tolerance=1)\n    array([[5, 5, 5, 5, 5, 5, 5],\n           [5, 5, 5, 5, 2, 2, 5],\n           [5, 5, 5, 5, 2, 2, 5],\n           [5, 5, 5, 5, 5, 5, 3]])\n    "
    mask = flood(image, seed_point, footprint=footprint, connectivity=connectivity, tolerance=tolerance)
    if not in_place:
        image = image.copy()
    image[mask] = new_value
    return image

def flood(image, seed_point, *, footprint=None, connectivity=None, tolerance=None):
    if False:
        while True:
            i = 10
    "Mask corresponding to a flood fill.\n\n    Starting at a specific `seed_point`, connected points equal or within\n    `tolerance` of the seed value are found.\n\n    Parameters\n    ----------\n    image : ndarray\n        An n-dimensional array.\n    seed_point : tuple or int\n        The point in `image` used as the starting point for the flood fill.  If\n        the image is 1D, this point may be given as an integer.\n    footprint : ndarray, optional\n        The footprint (structuring element) used to determine the neighborhood\n        of each evaluated pixel. It must contain only 1's and 0's, have the\n        same number of dimensions as `image`. If not given, all adjacent pixels\n        are considered as part of the neighborhood (fully connected).\n    connectivity : int, optional\n        A number used to determine the neighborhood of each evaluated pixel.\n        Adjacent pixels whose squared distance from the center is less than or\n        equal to `connectivity` are considered neighbors. Ignored if\n        `footprint` is not None.\n    tolerance : float or int, optional\n        If None (default), adjacent values must be strictly equal to the\n        initial value of `image` at `seed_point`.  This is fastest.  If a value\n        is given, a comparison will be done at every point and if within\n        tolerance of the initial value will also be filled (inclusive).\n\n    Returns\n    -------\n    mask : ndarray\n        A Boolean array with the same shape as `image` is returned, with True\n        values for areas connected to and equal (or within tolerance of) the\n        seed point.  All other values are False.\n\n    Notes\n    -----\n    The conceptual analogy of this operation is the 'paint bucket' tool in many\n    raster graphics programs.  This function returns just the mask\n    representing the fill.\n\n    If indices are desired rather than masks for memory reasons, the user can\n    simply run `numpy.nonzero` on the result, save the indices, and discard\n    this mask.\n\n    Examples\n    --------\n    >>> from skimage.morphology import flood\n    >>> image = np.zeros((4, 7), dtype=int)\n    >>> image[1:3, 1:3] = 1\n    >>> image[3, 0] = 1\n    >>> image[1:3, 4:6] = 2\n    >>> image[3, 6] = 3\n    >>> image\n    array([[0, 0, 0, 0, 0, 0, 0],\n           [0, 1, 1, 0, 2, 2, 0],\n           [0, 1, 1, 0, 2, 2, 0],\n           [1, 0, 0, 0, 0, 0, 3]])\n\n    Fill connected ones with 5, with full connectivity (diagonals included):\n\n    >>> mask = flood(image, (1, 1))\n    >>> image_flooded = image.copy()\n    >>> image_flooded[mask] = 5\n    >>> image_flooded\n    array([[0, 0, 0, 0, 0, 0, 0],\n           [0, 5, 5, 0, 2, 2, 0],\n           [0, 5, 5, 0, 2, 2, 0],\n           [5, 0, 0, 0, 0, 0, 3]])\n\n    Fill connected ones with 5, excluding diagonal points (connectivity 1):\n\n    >>> mask = flood(image, (1, 1), connectivity=1)\n    >>> image_flooded = image.copy()\n    >>> image_flooded[mask] = 5\n    >>> image_flooded\n    array([[0, 0, 0, 0, 0, 0, 0],\n           [0, 5, 5, 0, 2, 2, 0],\n           [0, 5, 5, 0, 2, 2, 0],\n           [1, 0, 0, 0, 0, 0, 3]])\n\n    Fill with a tolerance:\n\n    >>> mask = flood(image, (0, 0), tolerance=1)\n    >>> image_flooded = image.copy()\n    >>> image_flooded[mask] = 5\n    >>> image_flooded\n    array([[5, 5, 5, 5, 5, 5, 5],\n           [5, 5, 5, 5, 2, 2, 5],\n           [5, 5, 5, 5, 2, 2, 5],\n           [5, 5, 5, 5, 5, 5, 3]])\n    "
    image = np.asarray(image)
    if image.flags.f_contiguous is True:
        order = 'F'
    elif image.flags.c_contiguous is True:
        order = 'C'
    else:
        image = np.ascontiguousarray(image)
        order = 'C'
    if 0 in image.shape:
        return np.zeros(image.shape, dtype=bool)
    try:
        iter(seed_point)
    except TypeError:
        seed_point = (seed_point,)
    seed_value = image[seed_point]
    seed_point = tuple(np.asarray(seed_point) % image.shape)
    footprint = _resolve_neighborhood(footprint, connectivity, image.ndim, enforce_adjacency=False)
    center = tuple((s // 2 for s in footprint.shape))
    pad_width = [(np.max(np.abs(idx - c)),) * 2 for (idx, c) in zip(np.nonzero(footprint), center)]
    working_image = np.pad(image, pad_width, mode='constant', constant_values=image.min())
    ravelled_seed_idx = np.ravel_multi_index([i + pad_start for (i, (pad_start, pad_end)) in zip(seed_point, pad_width)], working_image.shape, order=order)
    neighbor_offsets = _offsets_to_raveled_neighbors(working_image.shape, footprint, center=center, order=order)
    flags = np.zeros(working_image.shape, dtype=np.uint8, order=order)
    _set_border_values(flags, value=2, border_width=pad_width)
    try:
        if tolerance is not None:
            try:
                max_value = np.finfo(working_image.dtype).max
                min_value = np.finfo(working_image.dtype).min
            except ValueError:
                max_value = np.iinfo(working_image.dtype).max
                min_value = np.iinfo(working_image.dtype).min
            high_tol = min(max_value, seed_value + tolerance)
            low_tol = max(min_value, seed_value - tolerance)
            _flood_fill_tolerance(working_image.ravel(order), flags.ravel(order), neighbor_offsets, ravelled_seed_idx, seed_value, low_tol, high_tol)
        else:
            _flood_fill_equal(working_image.ravel(order), flags.ravel(order), neighbor_offsets, ravelled_seed_idx, seed_value)
    except TypeError:
        if working_image.dtype == np.float16:
            raise TypeError('dtype of `image` is float16 which is not supported, try upcasting to float32')
        else:
            raise
    return crop(flags, pad_width, copy=False).view(bool)