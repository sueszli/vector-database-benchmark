import numpy as np

def _round_safe(coords):
    if False:
        return 10
    'Round coords while ensuring successive values are less than 1 apart.\n\n    When rounding coordinates for `line_nd`, we want coordinates that are less\n    than 1 apart (always the case, by design) to remain less than one apart.\n    However, NumPy rounds values to the nearest *even* integer, so:\n\n    >>> np.round([0.5, 1.5, 2.5, 3.5, 4.5])\n    array([0., 2., 2., 4., 4.])\n\n    So, for our application, we detect whether the above case occurs, and use\n    ``np.floor`` if so. It is sufficient to detect that the first coordinate\n    falls on 0.5 and that the second coordinate is 1.0 apart, since we assume\n    by construction that the inter-point distance is less than or equal to 1\n    and that all successive points are equidistant.\n\n    Parameters\n    ----------\n    coords : 1D array of float\n        The coordinates array. We assume that all successive values are\n        equidistant (``np.all(np.diff(coords) = coords[1] - coords[0])``)\n        and that this distance is no more than 1\n        (``np.abs(coords[1] - coords[0]) <= 1``).\n\n    Returns\n    -------\n    rounded : 1D array of int\n        The array correctly rounded for an indexing operation, such that no\n        successive indices will be more than 1 apart.\n\n    Examples\n    --------\n    >>> coords0 = np.array([0.5, 1.25, 2., 2.75, 3.5])\n    >>> _round_safe(coords0)\n    array([0, 1, 2, 3, 4])\n    >>> coords1 = np.arange(0.5, 8, 1)\n    >>> coords1\n    array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5])\n    >>> _round_safe(coords1)\n    array([0, 1, 2, 3, 4, 5, 6, 7])\n    '
    if len(coords) > 1 and coords[0] % 1 == 0.5 and (coords[1] - coords[0] == 1):
        _round_function = np.floor
    else:
        _round_function = np.round
    return _round_function(coords).astype(int)

def line_nd(start, stop, *, endpoint=False, integer=True):
    if False:
        i = 10
        return i + 15
    'Draw a single-pixel thick line in n dimensions.\n\n    The line produced will be ndim-connected. That is, two subsequent\n    pixels in the line will be either direct or diagonal neighbors in\n    n dimensions.\n\n    Parameters\n    ----------\n    start : array-like, shape (N,)\n        The start coordinates of the line.\n    stop : array-like, shape (N,)\n        The end coordinates of the line.\n    endpoint : bool, optional\n        Whether to include the endpoint in the returned line. Defaults\n        to False, which allows for easy drawing of multi-point paths.\n    integer : bool, optional\n        Whether to round the coordinates to integer. If True (default),\n        the returned coordinates can be used to directly index into an\n        array. `False` could be used for e.g. vector drawing.\n\n    Returns\n    -------\n    coords : tuple of arrays\n        The coordinates of points on the line.\n\n    Examples\n    --------\n    >>> lin = line_nd((1, 1), (5, 2.5), endpoint=False)\n    >>> lin\n    (array([1, 2, 3, 4]), array([1, 1, 2, 2]))\n    >>> im = np.zeros((6, 5), dtype=int)\n    >>> im[lin] = 1\n    >>> im\n    array([[0, 0, 0, 0, 0],\n           [0, 1, 0, 0, 0],\n           [0, 1, 0, 0, 0],\n           [0, 0, 1, 0, 0],\n           [0, 0, 1, 0, 0],\n           [0, 0, 0, 0, 0]])\n    >>> line_nd([2, 1, 1], [5, 5, 2.5], endpoint=True)\n    (array([2, 3, 4, 4, 5]), array([1, 2, 3, 4, 5]), array([1, 1, 2, 2, 2]))\n    '
    start = np.asarray(start)
    stop = np.asarray(stop)
    npoints = int(np.ceil(np.max(np.abs(stop - start))))
    if endpoint:
        npoints += 1
    coords = np.linspace(start, stop, num=npoints, endpoint=endpoint).T
    if integer:
        for dim in range(len(start)):
            coords[dim, :] = _round_safe(coords[dim, :])
        coords = coords.astype(int)
    return tuple(coords)