from ._pnpoly import _grid_points_in_poly, _points_in_poly

def grid_points_in_poly(shape, verts, binarize=True):
    if False:
        i = 10
        return i + 15
    'Test whether points on a specified grid are inside a polygon.\n\n    For each ``(r, c)`` coordinate on a grid, i.e. ``(0, 0)``, ``(0, 1)`` etc.,\n    test whether that point lies inside a polygon.\n\n    You can control the output type with the `binarize` flag. Please refer to its\n    documentation for further details.\n\n    Parameters\n    ----------\n    shape : tuple (M, N)\n        Shape of the grid.\n    verts : (V, 2) array\n        Specify the V vertices of the polygon, sorted either clockwise\n        or anti-clockwise. The first point may (but does not need to be)\n        duplicated.\n    binarize: bool\n        If `True`, the output of the function is a boolean mask.\n        Otherwise, it is a labeled array. The labels are:\n        O - outside, 1 - inside, 2 - vertex, 3 - edge.\n\n    See Also\n    --------\n    points_in_poly\n\n    Returns\n    -------\n    mask : (M, N) ndarray\n        If `binarize` is True, the output is a boolean mask. True means the\n        corresponding pixel falls inside the polygon.\n        If `binarize` is False, the output is a labeled array, with pixels\n        having a label between 0 and 3. The meaning of the values is:\n        O - outside, 1 - inside, 2 - vertex, 3 - edge.\n\n    '
    output = _grid_points_in_poly(shape, verts)
    if binarize:
        output = output.astype(bool)
    return output

def points_in_poly(points, verts):
    if False:
        print('Hello World!')
    'Test whether points lie inside a polygon.\n\n    Parameters\n    ----------\n    points : (K, 2) array\n        Input points, ``(x, y)``.\n    verts : (L, 2) array\n        Vertices of the polygon, sorted either clockwise or anti-clockwise.\n        The first point may (but does not need to be) duplicated.\n\n    See Also\n    --------\n    grid_points_in_poly\n\n    Returns\n    -------\n    mask : (K,) array of bool\n        True if corresponding point is inside the polygon.\n\n    '
    return _points_in_poly(points, verts)