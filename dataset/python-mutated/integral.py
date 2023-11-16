import numpy as np

def integral_image(image, *, dtype=None):
    if False:
        while True:
            i = 10
    'Integral image / summed area table.\n\n    The integral image contains the sum of all elements above and to the\n    left of it, i.e.:\n\n    .. math::\n\n       S[m, n] = \\sum_{i \\leq m} \\sum_{j \\leq n} X[i, j]\n\n    Parameters\n    ----------\n    image : ndarray\n        Input image.\n\n    Returns\n    -------\n    S : ndarray\n        Integral image/summed area table of same shape as input image.\n\n    Notes\n    -----\n    For better accuracy and to avoid potential overflow, the data type of the\n    output may differ from the input\'s when the default dtype of None is used.\n    For inputs with integer dtype, the behavior matches that for\n    :func:`numpy.cumsum`. Floating point inputs will be promoted to at least\n    double precision. The user can set `dtype` to override this behavior.\n\n    References\n    ----------\n    .. [1] F.C. Crow, "Summed-area tables for texture mapping,"\n           ACM SIGGRAPH Computer Graphics, vol. 18, 1984, pp. 207-212.\n\n    '
    if dtype is None and image.real.dtype.kind == 'f':
        dtype = np.promote_types(image.dtype, np.float64)
    S = image
    for i in range(image.ndim):
        S = S.cumsum(axis=i, dtype=dtype)
    return S

def integrate(ii, start, end):
    if False:
        print('Hello World!')
    'Use an integral image to integrate over a given window.\n\n    Parameters\n    ----------\n    ii : ndarray\n        Integral image.\n    start : List of tuples, each tuple of length equal to dimension of `ii`\n        Coordinates of top left corner of window(s).\n        Each tuple in the list contains the starting row, col, ... index\n        i.e `[(row_win1, col_win1, ...), (row_win2, col_win2,...), ...]`.\n    end : List of tuples, each tuple of length equal to dimension of `ii`\n        Coordinates of bottom right corner of window(s).\n        Each tuple in the list containing the end row, col, ... index i.e\n        `[(row_win1, col_win1, ...), (row_win2, col_win2, ...), ...]`.\n\n    Returns\n    -------\n    S : scalar or ndarray\n        Integral (sum) over the given window(s).\n\n\n    Examples\n    --------\n    >>> arr = np.ones((5, 6), dtype=float)\n    >>> ii = integral_image(arr)\n    >>> integrate(ii, (1, 0), (1, 2))  # sum from (1, 0) to (1, 2)\n    array([3.])\n    >>> integrate(ii, [(3, 3)], [(4, 5)])  # sum from (3, 3) to (4, 5)\n    array([6.])\n    >>> # sum from (1, 0) to (1, 2) and from (3, 3) to (4, 5)\n    >>> integrate(ii, [(1, 0), (3, 3)], [(1, 2), (4, 5)])\n    array([3., 6.])\n    '
    start = np.atleast_2d(np.array(start))
    end = np.atleast_2d(np.array(end))
    rows = start.shape[0]
    total_shape = ii.shape
    total_shape = np.tile(total_shape, [rows, 1])
    start_negatives = start < 0
    end_negatives = end < 0
    start = (start + total_shape) * start_negatives + start * ~start_negatives
    end = (end + total_shape) * end_negatives + end * ~end_negatives
    if np.any(end - start < 0):
        raise IndexError('end coordinates must be greater or equal to start')
    S = np.zeros(rows)
    bit_perm = 2 ** ii.ndim
    width = len(bin(bit_perm - 1)[2:])
    for i in range(bit_perm):
        binary = bin(i)[2:].zfill(width)
        bool_mask = [bit == '1' for bit in binary]
        sign = (-1) ** sum(bool_mask)
        bad = [np.any((start[r] - 1) * bool_mask < 0) for r in range(rows)]
        corner_points = end * np.invert(bool_mask) + (start - 1) * bool_mask
        S += [sign * ii[tuple(corner_points[r])] if not bad[r] else 0 for r in range(rows)]
    return S