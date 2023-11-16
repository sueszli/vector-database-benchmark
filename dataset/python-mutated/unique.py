import numpy as np

def unique_rows(ar):
    if False:
        for i in range(10):
            print('nop')
    'Remove repeated rows from a 2D array.\n\n    In particular, if given an array of coordinates of shape\n    (Npoints, Ndim), it will remove repeated points.\n\n    Parameters\n    ----------\n    ar : ndarray, shape (M, N)\n        The input array.\n\n    Returns\n    -------\n    ar_out : ndarray, shape (P, N)\n        A copy of the input array with repeated rows removed.\n\n    Raises\n    ------\n    ValueError : if `ar` is not two-dimensional.\n\n    Notes\n    -----\n    The function will generate a copy of `ar` if it is not\n    C-contiguous, which will negatively affect performance for large\n    input arrays.\n\n    Examples\n    --------\n    >>> ar = np.array([[1, 0, 1],\n    ...                [0, 1, 0],\n    ...                [1, 0, 1]], np.uint8)\n    >>> unique_rows(ar)\n    array([[0, 1, 0],\n           [1, 0, 1]], dtype=uint8)\n    '
    if ar.ndim != 2:
        raise ValueError(f'unique_rows() only makes sense for 2D arrays, got {ar.ndim}')
    ar = np.ascontiguousarray(ar)
    ar_row_view = ar.view(f'|S{ar.itemsize * ar.shape[1]}')
    (_, unique_row_indices) = np.unique(ar_row_view, return_index=True)
    ar_out = ar[unique_row_indices]
    return ar_out