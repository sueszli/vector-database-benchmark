import numpy as np

def regular_grid(ar_shape, n_points):
    if False:
        for i in range(10):
            print('nop')
    'Find `n_points` regularly spaced along `ar_shape`.\n\n    The returned points (as slices) should be as close to cubically-spaced as\n    possible. Essentially, the points are spaced by the Nth root of the input\n    array size, where N is the number of dimensions. However, if an array\n    dimension cannot fit a full step size, it is "discarded", and the\n    computation is done for only the remaining dimensions.\n\n    Parameters\n    ----------\n    ar_shape : array-like of ints\n        The shape of the space embedding the grid. ``len(ar_shape)`` is the\n        number of dimensions.\n    n_points : int\n        The (approximate) number of points to embed in the space.\n\n    Returns\n    -------\n    slices : tuple of slice objects\n        A slice along each dimension of `ar_shape`, such that the intersection\n        of all the slices give the coordinates of regularly spaced points.\n\n        .. versionchanged:: 0.14.1\n            In scikit-image 0.14.1 and 0.15, the return type was changed from a\n            list to a tuple to ensure `compatibility with Numpy 1.15`_ and\n            higher. If your code requires the returned result to be a list, you\n            may convert the output of this function to a list with:\n\n            >>> result = list(regular_grid(ar_shape=(3, 20, 40), n_points=8))\n\n            .. _compatibility with NumPy 1.15: https://github.com/numpy/numpy/blob/master/doc/release/1.15.0-notes.rst#deprecations\n\n    Examples\n    --------\n    >>> ar = np.zeros((20, 40))\n    >>> g = regular_grid(ar.shape, 8)\n    >>> g\n    (slice(5, None, 10), slice(5, None, 10))\n    >>> ar[g] = 1\n    >>> ar.sum()\n    8.0\n    >>> ar = np.zeros((20, 40))\n    >>> g = regular_grid(ar.shape, 32)\n    >>> g\n    (slice(2, None, 5), slice(2, None, 5))\n    >>> ar[g] = 1\n    >>> ar.sum()\n    32.0\n    >>> ar = np.zeros((3, 20, 40))\n    >>> g = regular_grid(ar.shape, 8)\n    >>> g\n    (slice(1, None, 3), slice(5, None, 10), slice(5, None, 10))\n    >>> ar[g] = 1\n    >>> ar.sum()\n    8.0\n    '
    ar_shape = np.asanyarray(ar_shape)
    ndim = len(ar_shape)
    unsort_dim_idxs = np.argsort(np.argsort(ar_shape))
    sorted_dims = np.sort(ar_shape)
    space_size = float(np.prod(ar_shape))
    if space_size <= n_points:
        return (slice(None),) * ndim
    stepsizes = np.full(ndim, (space_size / n_points) ** (1.0 / ndim), dtype='float64')
    if (sorted_dims < stepsizes).any():
        for dim in range(ndim):
            stepsizes[dim] = sorted_dims[dim]
            space_size = float(np.prod(sorted_dims[dim + 1:]))
            stepsizes[dim + 1:] = (space_size / n_points) ** (1.0 / (ndim - dim - 1))
            if (sorted_dims >= stepsizes).all():
                break
    starts = (stepsizes // 2).astype(int)
    stepsizes = np.round(stepsizes).astype(int)
    slices = [slice(start, None, step) for (start, step) in zip(starts, stepsizes)]
    slices = tuple((slices[i] for i in unsort_dim_idxs))
    return slices

def regular_seeds(ar_shape, n_points, dtype=int):
    if False:
        while True:
            i = 10
    'Return an image with ~`n_points` regularly-spaced nonzero pixels.\n\n    Parameters\n    ----------\n    ar_shape : tuple of int\n        The shape of the desired output image.\n    n_points : int\n        The desired number of nonzero points.\n    dtype : numpy data type, optional\n        The desired data type of the output.\n\n    Returns\n    -------\n    seed_img : array of int or bool\n        The desired image.\n\n    Examples\n    --------\n    >>> regular_seeds((5, 5), 4)\n    array([[0, 0, 0, 0, 0],\n           [0, 1, 0, 2, 0],\n           [0, 0, 0, 0, 0],\n           [0, 3, 0, 4, 0],\n           [0, 0, 0, 0, 0]])\n    '
    grid = regular_grid(ar_shape, n_points)
    seed_img = np.zeros(ar_shape, dtype=dtype)
    seed_img[grid] = 1 + np.reshape(np.arange(seed_img[grid].size), seed_img[grid].shape)
    return seed_img