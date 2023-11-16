import itertools
import numpy as np
from .._shared.utils import _supported_float_type, check_nD
from . import _moments_cy
from ._moments_analytical import moments_raw_to_central

def moments_coords(coords, order=3):
    if False:
        print('Hello World!')
    'Calculate all raw image moments up to a certain order.\n\n    The following properties can be calculated from raw image moments:\n     * Area as: ``M[0, 0]``.\n     * Centroid as: {``M[1, 0] / M[0, 0]``, ``M[0, 1] / M[0, 0]``}.\n\n    Note that raw moments are neither translation, scale, nor rotation\n    invariant.\n\n    Parameters\n    ----------\n    coords : (N, D) double or uint8 array\n        Array of N points that describe an image of D dimensionality in\n        Cartesian space.\n    order : int, optional\n        Maximum order of moments. Default is 3.\n\n    Returns\n    -------\n    M : (``order + 1``, ``order + 1``, ...) array\n        Raw image moments. (D dimensions)\n\n    References\n    ----------\n    .. [1] Johannes Kilian. Simple Image Analysis By Moments. Durham\n           University, version 0.2, Durham, 2001.\n\n    Examples\n    --------\n    >>> coords = np.array([[row, col]\n    ...                    for row in range(13, 17)\n    ...                    for col in range(14, 18)], dtype=np.float64)\n    >>> M = moments_coords(coords)\n    >>> centroid = (M[1, 0] / M[0, 0], M[0, 1] / M[0, 0])\n    >>> centroid\n    (14.5, 15.5)\n    '
    return moments_coords_central(coords, 0, order=order)

def moments_coords_central(coords, center=None, order=3):
    if False:
        while True:
            i = 10
    'Calculate all central image moments up to a certain order.\n\n    The following properties can be calculated from raw image moments:\n     * Area as: ``M[0, 0]``.\n     * Centroid as: {``M[1, 0] / M[0, 0]``, ``M[0, 1] / M[0, 0]``}.\n\n    Note that raw moments are neither translation, scale nor rotation\n    invariant.\n\n    Parameters\n    ----------\n    coords : (N, D) double or uint8 array\n        Array of N points that describe an image of D dimensionality in\n        Cartesian space. A tuple of coordinates as returned by\n        ``np.nonzero`` is also accepted as input.\n    center : tuple of float, optional\n        Coordinates of the image centroid. This will be computed if it\n        is not provided.\n    order : int, optional\n        Maximum order of moments. Default is 3.\n\n    Returns\n    -------\n    Mc : (``order + 1``, ``order + 1``, ...) array\n        Central image moments. (D dimensions)\n\n    References\n    ----------\n    .. [1] Johannes Kilian. Simple Image Analysis By Moments. Durham\n           University, version 0.2, Durham, 2001.\n\n    Examples\n    --------\n    >>> coords = np.array([[row, col]\n    ...                    for row in range(13, 17)\n    ...                    for col in range(14, 18)])\n    >>> moments_coords_central(coords)\n    array([[16.,  0., 20.,  0.],\n           [ 0.,  0.,  0.,  0.],\n           [20.,  0., 25.,  0.],\n           [ 0.,  0.,  0.,  0.]])\n\n    As seen above, for symmetric objects, odd-order moments (columns 1 and 3,\n    rows 1 and 3) are zero when centered on the centroid, or center of mass,\n    of the object (the default). If we break the symmetry by adding a new\n    point, this no longer holds:\n\n    >>> coords2 = np.concatenate((coords, [[17, 17]]), axis=0)\n    >>> np.round(moments_coords_central(coords2),\n    ...          decimals=2)  # doctest: +NORMALIZE_WHITESPACE\n    array([[17.  ,  0.  , 22.12, -2.49],\n           [ 0.  ,  3.53,  1.73,  7.4 ],\n           [25.88,  6.02, 36.63,  8.83],\n           [ 4.15, 19.17, 14.8 , 39.6 ]])\n\n    Image moments and central image moments are equivalent (by definition)\n    when the center is (0, 0):\n\n    >>> np.allclose(moments_coords(coords),\n    ...             moments_coords_central(coords, (0, 0)))\n    True\n    '
    if isinstance(coords, tuple):
        coords = np.stack(coords, axis=-1)
    check_nD(coords, 2)
    ndim = coords.shape[1]
    float_type = _supported_float_type(coords.dtype)
    if center is None:
        center = np.mean(coords, axis=0, dtype=float)
    coords = coords.astype(float_type, copy=False) - center
    coords = np.stack([coords ** c for c in range(order + 1)], axis=-1)
    coords = coords.reshape(coords.shape + (1,) * (ndim - 1))
    calc = 1
    for axis in range(ndim):
        isolated_axis = coords[:, axis]
        isolated_axis = np.moveaxis(isolated_axis, 1, 1 + axis)
        calc = calc * isolated_axis
    Mc = np.sum(calc, axis=0)
    return Mc

def moments(image, order=3, *, spacing=None):
    if False:
        return 10
    'Calculate all raw image moments up to a certain order.\n\n    The following properties can be calculated from raw image moments:\n     * Area as: ``M[0, 0]``.\n     * Centroid as: {``M[1, 0] / M[0, 0]``, ``M[0, 1] / M[0, 0]``}.\n\n    Note that raw moments are neither translation, scale nor rotation\n    invariant.\n\n    Parameters\n    ----------\n    image : (N[, ...]) double or uint8 array\n        Rasterized shape as image.\n    order : int, optional\n        Maximum order of moments. Default is 3.\n    spacing: tuple of float, shape (ndim,)\n        The pixel spacing along each axis of the image.\n\n    Returns\n    -------\n    m : (``order + 1``, ``order + 1``) array\n        Raw image moments.\n\n    References\n    ----------\n    .. [1] Wilhelm Burger, Mark Burge. Principles of Digital Image Processing:\n           Core Algorithms. Springer-Verlag, London, 2009.\n    .. [2] B. Jähne. Digital Image Processing. Springer-Verlag,\n           Berlin-Heidelberg, 6. edition, 2005.\n    .. [3] T. H. Reiss. Recognizing Planar Objects Using Invariant Image\n           Features, from Lecture notes in computer science, p. 676. Springer,\n           Berlin, 1993.\n    .. [4] https://en.wikipedia.org/wiki/Image_moment\n\n    Examples\n    --------\n    >>> image = np.zeros((20, 20), dtype=np.float64)\n    >>> image[13:17, 13:17] = 1\n    >>> M = moments(image)\n    >>> centroid = (M[1, 0] / M[0, 0], M[0, 1] / M[0, 0])\n    >>> centroid\n    (14.5, 14.5)\n    '
    return moments_central(image, (0,) * image.ndim, order=order, spacing=spacing)

def moments_central(image, center=None, order=3, *, spacing=None, **kwargs):
    if False:
        print('Hello World!')
    'Calculate all central image moments up to a certain order.\n\n    The center coordinates (cr, cc) can be calculated from the raw moments as:\n    {``M[1, 0] / M[0, 0]``, ``M[0, 1] / M[0, 0]``}.\n\n    Note that central moments are translation invariant but not scale and\n    rotation invariant.\n\n    Parameters\n    ----------\n    image : (N[, ...]) double or uint8 array\n        Rasterized shape as image.\n    center : tuple of float, optional\n        Coordinates of the image centroid. This will be computed if it\n        is not provided.\n    order : int, optional\n        The maximum order of moments computed.\n    spacing: tuple of float, shape (ndim,)\n        The pixel spacing along each axis of the image.\n\n    Returns\n    -------\n    mu : (``order + 1``, ``order + 1``) array\n        Central image moments.\n\n    References\n    ----------\n    .. [1] Wilhelm Burger, Mark Burge. Principles of Digital Image Processing:\n           Core Algorithms. Springer-Verlag, London, 2009.\n    .. [2] B. Jähne. Digital Image Processing. Springer-Verlag,\n           Berlin-Heidelberg, 6. edition, 2005.\n    .. [3] T. H. Reiss. Recognizing Planar Objects Using Invariant Image\n           Features, from Lecture notes in computer science, p. 676. Springer,\n           Berlin, 1993.\n    .. [4] https://en.wikipedia.org/wiki/Image_moment\n\n    Examples\n    --------\n    >>> image = np.zeros((20, 20), dtype=np.float64)\n    >>> image[13:17, 13:17] = 1\n    >>> M = moments(image)\n    >>> centroid = (M[1, 0] / M[0, 0], M[0, 1] / M[0, 0])\n    >>> moments_central(image, centroid)\n    array([[16.,  0., 20.,  0.],\n           [ 0.,  0.,  0.,  0.],\n           [20.,  0., 25.,  0.],\n           [ 0.,  0.,  0.,  0.]])\n    '
    if center is None:
        moments_raw = moments(image, order=order, spacing=spacing)
        return moments_raw_to_central(moments_raw)
    if spacing is None:
        spacing = np.ones(image.ndim)
    float_dtype = _supported_float_type(image.dtype)
    calc = image.astype(float_dtype, copy=False)
    for (dim, dim_length) in enumerate(image.shape):
        delta = np.arange(dim_length, dtype=float_dtype) * spacing[dim] - center[dim]
        powers_of_delta = delta[:, np.newaxis] ** np.arange(order + 1, dtype=float_dtype)
        calc = np.rollaxis(calc, dim, image.ndim)
        calc = np.dot(calc, powers_of_delta)
        calc = np.rollaxis(calc, -1, dim)
    return calc

def moments_normalized(mu, order=3, spacing=None):
    if False:
        while True:
            i = 10
    'Calculate all normalized central image moments up to a certain order.\n\n    Note that normalized central moments are translation and scale invariant\n    but not rotation invariant.\n\n    Parameters\n    ----------\n    mu : (M[, ...], M) array\n        Central image moments, where M must be greater than or equal\n        to ``order``.\n    order : int, optional\n        Maximum order of moments. Default is 3.\n\n    Returns\n    -------\n    nu : (``order + 1``[, ...], ``order + 1``) array\n        Normalized central image moments.\n\n    References\n    ----------\n    .. [1] Wilhelm Burger, Mark Burge. Principles of Digital Image Processing:\n           Core Algorithms. Springer-Verlag, London, 2009.\n    .. [2] B. Jähne. Digital Image Processing. Springer-Verlag,\n           Berlin-Heidelberg, 6. edition, 2005.\n    .. [3] T. H. Reiss. Recognizing Planar Objects Using Invariant Image\n           Features, from Lecture notes in computer science, p. 676. Springer,\n           Berlin, 1993.\n    .. [4] https://en.wikipedia.org/wiki/Image_moment\n\n    Examples\n    --------\n    >>> image = np.zeros((20, 20), dtype=np.float64)\n    >>> image[13:17, 13:17] = 1\n    >>> m = moments(image)\n    >>> centroid = (m[0, 1] / m[0, 0], m[1, 0] / m[0, 0])\n    >>> mu = moments_central(image, centroid)\n    >>> moments_normalized(mu)\n    array([[       nan,        nan, 0.078125  , 0.        ],\n           [       nan, 0.        , 0.        , 0.        ],\n           [0.078125  , 0.        , 0.00610352, 0.        ],\n           [0.        , 0.        , 0.        , 0.        ]])\n    '
    if np.any(np.array(mu.shape) <= order):
        raise ValueError('Shape of image moments must be >= `order`')
    if spacing is None:
        spacing = np.ones(mu.ndim)
    nu = np.zeros_like(mu)
    mu0 = mu.ravel()[0]
    scale = min(spacing)
    for powers in itertools.product(range(order + 1), repeat=mu.ndim):
        if sum(powers) < 2:
            nu[powers] = np.nan
        else:
            nu[powers] = mu[powers] / scale ** sum(powers) / mu0 ** (sum(powers) / nu.ndim + 1)
    return nu

def moments_hu(nu):
    if False:
        print('Hello World!')
    'Calculate Hu\'s set of image moments (2D-only).\n\n    Note that this set of moments is proved to be translation, scale and\n    rotation invariant.\n\n    Parameters\n    ----------\n    nu : (M, M) array\n        Normalized central image moments, where M must be >= 4.\n\n    Returns\n    -------\n    nu : (7,) array\n        Hu\'s set of image moments.\n\n    References\n    ----------\n    .. [1] M. K. Hu, "Visual Pattern Recognition by Moment Invariants",\n           IRE Trans. Info. Theory, vol. IT-8, pp. 179-187, 1962\n    .. [2] Wilhelm Burger, Mark Burge. Principles of Digital Image Processing:\n           Core Algorithms. Springer-Verlag, London, 2009.\n    .. [3] B. Jähne. Digital Image Processing. Springer-Verlag,\n           Berlin-Heidelberg, 6. edition, 2005.\n    .. [4] T. H. Reiss. Recognizing Planar Objects Using Invariant Image\n           Features, from Lecture notes in computer science, p. 676. Springer,\n           Berlin, 1993.\n    .. [5] https://en.wikipedia.org/wiki/Image_moment\n\n    Examples\n    --------\n    >>> image = np.zeros((20, 20), dtype=np.float64)\n    >>> image[13:17, 13:17] = 0.5\n    >>> image[10:12, 10:12] = 1\n    >>> mu = moments_central(image)\n    >>> nu = moments_normalized(mu)\n    >>> moments_hu(nu)\n    array([0.74537037, 0.35116598, 0.10404918, 0.04064421, 0.00264312,\n           0.02408546, 0.        ])\n    '
    dtype = np.float32 if nu.dtype == 'float32' else np.float64
    return _moments_cy.moments_hu(nu.astype(dtype, copy=False))

def centroid(image, *, spacing=None):
    if False:
        print('Hello World!')
    'Return the (weighted) centroid of an image.\n\n    Parameters\n    ----------\n    image : array\n        The input image.\n    spacing: tuple of float, shape (ndim,)\n        The pixel spacing along each axis of the image.\n\n    Returns\n    -------\n    center : tuple of float, length ``image.ndim``\n        The centroid of the (nonzero) pixels in ``image``.\n\n    Examples\n    --------\n    >>> image = np.zeros((20, 20), dtype=np.float64)\n    >>> image[13:17, 13:17] = 0.5\n    >>> image[10:12, 10:12] = 1\n    >>> centroid(image)\n    array([13.16666667, 13.16666667])\n    '
    M = moments_central(image, center=(0,) * image.ndim, order=1, spacing=spacing)
    center = M[tuple(np.eye(image.ndim, dtype=int))] / M[(0,) * image.ndim]
    return center

def inertia_tensor(image, mu=None, *, spacing=None):
    if False:
        i = 10
        return i + 15
    'Compute the inertia tensor of the input image.\n\n    Parameters\n    ----------\n    image : array\n        The input image.\n    mu : array, optional\n        The pre-computed central moments of ``image``. The inertia tensor\n        computation requires the central moments of the image. If an\n        application requires both the central moments and the inertia tensor\n        (for example, `skimage.measure.regionprops`), then it is more\n        efficient to pre-compute them and pass them to the inertia tensor\n        call.\n    spacing: tuple of float, shape (ndim,)\n        The pixel spacing along each axis of the image.\n\n    Returns\n    -------\n    T : array, shape ``(image.ndim, image.ndim)``\n        The inertia tensor of the input image. :math:`T_{i, j}` contains\n        the covariance of image intensity along axes :math:`i` and :math:`j`.\n\n    References\n    ----------\n    .. [1] https://en.wikipedia.org/wiki/Moment_of_inertia#Inertia_tensor\n    .. [2] Bernd Jähne. Spatio-Temporal Image Processing: Theory and\n           Scientific Applications. (Chapter 8: Tensor Methods) Springer, 1993.\n    '
    if mu is None:
        mu = moments_central(image, order=2, spacing=spacing)
    mu0 = mu[(0,) * image.ndim]
    result = np.zeros((image.ndim, image.ndim), dtype=mu.dtype)
    corners2 = tuple(2 * np.eye(image.ndim, dtype=int))
    d = np.diag(result)
    d.flags.writeable = True
    d[:] = (np.sum(mu[corners2]) - mu[corners2]) / mu0
    for dims in itertools.combinations(range(image.ndim), 2):
        mu_index = np.zeros(image.ndim, dtype=int)
        mu_index[list(dims)] = 1
        result[dims] = -mu[tuple(mu_index)] / mu0
        result.T[dims] = -mu[tuple(mu_index)] / mu0
    return result

def inertia_tensor_eigvals(image, mu=None, T=None, *, spacing=None):
    if False:
        return 10
    'Compute the eigenvalues of the inertia tensor of the image.\n\n    The inertia tensor measures covariance of the image intensity along\n    the image axes. (See `inertia_tensor`.) The relative magnitude of the\n    eigenvalues of the tensor is thus a measure of the elongation of a\n    (bright) object in the image.\n\n    Parameters\n    ----------\n    image : array\n        The input image.\n    mu : array, optional\n        The pre-computed central moments of ``image``.\n    T : array, shape ``(image.ndim, image.ndim)``\n        The pre-computed inertia tensor. If ``T`` is given, ``mu`` and\n        ``image`` are ignored.\n    spacing: tuple of float, shape (ndim,)\n        The pixel spacing along each axis of the image.\n\n    Returns\n    -------\n    eigvals : list of float, length ``image.ndim``\n        The eigenvalues of the inertia tensor of ``image``, in descending\n        order.\n\n    Notes\n    -----\n    Computing the eigenvalues requires the inertia tensor of the input image.\n    This is much faster if the central moments (``mu``) are provided, or,\n    alternatively, one can provide the inertia tensor (``T``) directly.\n    '
    if T is None:
        T = inertia_tensor(image, mu, spacing=spacing)
    eigvals = np.linalg.eigvalsh(T)
    eigvals = np.clip(eigvals, 0, None, out=eigvals)
    return sorted(eigvals, reverse=True)