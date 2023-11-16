from __future__ import annotations
__all__ = ['geometric_slerp']
import warnings
from typing import TYPE_CHECKING
import numpy as np
from scipy.spatial.distance import euclidean
if TYPE_CHECKING:
    import numpy.typing as npt

def _geometric_slerp(start, end, t):
    if False:
        i = 10
        return i + 15
    basis = np.vstack([start, end])
    (Q, R) = np.linalg.qr(basis.T)
    signs = 2 * (np.diag(R) >= 0) - 1
    Q = Q.T * signs.T[:, np.newaxis]
    R = R.T * signs.T[:, np.newaxis]
    c = np.dot(start, end)
    s = np.linalg.det(R)
    omega = np.arctan2(s, c)
    (start, end) = Q
    s = np.sin(t * omega)
    c = np.cos(t * omega)
    return start * c[:, np.newaxis] + end * s[:, np.newaxis]

def geometric_slerp(start: npt.ArrayLike, end: npt.ArrayLike, t: npt.ArrayLike, tol: float=1e-07) -> np.ndarray:
    if False:
        i = 10
        return i + 15
    "\n    Geometric spherical linear interpolation.\n\n    The interpolation occurs along a unit-radius\n    great circle arc in arbitrary dimensional space.\n\n    Parameters\n    ----------\n    start : (n_dimensions, ) array-like\n        Single n-dimensional input coordinate in a 1-D array-like\n        object. `n` must be greater than 1.\n    end : (n_dimensions, ) array-like\n        Single n-dimensional input coordinate in a 1-D array-like\n        object. `n` must be greater than 1.\n    t : float or (n_points,) 1D array-like\n        A float or 1D array-like of doubles representing interpolation\n        parameters, with values required in the inclusive interval\n        between 0 and 1. A common approach is to generate the array\n        with ``np.linspace(0, 1, n_pts)`` for linearly spaced points.\n        Ascending, descending, and scrambled orders are permitted.\n    tol : float\n        The absolute tolerance for determining if the start and end\n        coordinates are antipodes.\n\n    Returns\n    -------\n    result : (t.size, D)\n        An array of doubles containing the interpolated\n        spherical path and including start and\n        end when 0 and 1 t are used. The\n        interpolated values should correspond to the\n        same sort order provided in the t array. The result\n        may be 1-dimensional if ``t`` is a float.\n\n    Raises\n    ------\n    ValueError\n        If ``start`` and ``end`` are antipodes, not on the\n        unit n-sphere, or for a variety of degenerate conditions.\n\n    See Also\n    --------\n    scipy.spatial.transform.Slerp : 3-D Slerp that works with quaternions\n\n    Notes\n    -----\n    The implementation is based on the mathematical formula provided in [1]_,\n    and the first known presentation of this algorithm, derived from study of\n    4-D geometry, is credited to Glenn Davis in a footnote of the original\n    quaternion Slerp publication by Ken Shoemake [2]_.\n\n    .. versionadded:: 1.5.0\n\n    References\n    ----------\n    .. [1] https://en.wikipedia.org/wiki/Slerp#Geometric_Slerp\n    .. [2] Ken Shoemake (1985) Animating rotation with quaternion curves.\n           ACM SIGGRAPH Computer Graphics, 19(3): 245-254.\n\n    Examples\n    --------\n    Interpolate four linearly-spaced values on the circumference of\n    a circle spanning 90 degrees:\n\n    >>> import numpy as np\n    >>> from scipy.spatial import geometric_slerp\n    >>> import matplotlib.pyplot as plt\n    >>> fig = plt.figure()\n    >>> ax = fig.add_subplot(111)\n    >>> start = np.array([1, 0])\n    >>> end = np.array([0, 1])\n    >>> t_vals = np.linspace(0, 1, 4)\n    >>> result = geometric_slerp(start,\n    ...                          end,\n    ...                          t_vals)\n\n    The interpolated results should be at 30 degree intervals\n    recognizable on the unit circle:\n\n    >>> ax.scatter(result[...,0], result[...,1], c='k')\n    >>> circle = plt.Circle((0, 0), 1, color='grey')\n    >>> ax.add_artist(circle)\n    >>> ax.set_aspect('equal')\n    >>> plt.show()\n\n    Attempting to interpolate between antipodes on a circle is\n    ambiguous because there are two possible paths, and on a\n    sphere there are infinite possible paths on the geodesic surface.\n    Nonetheless, one of the ambiguous paths is returned along\n    with a warning:\n\n    >>> opposite_pole = np.array([-1, 0])\n    >>> with np.testing.suppress_warnings() as sup:\n    ...     sup.filter(UserWarning)\n    ...     geometric_slerp(start,\n    ...                     opposite_pole,\n    ...                     t_vals)\n    array([[ 1.00000000e+00,  0.00000000e+00],\n           [ 5.00000000e-01,  8.66025404e-01],\n           [-5.00000000e-01,  8.66025404e-01],\n           [-1.00000000e+00,  1.22464680e-16]])\n\n    Extend the original example to a sphere and plot interpolation\n    points in 3D:\n\n    >>> from mpl_toolkits.mplot3d import proj3d\n    >>> fig = plt.figure()\n    >>> ax = fig.add_subplot(111, projection='3d')\n\n    Plot the unit sphere for reference (optional):\n\n    >>> u = np.linspace(0, 2 * np.pi, 100)\n    >>> v = np.linspace(0, np.pi, 100)\n    >>> x = np.outer(np.cos(u), np.sin(v))\n    >>> y = np.outer(np.sin(u), np.sin(v))\n    >>> z = np.outer(np.ones(np.size(u)), np.cos(v))\n    >>> ax.plot_surface(x, y, z, color='y', alpha=0.1)\n\n    Interpolating over a larger number of points\n    may provide the appearance of a smooth curve on\n    the surface of the sphere, which is also useful\n    for discretized integration calculations on a\n    sphere surface:\n\n    >>> start = np.array([1, 0, 0])\n    >>> end = np.array([0, 0, 1])\n    >>> t_vals = np.linspace(0, 1, 200)\n    >>> result = geometric_slerp(start,\n    ...                          end,\n    ...                          t_vals)\n    >>> ax.plot(result[...,0],\n    ...         result[...,1],\n    ...         result[...,2],\n    ...         c='k')\n    >>> plt.show()\n    "
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    t = np.asarray(t)
    if t.ndim > 1:
        raise ValueError('The interpolation parameter value must be one dimensional.')
    if start.ndim != 1 or end.ndim != 1:
        raise ValueError('Start and end coordinates must be one-dimensional')
    if start.size != end.size:
        raise ValueError('The dimensions of start and end must match (have same size)')
    if start.size < 2 or end.size < 2:
        raise ValueError('The start and end coordinates must both be in at least two-dimensional space')
    if np.array_equal(start, end):
        return np.linspace(start, start, t.size)
    for coord in [start, end]:
        if not np.allclose(np.linalg.norm(coord), 1.0, rtol=1e-09, atol=0):
            raise ValueError('start and end are not on a unit n-sphere')
    if not isinstance(tol, float):
        raise ValueError('tol must be a float')
    else:
        tol = np.fabs(tol)
    coord_dist = euclidean(start, end)
    if np.allclose(coord_dist, 2.0, rtol=0, atol=tol):
        warnings.warn('start and end are antipodes using the specified tolerance; this may cause ambiguous slerp paths')
    t = np.asarray(t, dtype=np.float64)
    if t.size == 0:
        return np.empty((0, start.size))
    if t.min() < 0 or t.max() > 1:
        raise ValueError('interpolation parameter must be in [0, 1]')
    if t.ndim == 0:
        return _geometric_slerp(start, end, np.atleast_1d(t)).ravel()
    else:
        return _geometric_slerp(start, end, t)