"""
Convenience interface to N-D interpolation

.. versionadded:: 0.9

"""
import numpy as np
from .interpnd import LinearNDInterpolator, NDInterpolatorBase, CloughTocher2DInterpolator, _ndim_coords_from_arrays
from scipy.spatial import cKDTree
__all__ = ['griddata', 'NearestNDInterpolator', 'LinearNDInterpolator', 'CloughTocher2DInterpolator']

class NearestNDInterpolator(NDInterpolatorBase):
    """NearestNDInterpolator(x, y).

    Nearest-neighbor interpolation in N > 1 dimensions.

    .. versionadded:: 0.9

    Methods
    -------
    __call__

    Parameters
    ----------
    x : (npoints, ndims) 2-D ndarray of floats
        Data point coordinates.
    y : (npoints, ) 1-D ndarray of float or complex
        Data values.
    rescale : boolean, optional
        Rescale points to unit cube before performing interpolation.
        This is useful if some of the input dimensions have
        incommensurable units and differ by many orders of magnitude.

        .. versionadded:: 0.14.0
    tree_options : dict, optional
        Options passed to the underlying ``cKDTree``.

        .. versionadded:: 0.17.0

    See Also
    --------
    griddata :
        Interpolate unstructured D-D data.
    LinearNDInterpolator :
        Piecewise linear interpolant in N dimensions.
    CloughTocher2DInterpolator :
        Piecewise cubic, C1 smooth, curvature-minimizing interpolant in 2D.
    interpn : Interpolation on a regular grid or rectilinear grid.
    RegularGridInterpolator : Interpolation on a regular or rectilinear grid
                              in arbitrary dimensions (`interpn` wraps this
                              class).

    Notes
    -----
    Uses ``scipy.spatial.cKDTree``

    .. note:: For data on a regular grid use `interpn` instead.

    Examples
    --------
    We can interpolate values on a 2D plane:

    >>> from scipy.interpolate import NearestNDInterpolator
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> rng = np.random.default_rng()
    >>> x = rng.random(10) - 0.5
    >>> y = rng.random(10) - 0.5
    >>> z = np.hypot(x, y)
    >>> X = np.linspace(min(x), max(x))
    >>> Y = np.linspace(min(y), max(y))
    >>> X, Y = np.meshgrid(X, Y)  # 2D grid for interpolation
    >>> interp = NearestNDInterpolator(list(zip(x, y)), z)
    >>> Z = interp(X, Y)
    >>> plt.pcolormesh(X, Y, Z, shading='auto')
    >>> plt.plot(x, y, "ok", label="input point")
    >>> plt.legend()
    >>> plt.colorbar()
    >>> plt.axis("equal")
    >>> plt.show()

    """

    def __init__(self, x, y, rescale=False, tree_options=None):
        if False:
            print('Hello World!')
        NDInterpolatorBase.__init__(self, x, y, rescale=rescale, need_contiguous=False, need_values=False)
        if tree_options is None:
            tree_options = dict()
        self.tree = cKDTree(self.points, **tree_options)
        self.values = np.asarray(y)

    def __call__(self, *args):
        if False:
            i = 10
            return i + 15
        '\n        Evaluate interpolator at given points.\n\n        Parameters\n        ----------\n        x1, x2, ... xn : array-like of float\n            Points where to interpolate data at.\n            x1, x2, ... xn can be array-like of float with broadcastable shape.\n            or x1 can be array-like of float with shape ``(..., ndim)``\n\n        '
        xi = _ndim_coords_from_arrays(args, ndim=self.points.shape[1])
        xi = self._check_call_shape(xi)
        xi = self._scale_x(xi)
        (dist, i) = self.tree.query(xi)
        return self.values[i]

def griddata(points, values, xi, method='linear', fill_value=np.nan, rescale=False):
    if False:
        for i in range(10):
            print('nop')
    "\n    Interpolate unstructured D-D data.\n\n    Parameters\n    ----------\n    points : 2-D ndarray of floats with shape (n, D), or length D tuple of 1-D ndarrays with shape (n,).\n        Data point coordinates.\n    values : ndarray of float or complex, shape (n,)\n        Data values.\n    xi : 2-D ndarray of floats with shape (m, D), or length D tuple of ndarrays broadcastable to the same shape.\n        Points at which to interpolate data.\n    method : {'linear', 'nearest', 'cubic'}, optional\n        Method of interpolation. One of\n\n        ``nearest``\n          return the value at the data point closest to\n          the point of interpolation. See `NearestNDInterpolator` for\n          more details.\n\n        ``linear``\n          tessellate the input point set to N-D\n          simplices, and interpolate linearly on each simplex. See\n          `LinearNDInterpolator` for more details.\n\n        ``cubic`` (1-D)\n          return the value determined from a cubic\n          spline.\n\n        ``cubic`` (2-D)\n          return the value determined from a\n          piecewise cubic, continuously differentiable (C1), and\n          approximately curvature-minimizing polynomial surface. See\n          `CloughTocher2DInterpolator` for more details.\n    fill_value : float, optional\n        Value used to fill in for requested points outside of the\n        convex hull of the input points. If not provided, then the\n        default is ``nan``. This option has no effect for the\n        'nearest' method.\n    rescale : bool, optional\n        Rescale points to unit cube before performing interpolation.\n        This is useful if some of the input dimensions have\n        incommensurable units and differ by many orders of magnitude.\n\n        .. versionadded:: 0.14.0\n\n    Returns\n    -------\n    ndarray\n        Array of interpolated values.\n\n    See Also\n    --------\n    LinearNDInterpolator :\n        Piecewise linear interpolant in N dimensions.\n    NearestNDInterpolator :\n        Nearest-neighbor interpolation in N dimensions.\n    CloughTocher2DInterpolator :\n        Piecewise cubic, C1 smooth, curvature-minimizing interpolant in 2D.\n    interpn : Interpolation on a regular grid or rectilinear grid.\n    RegularGridInterpolator : Interpolation on a regular or rectilinear grid\n                              in arbitrary dimensions (`interpn` wraps this\n                              class).\n\n    Notes\n    -----\n\n    .. versionadded:: 0.9\n\n    .. note:: For data on a regular grid use `interpn` instead.\n\n    Examples\n    --------\n\n    Suppose we want to interpolate the 2-D function\n\n    >>> import numpy as np\n    >>> def func(x, y):\n    ...     return x*(1-x)*np.cos(4*np.pi*x) * np.sin(4*np.pi*y**2)**2\n\n    on a grid in [0, 1]x[0, 1]\n\n    >>> grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]\n\n    but we only know its values at 1000 data points:\n\n    >>> rng = np.random.default_rng()\n    >>> points = rng.random((1000, 2))\n    >>> values = func(points[:,0], points[:,1])\n\n    This can be done with `griddata` -- below we try out all of the\n    interpolation methods:\n\n    >>> from scipy.interpolate import griddata\n    >>> grid_z0 = griddata(points, values, (grid_x, grid_y), method='nearest')\n    >>> grid_z1 = griddata(points, values, (grid_x, grid_y), method='linear')\n    >>> grid_z2 = griddata(points, values, (grid_x, grid_y), method='cubic')\n\n    One can see that the exact result is reproduced by all of the\n    methods to some degree, but for this smooth function the piecewise\n    cubic interpolant gives the best results:\n\n    >>> import matplotlib.pyplot as plt\n    >>> plt.subplot(221)\n    >>> plt.imshow(func(grid_x, grid_y).T, extent=(0,1,0,1), origin='lower')\n    >>> plt.plot(points[:,0], points[:,1], 'k.', ms=1)\n    >>> plt.title('Original')\n    >>> plt.subplot(222)\n    >>> plt.imshow(grid_z0.T, extent=(0,1,0,1), origin='lower')\n    >>> plt.title('Nearest')\n    >>> plt.subplot(223)\n    >>> plt.imshow(grid_z1.T, extent=(0,1,0,1), origin='lower')\n    >>> plt.title('Linear')\n    >>> plt.subplot(224)\n    >>> plt.imshow(grid_z2.T, extent=(0,1,0,1), origin='lower')\n    >>> plt.title('Cubic')\n    >>> plt.gcf().set_size_inches(6, 6)\n    >>> plt.show()\n\n    "
    points = _ndim_coords_from_arrays(points)
    if points.ndim < 2:
        ndim = points.ndim
    else:
        ndim = points.shape[-1]
    if ndim == 1 and method in ('nearest', 'linear', 'cubic'):
        from ._interpolate import interp1d
        points = points.ravel()
        if isinstance(xi, tuple):
            if len(xi) != 1:
                raise ValueError('invalid number of dimensions in xi')
            (xi,) = xi
        idx = np.argsort(points)
        points = points[idx]
        values = values[idx]
        if method == 'nearest':
            fill_value = 'extrapolate'
        ip = interp1d(points, values, kind=method, axis=0, bounds_error=False, fill_value=fill_value)
        return ip(xi)
    elif method == 'nearest':
        ip = NearestNDInterpolator(points, values, rescale=rescale)
        return ip(xi)
    elif method == 'linear':
        ip = LinearNDInterpolator(points, values, fill_value=fill_value, rescale=rescale)
        return ip(xi)
    elif method == 'cubic' and ndim == 2:
        ip = CloughTocher2DInterpolator(points, values, fill_value=fill_value, rescale=rescale)
        return ip(xi)
    else:
        raise ValueError('Unknown interpolation method %r for %d dimensional data' % (method, ndim))