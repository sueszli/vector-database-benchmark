"""
Interpolation inside triangular grids.
"""
import numpy as np
from matplotlib import _api
from matplotlib.tri import Triangulation
from matplotlib.tri._trifinder import TriFinder
from matplotlib.tri._tritools import TriAnalyzer
__all__ = ('TriInterpolator', 'LinearTriInterpolator', 'CubicTriInterpolator')

class TriInterpolator:
    """
    Abstract base class for classes used to interpolate on a triangular grid.

    Derived classes implement the following methods:

    - ``__call__(x, y)``,
      where x, y are array-like point coordinates of the same shape, and
      that returns a masked array of the same shape containing the
      interpolated z-values.

    - ``gradient(x, y)``,
      where x, y are array-like point coordinates of the same
      shape, and that returns a list of 2 masked arrays of the same shape
      containing the 2 derivatives of the interpolator (derivatives of
      interpolated z values with respect to x and y).
    """

    def __init__(self, triangulation, z, trifinder=None):
        if False:
            print('Hello World!')
        _api.check_isinstance(Triangulation, triangulation=triangulation)
        self._triangulation = triangulation
        self._z = np.asarray(z)
        if self._z.shape != self._triangulation.x.shape:
            raise ValueError('z array must have same length as triangulation x and y arrays')
        _api.check_isinstance((TriFinder, None), trifinder=trifinder)
        self._trifinder = trifinder or self._triangulation.get_trifinder()
        self._unit_x = 1.0
        self._unit_y = 1.0
        self._tri_renum = None
    _docstring__call__ = '\n        Returns a masked array containing interpolated values at the specified\n        (x, y) points.\n\n        Parameters\n        ----------\n        x, y : array-like\n            x and y coordinates of the same shape and any number of\n            dimensions.\n\n        Returns\n        -------\n        np.ma.array\n            Masked array of the same shape as *x* and *y*; values corresponding\n            to (*x*, *y*) points outside of the triangulation are masked out.\n\n        '
    _docstringgradient = '\n        Returns a list of 2 masked arrays containing interpolated derivatives\n        at the specified (x, y) points.\n\n        Parameters\n        ----------\n        x, y : array-like\n            x and y coordinates of the same shape and any number of\n            dimensions.\n\n        Returns\n        -------\n        dzdx, dzdy : np.ma.array\n            2 masked arrays of the same shape as *x* and *y*; values\n            corresponding to (x, y) points outside of the triangulation\n            are masked out.\n            The first returned array contains the values of\n            :math:`\\frac{\\partial z}{\\partial x}` and the second those of\n            :math:`\\frac{\\partial z}{\\partial y}`.\n\n        '

    def _interpolate_multikeys(self, x, y, tri_index=None, return_keys=('z',)):
        if False:
            return 10
        "\n        Versatile (private) method defined for all TriInterpolators.\n\n        :meth:`_interpolate_multikeys` is a wrapper around method\n        :meth:`_interpolate_single_key` (to be defined in the child\n        subclasses).\n        :meth:`_interpolate_single_key actually performs the interpolation,\n        but only for 1-dimensional inputs and at valid locations (inside\n        unmasked triangles of the triangulation).\n\n        The purpose of :meth:`_interpolate_multikeys` is to implement the\n        following common tasks needed in all subclasses implementations:\n\n        - calculation of containing triangles\n        - dealing with more than one interpolation request at the same\n          location (e.g., if the 2 derivatives are requested, it is\n          unnecessary to compute the containing triangles twice)\n        - scaling according to self._unit_x, self._unit_y\n        - dealing with points outside of the grid (with fill value np.nan)\n        - dealing with multi-dimensional *x*, *y* arrays: flattening for\n          :meth:`_interpolate_params` call and final reshaping.\n\n        (Note that np.vectorize could do most of those things very well for\n        you, but it does it by function evaluations over successive tuples of\n        the input arrays. Therefore, this tends to be more time-consuming than\n        using optimized numpy functions - e.g., np.dot - which can be used\n        easily on the flattened inputs, in the child-subclass methods\n        :meth:`_interpolate_single_key`.)\n\n        It is guaranteed that the calls to :meth:`_interpolate_single_key`\n        will be done with flattened (1-d) array-like input parameters *x*, *y*\n        and with flattened, valid `tri_index` arrays (no -1 index allowed).\n\n        Parameters\n        ----------\n        x, y : array-like\n            x and y coordinates where interpolated values are requested.\n        tri_index : array-like of int, optional\n            Array of the containing triangle indices, same shape as\n            *x* and *y*. Defaults to None. If None, these indices\n            will be computed by a TriFinder instance.\n            (Note: For point outside the grid, tri_index[ipt] shall be -1).\n        return_keys : tuple of keys from {'z', 'dzdx', 'dzdy'}\n            Defines the interpolation arrays to return, and in which order.\n\n        Returns\n        -------\n        list of arrays\n            Each array-like contains the expected interpolated values in the\n            order defined by *return_keys* parameter.\n        "
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        sh_ret = x.shape
        if x.shape != y.shape:
            raise ValueError(f'x and y shall have same shapes. Given: {x.shape} and {y.shape}')
        x = np.ravel(x)
        y = np.ravel(y)
        x_scaled = x / self._unit_x
        y_scaled = y / self._unit_y
        size_ret = np.size(x_scaled)
        if tri_index is None:
            tri_index = self._trifinder(x, y)
        else:
            if tri_index.shape != sh_ret:
                raise ValueError(f'tri_index array is provided and shall have same shape as x and y. Given: {tri_index.shape} and {sh_ret}')
            tri_index = np.ravel(tri_index)
        mask_in = tri_index != -1
        if self._tri_renum is None:
            valid_tri_index = tri_index[mask_in]
        else:
            valid_tri_index = self._tri_renum[tri_index[mask_in]]
        valid_x = x_scaled[mask_in]
        valid_y = y_scaled[mask_in]
        ret = []
        for return_key in return_keys:
            try:
                return_index = {'z': 0, 'dzdx': 1, 'dzdy': 2}[return_key]
            except KeyError as err:
                raise ValueError("return_keys items shall take values in {'z', 'dzdx', 'dzdy'}") from err
            scale = [1.0, 1.0 / self._unit_x, 1.0 / self._unit_y][return_index]
            ret_loc = np.empty(size_ret, dtype=np.float64)
            ret_loc[~mask_in] = np.nan
            ret_loc[mask_in] = self._interpolate_single_key(return_key, valid_tri_index, valid_x, valid_y) * scale
            ret += [np.ma.masked_invalid(ret_loc.reshape(sh_ret), copy=False)]
        return ret

    def _interpolate_single_key(self, return_key, tri_index, x, y):
        if False:
            for i in range(10):
                print('nop')
        "\n        Interpolate at points belonging to the triangulation\n        (inside an unmasked triangles).\n\n        Parameters\n        ----------\n        return_key : {'z', 'dzdx', 'dzdy'}\n            The requested values (z or its derivatives).\n        tri_index : 1D int array\n            Valid triangle index (cannot be -1).\n        x, y : 1D arrays, same shape as `tri_index`\n            Valid locations where interpolation is requested.\n\n        Returns\n        -------\n        1-d array\n            Returned array of the same size as *tri_index*\n        "
        raise NotImplementedError('TriInterpolator subclasses' + 'should implement _interpolate_single_key!')

class LinearTriInterpolator(TriInterpolator):
    """
    Linear interpolator on a triangular grid.

    Each triangle is represented by a plane so that an interpolated value at
    point (x, y) lies on the plane of the triangle containing (x, y).
    Interpolated values are therefore continuous across the triangulation, but
    their first derivatives are discontinuous at edges between triangles.

    Parameters
    ----------
    triangulation : `~matplotlib.tri.Triangulation`
        The triangulation to interpolate over.
    z : (npoints,) array-like
        Array of values, defined at grid points, to interpolate between.
    trifinder : `~matplotlib.tri.TriFinder`, optional
        If this is not specified, the Triangulation's default TriFinder will
        be used by calling `.Triangulation.get_trifinder`.

    Methods
    -------
    `__call__` (x, y) : Returns interpolated values at (x, y) points.
    `gradient` (x, y) : Returns interpolated derivatives at (x, y) points.

    """

    def __init__(self, triangulation, z, trifinder=None):
        if False:
            while True:
                i = 10
        super().__init__(triangulation, z, trifinder)
        self._plane_coefficients = self._triangulation.calculate_plane_coefficients(self._z)

    def __call__(self, x, y):
        if False:
            return 10
        return self._interpolate_multikeys(x, y, tri_index=None, return_keys=('z',))[0]
    __call__.__doc__ = TriInterpolator._docstring__call__

    def gradient(self, x, y):
        if False:
            for i in range(10):
                print('nop')
        return self._interpolate_multikeys(x, y, tri_index=None, return_keys=('dzdx', 'dzdy'))
    gradient.__doc__ = TriInterpolator._docstringgradient

    def _interpolate_single_key(self, return_key, tri_index, x, y):
        if False:
            for i in range(10):
                print('nop')
        _api.check_in_list(['z', 'dzdx', 'dzdy'], return_key=return_key)
        if return_key == 'z':
            return self._plane_coefficients[tri_index, 0] * x + self._plane_coefficients[tri_index, 1] * y + self._plane_coefficients[tri_index, 2]
        elif return_key == 'dzdx':
            return self._plane_coefficients[tri_index, 0]
        else:
            return self._plane_coefficients[tri_index, 1]

class CubicTriInterpolator(TriInterpolator):
    """
    Cubic interpolator on a triangular grid.

    In one-dimension - on a segment - a cubic interpolating function is
    defined by the values of the function and its derivative at both ends.
    This is almost the same in 2D inside a triangle, except that the values
    of the function and its 2 derivatives have to be defined at each triangle
    node.

    The CubicTriInterpolator takes the value of the function at each node -
    provided by the user - and internally computes the value of the
    derivatives, resulting in a smooth interpolation.
    (As a special feature, the user can also impose the value of the
    derivatives at each node, but this is not supposed to be the common
    usage.)

    Parameters
    ----------
    triangulation : `~matplotlib.tri.Triangulation`
        The triangulation to interpolate over.
    z : (npoints,) array-like
        Array of values, defined at grid points, to interpolate between.
    kind : {'min_E', 'geom', 'user'}, optional
        Choice of the smoothing algorithm, in order to compute
        the interpolant derivatives (defaults to 'min_E'):

        - if 'min_E': (default) The derivatives at each node is computed
          to minimize a bending energy.
        - if 'geom': The derivatives at each node is computed as a
          weighted average of relevant triangle normals. To be used for
          speed optimization (large grids).
        - if 'user': The user provides the argument *dz*, no computation
          is hence needed.

    trifinder : `~matplotlib.tri.TriFinder`, optional
        If not specified, the Triangulation's default TriFinder will
        be used by calling `.Triangulation.get_trifinder`.
    dz : tuple of array-likes (dzdx, dzdy), optional
        Used only if  *kind* ='user'. In this case *dz* must be provided as
        (dzdx, dzdy) where dzdx, dzdy are arrays of the same shape as *z* and
        are the interpolant first derivatives at the *triangulation* points.

    Methods
    -------
    `__call__` (x, y) : Returns interpolated values at (x, y) points.
    `gradient` (x, y) : Returns interpolated derivatives at (x, y) points.

    Notes
    -----
    This note is a bit technical and details how the cubic interpolation is
    computed.

    The interpolation is based on a Clough-Tocher subdivision scheme of
    the *triangulation* mesh (to make it clearer, each triangle of the
    grid will be divided in 3 child-triangles, and on each child triangle
    the interpolated function is a cubic polynomial of the 2 coordinates).
    This technique originates from FEM (Finite Element Method) analysis;
    the element used is a reduced Hsieh-Clough-Tocher (HCT)
    element. Its shape functions are described in [1]_.
    The assembled function is guaranteed to be C1-smooth, i.e. it is
    continuous and its first derivatives are also continuous (this
    is easy to show inside the triangles but is also true when crossing the
    edges).

    In the default case (*kind* ='min_E'), the interpolant minimizes a
    curvature energy on the functional space generated by the HCT element
    shape functions - with imposed values but arbitrary derivatives at each
    node. The minimized functional is the integral of the so-called total
    curvature (implementation based on an algorithm from [2]_ - PCG sparse
    solver):

    .. math::

        E(z) = \\frac{1}{2} \\int_{\\Omega} \\left(
            \\left( \\frac{\\partial^2{z}}{\\partial{x}^2} \\right)^2 +
            \\left( \\frac{\\partial^2{z}}{\\partial{y}^2} \\right)^2 +
            2\\left( \\frac{\\partial^2{z}}{\\partial{y}\\partial{x}} \\right)^2
        \\right) dx\\,dy

    If the case *kind* ='geom' is chosen by the user, a simple geometric
    approximation is used (weighted average of the triangle normal
    vectors), which could improve speed on very large grids.

    References
    ----------
    .. [1] Michel Bernadou, Kamal Hassan, "Basis functions for general
        Hsieh-Clough-Tocher triangles, complete or reduced.",
        International Journal for Numerical Methods in Engineering,
        17(5):784 - 789. 2.01.
    .. [2] C.T. Kelley, "Iterative Methods for Optimization".

    """

    def __init__(self, triangulation, z, kind='min_E', trifinder=None, dz=None):
        if False:
            while True:
                i = 10
        super().__init__(triangulation, z, trifinder)
        self._triangulation.get_cpp_triangulation()
        tri_analyzer = TriAnalyzer(self._triangulation)
        (compressed_triangles, compressed_x, compressed_y, tri_renum, node_renum) = tri_analyzer._get_compressed_triangulation()
        self._triangles = compressed_triangles
        self._tri_renum = tri_renum
        valid_node = node_renum != -1
        self._z[node_renum[valid_node]] = self._z[valid_node]
        self._unit_x = np.ptp(compressed_x)
        self._unit_y = np.ptp(compressed_y)
        self._pts = np.column_stack([compressed_x / self._unit_x, compressed_y / self._unit_y])
        self._tris_pts = self._pts[self._triangles]
        self._eccs = self._compute_tri_eccentricities(self._tris_pts)
        _api.check_in_list(['user', 'geom', 'min_E'], kind=kind)
        self._dof = self._compute_dof(kind, dz=dz)
        self._ReferenceElement = _ReducedHCT_Element()

    def __call__(self, x, y):
        if False:
            return 10
        return self._interpolate_multikeys(x, y, tri_index=None, return_keys=('z',))[0]
    __call__.__doc__ = TriInterpolator._docstring__call__

    def gradient(self, x, y):
        if False:
            while True:
                i = 10
        return self._interpolate_multikeys(x, y, tri_index=None, return_keys=('dzdx', 'dzdy'))
    gradient.__doc__ = TriInterpolator._docstringgradient

    def _interpolate_single_key(self, return_key, tri_index, x, y):
        if False:
            return 10
        _api.check_in_list(['z', 'dzdx', 'dzdy'], return_key=return_key)
        tris_pts = self._tris_pts[tri_index]
        alpha = self._get_alpha_vec(x, y, tris_pts)
        ecc = self._eccs[tri_index]
        dof = np.expand_dims(self._dof[tri_index], axis=1)
        if return_key == 'z':
            return self._ReferenceElement.get_function_values(alpha, ecc, dof)
        else:
            J = self._get_jacobian(tris_pts)
            dzdx = self._ReferenceElement.get_function_derivatives(alpha, J, ecc, dof)
            if return_key == 'dzdx':
                return dzdx[:, 0, 0]
            else:
                return dzdx[:, 1, 0]

    def _compute_dof(self, kind, dz=None):
        if False:
            return 10
        "\n        Compute and return nodal dofs according to kind.\n\n        Parameters\n        ----------\n        kind : {'min_E', 'geom', 'user'}\n            Choice of the _DOF_estimator subclass to estimate the gradient.\n        dz : tuple of array-likes (dzdx, dzdy), optional\n            Used only if *kind*=user; in this case passed to the\n            :class:`_DOF_estimator_user`.\n\n        Returns\n        -------\n        array-like, shape (npts, 2)\n            Estimation of the gradient at triangulation nodes (stored as\n            degree of freedoms of reduced-HCT triangle elements).\n        "
        if kind == 'user':
            if dz is None:
                raise ValueError("For a CubicTriInterpolator with *kind*='user', a valid *dz* argument is expected.")
            TE = _DOF_estimator_user(self, dz=dz)
        elif kind == 'geom':
            TE = _DOF_estimator_geom(self)
        else:
            TE = _DOF_estimator_min_E(self)
        return TE.compute_dof_from_df()

    @staticmethod
    def _get_alpha_vec(x, y, tris_pts):
        if False:
            i = 10
            return i + 15
        '\n        Fast (vectorized) function to compute barycentric coordinates alpha.\n\n        Parameters\n        ----------\n        x, y : array-like of dim 1 (shape (nx,))\n            Coordinates of the points whose points barycentric coordinates are\n            requested.\n        tris_pts : array like of dim 3 (shape: (nx, 3, 2))\n            Coordinates of the containing triangles apexes.\n\n        Returns\n        -------\n        array of dim 2 (shape (nx, 3))\n            Barycentric coordinates of the points inside the containing\n            triangles.\n        '
        ndim = tris_pts.ndim - 2
        a = tris_pts[:, 1, :] - tris_pts[:, 0, :]
        b = tris_pts[:, 2, :] - tris_pts[:, 0, :]
        abT = np.stack([a, b], axis=-1)
        ab = _transpose_vectorized(abT)
        OM = np.stack([x, y], axis=1) - tris_pts[:, 0, :]
        metric = ab @ abT
        metric_inv = _pseudo_inv22sym_vectorized(metric)
        Covar = ab @ _transpose_vectorized(np.expand_dims(OM, ndim))
        ksi = metric_inv @ Covar
        alpha = _to_matrix_vectorized([[1 - ksi[:, 0, 0] - ksi[:, 1, 0]], [ksi[:, 0, 0]], [ksi[:, 1, 0]]])
        return alpha

    @staticmethod
    def _get_jacobian(tris_pts):
        if False:
            print('Hello World!')
        '\n        Fast (vectorized) function to compute triangle jacobian matrix.\n\n        Parameters\n        ----------\n        tris_pts : array like of dim 3 (shape: (nx, 3, 2))\n            Coordinates of the containing triangles apexes.\n\n        Returns\n        -------\n        array of dim 3 (shape (nx, 2, 2))\n            Barycentric coordinates of the points inside the containing\n            triangles.\n            J[itri, :, :] is the jacobian matrix at apex 0 of the triangle\n            itri, so that the following (matrix) relationship holds:\n               [dz/dksi] = [J] x [dz/dx]\n            with x: global coordinates\n                 ksi: element parametric coordinates in triangle first apex\n                 local basis.\n        '
        a = np.array(tris_pts[:, 1, :] - tris_pts[:, 0, :])
        b = np.array(tris_pts[:, 2, :] - tris_pts[:, 0, :])
        J = _to_matrix_vectorized([[a[:, 0], a[:, 1]], [b[:, 0], b[:, 1]]])
        return J

    @staticmethod
    def _compute_tri_eccentricities(tris_pts):
        if False:
            while True:
                i = 10
        '\n        Compute triangle eccentricities.\n\n        Parameters\n        ----------\n        tris_pts : array like of dim 3 (shape: (nx, 3, 2))\n            Coordinates of the triangles apexes.\n\n        Returns\n        -------\n        array like of dim 2 (shape: (nx, 3))\n            The so-called eccentricity parameters [1] needed for HCT triangular\n            element.\n        '
        a = np.expand_dims(tris_pts[:, 2, :] - tris_pts[:, 1, :], axis=2)
        b = np.expand_dims(tris_pts[:, 0, :] - tris_pts[:, 2, :], axis=2)
        c = np.expand_dims(tris_pts[:, 1, :] - tris_pts[:, 0, :], axis=2)
        dot_a = (_transpose_vectorized(a) @ a)[:, 0, 0]
        dot_b = (_transpose_vectorized(b) @ b)[:, 0, 0]
        dot_c = (_transpose_vectorized(c) @ c)[:, 0, 0]
        return _to_matrix_vectorized([[(dot_c - dot_b) / dot_a], [(dot_a - dot_c) / dot_b], [(dot_b - dot_a) / dot_c]])

class _ReducedHCT_Element:
    """
    Implementation of reduced HCT triangular element with explicit shape
    functions.

    Computes z, dz, d2z and the element stiffness matrix for bending energy:
    E(f) = integral( (d2z/dx2 + d2z/dy2)**2 dA)

    *** Reference for the shape functions: ***
    [1] Basis functions for general Hsieh-Clough-Tocher _triangles, complete or
        reduced.
        Michel Bernadou, Kamal Hassan
        International Journal for Numerical Methods in Engineering.
        17(5):784 - 789.  2.01

    *** Element description: ***
    9 dofs: z and dz given at 3 apex
    C1 (conform)

    """
    M = np.array([[0.0, 0.0, 0.0, 4.5, 4.5, 0.0, 0.0, 0.0, 0.0, 0.0], [-0.25, 0.0, 0.0, 0.5, 1.25, 0.0, 0.0, 0.0, 0.0, 0.0], [-0.25, 0.0, 0.0, 1.25, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5, 1.0, 0.0, -1.5, 0.0, 3.0, 3.0, 0.0, 0.0, 3.0], [0.0, 0.0, 0.0, -0.25, 0.25, 0.0, 1.0, 0.0, 0.0, 0.5], [0.25, 0.0, 0.0, -0.5, -0.25, 1.0, 0.0, 0.0, 0.0, 1.0], [0.5, 0.0, 1.0, 0.0, -1.5, 0.0, 0.0, 3.0, 3.0, 3.0], [0.25, 0.0, 0.0, -0.25, -0.5, 0.0, 0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.25, -0.25, 0.0, 0.0, 1.0, 0.0, 0.5]])
    M0 = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 1.5, 1.5, 0.0, 0.0, 0.0, 0.0, -3.0], [-0.5, 0.0, 0.0, 0.75, 0.75, 0.0, 0.0, 0.0, 0.0, -1.5], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, -1.5, -1.5, 0.0, 0.0, 0.0, 0.0, 3.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5, 0.0, 0.0, -0.75, -0.75, 0.0, 0.0, 0.0, 0.0, 1.5]])
    M1 = np.array([[-0.5, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [-0.25, 0.0, 0.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5, 0.0, 0.0, -1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.25, 0.0, 0.0, -0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    M2 = np.array([[0.5, 0.0, 0.0, 0.0, -1.5, 0.0, 0.0, 0.0, 0.0, 0.0], [0.25, 0.0, 0.0, 0.0, -0.75, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [-0.5, 0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [-0.25, 0.0, 0.0, 0.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    rotate_dV = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [-1.0, -1.0], [-1.0, -1.0], [1.0, 0.0]])
    rotate_d2V = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, -2.0, -1.0], [1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [-2.0, 0.0, -1.0]])
    n_gauss = 9
    gauss_pts = np.array([[13.0 / 18.0, 4.0 / 18.0, 1.0 / 18.0], [4.0 / 18.0, 13.0 / 18.0, 1.0 / 18.0], [7.0 / 18.0, 7.0 / 18.0, 4.0 / 18.0], [1.0 / 18.0, 13.0 / 18.0, 4.0 / 18.0], [1.0 / 18.0, 4.0 / 18.0, 13.0 / 18.0], [4.0 / 18.0, 7.0 / 18.0, 7.0 / 18.0], [4.0 / 18.0, 1.0 / 18.0, 13.0 / 18.0], [13.0 / 18.0, 1.0 / 18.0, 4.0 / 18.0], [7.0 / 18.0, 4.0 / 18.0, 7.0 / 18.0]], dtype=np.float64)
    gauss_w = np.ones([9], dtype=np.float64) / 9.0
    E = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]])
    J0_to_J1 = np.array([[-1.0, 1.0], [-1.0, 0.0]])
    J0_to_J2 = np.array([[0.0, -1.0], [1.0, -1.0]])

    def get_function_values(self, alpha, ecc, dofs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Parameters\n        ----------\n        alpha : is a (N x 3 x 1) array (array of column-matrices) of\n        barycentric coordinates,\n        ecc : is a (N x 3 x 1) array (array of column-matrices) of triangle\n        eccentricities,\n        dofs : is a (N x 1 x 9) arrays (arrays of row-matrices) of computed\n        degrees of freedom.\n\n        Returns\n        -------\n        Returns the N-array of interpolated function values.\n        '
        subtri = np.argmin(alpha, axis=1)[:, 0]
        ksi = _roll_vectorized(alpha, -subtri, axis=0)
        E = _roll_vectorized(ecc, -subtri, axis=0)
        x = ksi[:, 0, 0]
        y = ksi[:, 1, 0]
        z = ksi[:, 2, 0]
        x_sq = x * x
        y_sq = y * y
        z_sq = z * z
        V = _to_matrix_vectorized([[x_sq * x], [y_sq * y], [z_sq * z], [x_sq * z], [x_sq * y], [y_sq * x], [y_sq * z], [z_sq * y], [z_sq * x], [x * y * z]])
        prod = self.M @ V
        prod += _scalar_vectorized(E[:, 0, 0], self.M0 @ V)
        prod += _scalar_vectorized(E[:, 1, 0], self.M1 @ V)
        prod += _scalar_vectorized(E[:, 2, 0], self.M2 @ V)
        s = _roll_vectorized(prod, 3 * subtri, axis=0)
        return (dofs @ s)[:, 0, 0]

    def get_function_derivatives(self, alpha, J, ecc, dofs):
        if False:
            print('Hello World!')
        '\n        Parameters\n        ----------\n        *alpha* is a (N x 3 x 1) array (array of column-matrices of\n        barycentric coordinates)\n        *J* is a (N x 2 x 2) array of jacobian matrices (jacobian matrix at\n        triangle first apex)\n        *ecc* is a (N x 3 x 1) array (array of column-matrices of triangle\n        eccentricities)\n        *dofs* is a (N x 1 x 9) arrays (arrays of row-matrices) of computed\n        degrees of freedom.\n\n        Returns\n        -------\n        Returns the values of interpolated function derivatives [dz/dx, dz/dy]\n        in global coordinates at locations alpha, as a column-matrices of\n        shape (N x 2 x 1).\n        '
        subtri = np.argmin(alpha, axis=1)[:, 0]
        ksi = _roll_vectorized(alpha, -subtri, axis=0)
        E = _roll_vectorized(ecc, -subtri, axis=0)
        x = ksi[:, 0, 0]
        y = ksi[:, 1, 0]
        z = ksi[:, 2, 0]
        x_sq = x * x
        y_sq = y * y
        z_sq = z * z
        dV = _to_matrix_vectorized([[-3.0 * x_sq, -3.0 * x_sq], [3.0 * y_sq, 0.0], [0.0, 3.0 * z_sq], [-2.0 * x * z, -2.0 * x * z + x_sq], [-2.0 * x * y + x_sq, -2.0 * x * y], [2.0 * x * y - y_sq, -y_sq], [2.0 * y * z, y_sq], [z_sq, 2.0 * y * z], [-z_sq, 2.0 * x * z - z_sq], [x * z - y * z, x * y - y * z]])
        dV = dV @ _extract_submatrices(self.rotate_dV, subtri, block_size=2, axis=0)
        prod = self.M @ dV
        prod += _scalar_vectorized(E[:, 0, 0], self.M0 @ dV)
        prod += _scalar_vectorized(E[:, 1, 0], self.M1 @ dV)
        prod += _scalar_vectorized(E[:, 2, 0], self.M2 @ dV)
        dsdksi = _roll_vectorized(prod, 3 * subtri, axis=0)
        dfdksi = dofs @ dsdksi
        J_inv = _safe_inv22_vectorized(J)
        dfdx = J_inv @ _transpose_vectorized(dfdksi)
        return dfdx

    def get_function_hessians(self, alpha, J, ecc, dofs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Parameters\n        ----------\n        *alpha* is a (N x 3 x 1) array (array of column-matrices) of\n        barycentric coordinates\n        *J* is a (N x 2 x 2) array of jacobian matrices (jacobian matrix at\n        triangle first apex)\n        *ecc* is a (N x 3 x 1) array (array of column-matrices) of triangle\n        eccentricities\n        *dofs* is a (N x 1 x 9) arrays (arrays of row-matrices) of computed\n        degrees of freedom.\n\n        Returns\n        -------\n        Returns the values of interpolated function 2nd-derivatives\n        [d2z/dx2, d2z/dy2, d2z/dxdy] in global coordinates at locations alpha,\n        as a column-matrices of shape (N x 3 x 1).\n        '
        d2sdksi2 = self.get_d2Sidksij2(alpha, ecc)
        d2fdksi2 = dofs @ d2sdksi2
        H_rot = self.get_Hrot_from_J(J)
        d2fdx2 = d2fdksi2 @ H_rot
        return _transpose_vectorized(d2fdx2)

    def get_d2Sidksij2(self, alpha, ecc):
        if False:
            for i in range(10):
                print('nop')
        '\n        Parameters\n        ----------\n        *alpha* is a (N x 3 x 1) array (array of column-matrices) of\n        barycentric coordinates\n        *ecc* is a (N x 3 x 1) array (array of column-matrices) of triangle\n        eccentricities\n\n        Returns\n        -------\n        Returns the arrays d2sdksi2 (N x 3 x 1) Hessian of shape functions\n        expressed in covariant coordinates in first apex basis.\n        '
        subtri = np.argmin(alpha, axis=1)[:, 0]
        ksi = _roll_vectorized(alpha, -subtri, axis=0)
        E = _roll_vectorized(ecc, -subtri, axis=0)
        x = ksi[:, 0, 0]
        y = ksi[:, 1, 0]
        z = ksi[:, 2, 0]
        d2V = _to_matrix_vectorized([[6.0 * x, 6.0 * x, 6.0 * x], [6.0 * y, 0.0, 0.0], [0.0, 6.0 * z, 0.0], [2.0 * z, 2.0 * z - 4.0 * x, 2.0 * z - 2.0 * x], [2.0 * y - 4.0 * x, 2.0 * y, 2.0 * y - 2.0 * x], [2.0 * x - 4.0 * y, 0.0, -2.0 * y], [2.0 * z, 0.0, 2.0 * y], [0.0, 2.0 * y, 2.0 * z], [0.0, 2.0 * x - 4.0 * z, -2.0 * z], [-2.0 * z, -2.0 * y, x - y - z]])
        d2V = d2V @ _extract_submatrices(self.rotate_d2V, subtri, block_size=3, axis=0)
        prod = self.M @ d2V
        prod += _scalar_vectorized(E[:, 0, 0], self.M0 @ d2V)
        prod += _scalar_vectorized(E[:, 1, 0], self.M1 @ d2V)
        prod += _scalar_vectorized(E[:, 2, 0], self.M2 @ d2V)
        d2sdksi2 = _roll_vectorized(prod, 3 * subtri, axis=0)
        return d2sdksi2

    def get_bending_matrices(self, J, ecc):
        if False:
            i = 10
            return i + 15
        '\n        Parameters\n        ----------\n        *J* is a (N x 2 x 2) array of jacobian matrices (jacobian matrix at\n        triangle first apex)\n        *ecc* is a (N x 3 x 1) array (array of column-matrices) of triangle\n        eccentricities\n\n        Returns\n        -------\n        Returns the element K matrices for bending energy expressed in\n        GLOBAL nodal coordinates.\n        K_ij = integral [ (d2zi/dx2 + d2zi/dy2) * (d2zj/dx2 + d2zj/dy2) dA]\n        tri_J is needed to rotate dofs from local basis to global basis\n        '
        n = np.size(ecc, 0)
        J1 = self.J0_to_J1 @ J
        J2 = self.J0_to_J2 @ J
        DOF_rot = np.zeros([n, 9, 9], dtype=np.float64)
        DOF_rot[:, 0, 0] = 1
        DOF_rot[:, 3, 3] = 1
        DOF_rot[:, 6, 6] = 1
        DOF_rot[:, 1:3, 1:3] = J
        DOF_rot[:, 4:6, 4:6] = J1
        DOF_rot[:, 7:9, 7:9] = J2
        (H_rot, area) = self.get_Hrot_from_J(J, return_area=True)
        K = np.zeros([n, 9, 9], dtype=np.float64)
        weights = self.gauss_w
        pts = self.gauss_pts
        for igauss in range(self.n_gauss):
            alpha = np.tile(pts[igauss, :], n).reshape(n, 3)
            alpha = np.expand_dims(alpha, 2)
            weight = weights[igauss]
            d2Skdksi2 = self.get_d2Sidksij2(alpha, ecc)
            d2Skdx2 = d2Skdksi2 @ H_rot
            K += weight * (d2Skdx2 @ self.E @ _transpose_vectorized(d2Skdx2))
        K = _transpose_vectorized(DOF_rot) @ K @ DOF_rot
        return _scalar_vectorized(area, K)

    def get_Hrot_from_J(self, J, return_area=False):
        if False:
            i = 10
            return i + 15
        '\n        Parameters\n        ----------\n        *J* is a (N x 2 x 2) array of jacobian matrices (jacobian matrix at\n        triangle first apex)\n\n        Returns\n        -------\n        Returns H_rot used to rotate Hessian from local basis of first apex,\n        to global coordinates.\n        if *return_area* is True, returns also the triangle area (0.5*det(J))\n        '
        J_inv = _safe_inv22_vectorized(J)
        Ji00 = J_inv[:, 0, 0]
        Ji11 = J_inv[:, 1, 1]
        Ji10 = J_inv[:, 1, 0]
        Ji01 = J_inv[:, 0, 1]
        H_rot = _to_matrix_vectorized([[Ji00 * Ji00, Ji10 * Ji10, Ji00 * Ji10], [Ji01 * Ji01, Ji11 * Ji11, Ji01 * Ji11], [2 * Ji00 * Ji01, 2 * Ji11 * Ji10, Ji00 * Ji11 + Ji10 * Ji01]])
        if not return_area:
            return H_rot
        else:
            area = 0.5 * (J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0])
            return (H_rot, area)

    def get_Kff_and_Ff(self, J, ecc, triangles, Uc):
        if False:
            i = 10
            return i + 15
        "\n        Build K and F for the following elliptic formulation:\n        minimization of curvature energy with value of function at node\n        imposed and derivatives 'free'.\n\n        Build the global Kff matrix in cco format.\n        Build the full Ff vec Ff = - Kfc x Uc.\n\n        Parameters\n        ----------\n        *J* is a (N x 2 x 2) array of jacobian matrices (jacobian matrix at\n        triangle first apex)\n        *ecc* is a (N x 3 x 1) array (array of column-matrices) of triangle\n        eccentricities\n        *triangles* is a (N x 3) array of nodes indexes.\n        *Uc* is (N x 3) array of imposed displacements at nodes\n\n        Returns\n        -------\n        (Kff_rows, Kff_cols, Kff_vals) Kff matrix in coo format - Duplicate\n        (row, col) entries must be summed.\n        Ff: force vector - dim npts * 3\n        "
        ntri = np.size(ecc, 0)
        vec_range = np.arange(ntri, dtype=np.int32)
        c_indices = np.full(ntri, -1, dtype=np.int32)
        f_dof = [1, 2, 4, 5, 7, 8]
        c_dof = [0, 3, 6]
        f_dof_indices = _to_matrix_vectorized([[c_indices, triangles[:, 0] * 2, triangles[:, 0] * 2 + 1, c_indices, triangles[:, 1] * 2, triangles[:, 1] * 2 + 1, c_indices, triangles[:, 2] * 2, triangles[:, 2] * 2 + 1]])
        expand_indices = np.ones([ntri, 9, 1], dtype=np.int32)
        f_row_indices = _transpose_vectorized(expand_indices @ f_dof_indices)
        f_col_indices = expand_indices @ f_dof_indices
        K_elem = self.get_bending_matrices(J, ecc)
        Kff_vals = np.ravel(K_elem[np.ix_(vec_range, f_dof, f_dof)])
        Kff_rows = np.ravel(f_row_indices[np.ix_(vec_range, f_dof, f_dof)])
        Kff_cols = np.ravel(f_col_indices[np.ix_(vec_range, f_dof, f_dof)])
        Kfc_elem = K_elem[np.ix_(vec_range, f_dof, c_dof)]
        Uc_elem = np.expand_dims(Uc, axis=2)
        Ff_elem = -(Kfc_elem @ Uc_elem)[:, :, 0]
        Ff_indices = f_dof_indices[np.ix_(vec_range, [0], f_dof)][:, 0, :]
        Ff = np.bincount(np.ravel(Ff_indices), weights=np.ravel(Ff_elem))
        return (Kff_rows, Kff_cols, Kff_vals, Ff)

class _DOF_estimator:
    """
    Abstract base class for classes used to estimate a function's first
    derivatives, and deduce the dofs for a CubicTriInterpolator using a
    reduced HCT element formulation.

    Derived classes implement ``compute_df(self, **kwargs)``, returning
    ``np.vstack([dfx, dfy]).T`` where ``dfx, dfy`` are the estimation of the 2
    gradient coordinates.
    """

    def __init__(self, interpolator, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        _api.check_isinstance(CubicTriInterpolator, interpolator=interpolator)
        self._pts = interpolator._pts
        self._tris_pts = interpolator._tris_pts
        self.z = interpolator._z
        self._triangles = interpolator._triangles
        (self._unit_x, self._unit_y) = (interpolator._unit_x, interpolator._unit_y)
        self.dz = self.compute_dz(**kwargs)
        self.compute_dof_from_df()

    def compute_dz(self, **kwargs):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def compute_dof_from_df(self):
        if False:
            while True:
                i = 10
        '\n        Compute reduced-HCT elements degrees of freedom, from the gradient.\n        '
        J = CubicTriInterpolator._get_jacobian(self._tris_pts)
        tri_z = self.z[self._triangles]
        tri_dz = self.dz[self._triangles]
        tri_dof = self.get_dof_vec(tri_z, tri_dz, J)
        return tri_dof

    @staticmethod
    def get_dof_vec(tri_z, tri_dz, J):
        if False:
            for i in range(10):
                print('nop')
        '\n        Compute the dof vector of a triangle, from the value of f, df and\n        of the local Jacobian at each node.\n\n        Parameters\n        ----------\n        tri_z : shape (3,) array\n            f nodal values.\n        tri_dz : shape (3, 2) array\n            df/dx, df/dy nodal values.\n        J\n            Jacobian matrix in local basis of apex 0.\n\n        Returns\n        -------\n        dof : shape (9,) array\n            For each apex ``iapex``::\n\n                dof[iapex*3+0] = f(Ai)\n                dof[iapex*3+1] = df(Ai).(AiAi+)\n                dof[iapex*3+2] = df(Ai).(AiAi-)\n        '
        npt = tri_z.shape[0]
        dof = np.zeros([npt, 9], dtype=np.float64)
        J1 = _ReducedHCT_Element.J0_to_J1 @ J
        J2 = _ReducedHCT_Element.J0_to_J2 @ J
        col0 = J @ np.expand_dims(tri_dz[:, 0, :], axis=2)
        col1 = J1 @ np.expand_dims(tri_dz[:, 1, :], axis=2)
        col2 = J2 @ np.expand_dims(tri_dz[:, 2, :], axis=2)
        dfdksi = _to_matrix_vectorized([[col0[:, 0, 0], col1[:, 0, 0], col2[:, 0, 0]], [col0[:, 1, 0], col1[:, 1, 0], col2[:, 1, 0]]])
        dof[:, 0:7:3] = tri_z
        dof[:, 1:8:3] = dfdksi[:, 0]
        dof[:, 2:9:3] = dfdksi[:, 1]
        return dof

class _DOF_estimator_user(_DOF_estimator):
    """dz is imposed by user; accounts for scaling if any."""

    def compute_dz(self, dz):
        if False:
            print('Hello World!')
        (dzdx, dzdy) = dz
        dzdx = dzdx * self._unit_x
        dzdy = dzdy * self._unit_y
        return np.vstack([dzdx, dzdy]).T

class _DOF_estimator_geom(_DOF_estimator):
    """Fast 'geometric' approximation, recommended for large arrays."""

    def compute_dz(self):
        if False:
            while True:
                i = 10
        '\n        self.df is computed as weighted average of _triangles sharing a common\n        node. On each triangle itri f is first assumed linear (= ~f), which\n        allows to compute d~f[itri]\n        Then the following approximation of df nodal values is then proposed:\n            f[ipt] = SUM ( w[itri] x d~f[itri] , for itri sharing apex ipt)\n        The weighted coeff. w[itri] are proportional to the angle of the\n        triangle itri at apex ipt\n        '
        el_geom_w = self.compute_geom_weights()
        el_geom_grad = self.compute_geom_grads()
        w_node_sum = np.bincount(np.ravel(self._triangles), weights=np.ravel(el_geom_w))
        dfx_el_w = np.empty_like(el_geom_w)
        dfy_el_w = np.empty_like(el_geom_w)
        for iapex in range(3):
            dfx_el_w[:, iapex] = el_geom_w[:, iapex] * el_geom_grad[:, 0]
            dfy_el_w[:, iapex] = el_geom_w[:, iapex] * el_geom_grad[:, 1]
        dfx_node_sum = np.bincount(np.ravel(self._triangles), weights=np.ravel(dfx_el_w))
        dfy_node_sum = np.bincount(np.ravel(self._triangles), weights=np.ravel(dfy_el_w))
        dfx_estim = dfx_node_sum / w_node_sum
        dfy_estim = dfy_node_sum / w_node_sum
        return np.vstack([dfx_estim, dfy_estim]).T

    def compute_geom_weights(self):
        if False:
            i = 10
            return i + 15
        '\n        Build the (nelems, 3) weights coeffs of _triangles angles,\n        renormalized so that np.sum(weights, axis=1) == np.ones(nelems)\n        '
        weights = np.zeros([np.size(self._triangles, 0), 3])
        tris_pts = self._tris_pts
        for ipt in range(3):
            p0 = tris_pts[:, ipt % 3, :]
            p1 = tris_pts[:, (ipt + 1) % 3, :]
            p2 = tris_pts[:, (ipt - 1) % 3, :]
            alpha1 = np.arctan2(p1[:, 1] - p0[:, 1], p1[:, 0] - p0[:, 0])
            alpha2 = np.arctan2(p2[:, 1] - p0[:, 1], p2[:, 0] - p0[:, 0])
            angle = np.abs((alpha2 - alpha1) / np.pi % 1)
            weights[:, ipt] = 0.5 - np.abs(angle - 0.5)
        return weights

    def compute_geom_grads(self):
        if False:
            i = 10
            return i + 15
        '\n        Compute the (global) gradient component of f assumed linear (~f).\n        returns array df of shape (nelems, 2)\n        df[ielem].dM[ielem] = dz[ielem] i.e. df = dz x dM = dM.T^-1 x dz\n        '
        tris_pts = self._tris_pts
        tris_f = self.z[self._triangles]
        dM1 = tris_pts[:, 1, :] - tris_pts[:, 0, :]
        dM2 = tris_pts[:, 2, :] - tris_pts[:, 0, :]
        dM = np.dstack([dM1, dM2])
        dM_inv = _safe_inv22_vectorized(dM)
        dZ1 = tris_f[:, 1] - tris_f[:, 0]
        dZ2 = tris_f[:, 2] - tris_f[:, 0]
        dZ = np.vstack([dZ1, dZ2]).T
        df = np.empty_like(dZ)
        df[:, 0] = dZ[:, 0] * dM_inv[:, 0, 0] + dZ[:, 1] * dM_inv[:, 1, 0]
        df[:, 1] = dZ[:, 0] * dM_inv[:, 0, 1] + dZ[:, 1] * dM_inv[:, 1, 1]
        return df

class _DOF_estimator_min_E(_DOF_estimator_geom):
    """
    The 'smoothest' approximation, df is computed through global minimization
    of the bending energy:
      E(f) = integral[(d2z/dx2 + d2z/dy2 + 2 d2z/dxdy)**2 dA]
    """

    def __init__(self, Interpolator):
        if False:
            i = 10
            return i + 15
        self._eccs = Interpolator._eccs
        super().__init__(Interpolator)

    def compute_dz(self):
        if False:
            return 10
        "\n        Elliptic solver for bending energy minimization.\n        Uses a dedicated 'toy' sparse Jacobi PCG solver.\n        "
        dz_init = super().compute_dz()
        Uf0 = np.ravel(dz_init)
        reference_element = _ReducedHCT_Element()
        J = CubicTriInterpolator._get_jacobian(self._tris_pts)
        eccs = self._eccs
        triangles = self._triangles
        Uc = self.z[self._triangles]
        (Kff_rows, Kff_cols, Kff_vals, Ff) = reference_element.get_Kff_and_Ff(J, eccs, triangles, Uc)
        tol = 1e-10
        n_dof = Ff.shape[0]
        Kff_coo = _Sparse_Matrix_coo(Kff_vals, Kff_rows, Kff_cols, shape=(n_dof, n_dof))
        Kff_coo.compress_csc()
        (Uf, err) = _cg(A=Kff_coo, b=Ff, x0=Uf0, tol=tol)
        err0 = np.linalg.norm(Kff_coo.dot(Uf0) - Ff)
        if err0 < err:
            _api.warn_external('In TriCubicInterpolator initialization, PCG sparse solver did not converge after 1000 iterations. `geom` approximation is used instead of `min_E`')
            Uf = Uf0
        dz = np.empty([self._pts.shape[0], 2], dtype=np.float64)
        dz[:, 0] = Uf[::2]
        dz[:, 1] = Uf[1::2]
        return dz

class _Sparse_Matrix_coo:

    def __init__(self, vals, rows, cols, shape):
        if False:
            i = 10
            return i + 15
        '\n        Create a sparse matrix in coo format.\n        *vals*: arrays of values of non-null entries of the matrix\n        *rows*: int arrays of rows of non-null entries of the matrix\n        *cols*: int arrays of cols of non-null entries of the matrix\n        *shape*: 2-tuple (n, m) of matrix shape\n        '
        (self.n, self.m) = shape
        self.vals = np.asarray(vals, dtype=np.float64)
        self.rows = np.asarray(rows, dtype=np.int32)
        self.cols = np.asarray(cols, dtype=np.int32)

    def dot(self, V):
        if False:
            while True:
                i = 10
        '\n        Dot product of self by a vector *V* in sparse-dense to dense format\n        *V* dense vector of shape (self.m,).\n        '
        assert V.shape == (self.m,)
        return np.bincount(self.rows, weights=self.vals * V[self.cols], minlength=self.m)

    def compress_csc(self):
        if False:
            print('Hello World!')
        '\n        Compress rows, cols, vals / summing duplicates. Sort for csc format.\n        '
        (_, unique, indices) = np.unique(self.rows + self.n * self.cols, return_index=True, return_inverse=True)
        self.rows = self.rows[unique]
        self.cols = self.cols[unique]
        self.vals = np.bincount(indices, weights=self.vals)

    def compress_csr(self):
        if False:
            while True:
                i = 10
        '\n        Compress rows, cols, vals / summing duplicates. Sort for csr format.\n        '
        (_, unique, indices) = np.unique(self.m * self.rows + self.cols, return_index=True, return_inverse=True)
        self.rows = self.rows[unique]
        self.cols = self.cols[unique]
        self.vals = np.bincount(indices, weights=self.vals)

    def to_dense(self):
        if False:
            while True:
                i = 10
        '\n        Return a dense matrix representing self, mainly for debugging purposes.\n        '
        ret = np.zeros([self.n, self.m], dtype=np.float64)
        nvals = self.vals.size
        for i in range(nvals):
            ret[self.rows[i], self.cols[i]] += self.vals[i]
        return ret

    def __str__(self):
        if False:
            while True:
                i = 10
        return self.to_dense().__str__()

    @property
    def diag(self):
        if False:
            print('Hello World!')
        'Return the (dense) vector of the diagonal elements.'
        in_diag = self.rows == self.cols
        diag = np.zeros(min(self.n, self.n), dtype=np.float64)
        diag[self.rows[in_diag]] = self.vals[in_diag]
        return diag

def _cg(A, b, x0=None, tol=1e-10, maxiter=1000):
    if False:
        i = 10
        return i + 15
    '\n    Use Preconditioned Conjugate Gradient iteration to solve A x = b\n    A simple Jacobi (diagonal) preconditioner is used.\n\n    Parameters\n    ----------\n    A : _Sparse_Matrix_coo\n        *A* must have been compressed before by compress_csc or\n        compress_csr method.\n    b : array\n        Right hand side of the linear system.\n    x0 : array, optional\n        Starting guess for the solution. Defaults to the zero vector.\n    tol : float, optional\n        Tolerance to achieve. The algorithm terminates when the relative\n        residual is below tol. Default is 1e-10.\n    maxiter : int, optional\n        Maximum number of iterations.  Iteration will stop after *maxiter*\n        steps even if the specified tolerance has not been achieved. Defaults\n        to 1000.\n\n    Returns\n    -------\n    x : array\n        The converged solution.\n    err : float\n        The absolute error np.linalg.norm(A.dot(x) - b)\n    '
    n = b.size
    assert A.n == n
    assert A.m == n
    b_norm = np.linalg.norm(b)
    kvec = A.diag
    kvec = np.maximum(kvec, 1e-06)
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0
    r = b - A.dot(x)
    w = r / kvec
    p = np.zeros(n)
    beta = 0.0
    rho = np.dot(r, w)
    k = 0
    while np.sqrt(abs(rho)) > tol * b_norm and k < maxiter:
        p = w + beta * p
        z = A.dot(p)
        alpha = rho / np.dot(p, z)
        r = r - alpha * z
        w = r / kvec
        rhoold = rho
        rho = np.dot(r, w)
        x = x + alpha * p
        beta = rho / rhoold
        k += 1
    err = np.linalg.norm(A.dot(x) - b)
    return (x, err)

def _safe_inv22_vectorized(M):
    if False:
        for i in range(10):
            print('nop')
    '\n    Inversion of arrays of (2, 2) matrices, returns 0 for rank-deficient\n    matrices.\n\n    *M* : array of (2, 2) matrices to inverse, shape (n, 2, 2)\n    '
    _api.check_shape((None, 2, 2), M=M)
    M_inv = np.empty_like(M)
    prod1 = M[:, 0, 0] * M[:, 1, 1]
    delta = prod1 - M[:, 0, 1] * M[:, 1, 0]
    rank2 = np.abs(delta) > 1e-08 * np.abs(prod1)
    if np.all(rank2):
        delta_inv = 1.0 / delta
    else:
        delta_inv = np.zeros(M.shape[0])
        delta_inv[rank2] = 1.0 / delta[rank2]
    M_inv[:, 0, 0] = M[:, 1, 1] * delta_inv
    M_inv[:, 0, 1] = -M[:, 0, 1] * delta_inv
    M_inv[:, 1, 0] = -M[:, 1, 0] * delta_inv
    M_inv[:, 1, 1] = M[:, 0, 0] * delta_inv
    return M_inv

def _pseudo_inv22sym_vectorized(M):
    if False:
        while True:
            i = 10
    '\n    Inversion of arrays of (2, 2) SYMMETRIC matrices; returns the\n    (Moore-Penrose) pseudo-inverse for rank-deficient matrices.\n\n    In case M is of rank 1, we have M = trace(M) x P where P is the orthogonal\n    projection on Im(M), and we return trace(M)^-1 x P == M / trace(M)**2\n    In case M is of rank 0, we return the null matrix.\n\n    *M* : array of (2, 2) matrices to inverse, shape (n, 2, 2)\n    '
    _api.check_shape((None, 2, 2), M=M)
    M_inv = np.empty_like(M)
    prod1 = M[:, 0, 0] * M[:, 1, 1]
    delta = prod1 - M[:, 0, 1] * M[:, 1, 0]
    rank2 = np.abs(delta) > 1e-08 * np.abs(prod1)
    if np.all(rank2):
        M_inv[:, 0, 0] = M[:, 1, 1] / delta
        M_inv[:, 0, 1] = -M[:, 0, 1] / delta
        M_inv[:, 1, 0] = -M[:, 1, 0] / delta
        M_inv[:, 1, 1] = M[:, 0, 0] / delta
    else:
        delta = delta[rank2]
        M_inv[rank2, 0, 0] = M[rank2, 1, 1] / delta
        M_inv[rank2, 0, 1] = -M[rank2, 0, 1] / delta
        M_inv[rank2, 1, 0] = -M[rank2, 1, 0] / delta
        M_inv[rank2, 1, 1] = M[rank2, 0, 0] / delta
        rank01 = ~rank2
        tr = M[rank01, 0, 0] + M[rank01, 1, 1]
        tr_zeros = np.abs(tr) < 1e-08
        sq_tr_inv = (1.0 - tr_zeros) / (tr ** 2 + tr_zeros)
        M_inv[rank01, 0, 0] = M[rank01, 0, 0] * sq_tr_inv
        M_inv[rank01, 0, 1] = M[rank01, 0, 1] * sq_tr_inv
        M_inv[rank01, 1, 0] = M[rank01, 1, 0] * sq_tr_inv
        M_inv[rank01, 1, 1] = M[rank01, 1, 1] * sq_tr_inv
    return M_inv

def _scalar_vectorized(scalar, M):
    if False:
        while True:
            i = 10
    '\n    Scalar product between scalars and matrices.\n    '
    return scalar[:, np.newaxis, np.newaxis] * M

def _transpose_vectorized(M):
    if False:
        for i in range(10):
            print('nop')
    '\n    Transposition of an array of matrices *M*.\n    '
    return np.transpose(M, [0, 2, 1])

def _roll_vectorized(M, roll_indices, axis):
    if False:
        return 10
    '\n    Roll an array of matrices along *axis* (0: rows, 1: columns) according to\n    an array of indices *roll_indices*.\n    '
    assert axis in [0, 1]
    ndim = M.ndim
    assert ndim == 3
    ndim_roll = roll_indices.ndim
    assert ndim_roll == 1
    sh = M.shape
    (r, c) = sh[-2:]
    assert sh[0] == roll_indices.shape[0]
    vec_indices = np.arange(sh[0], dtype=np.int32)
    M_roll = np.empty_like(M)
    if axis == 0:
        for ir in range(r):
            for ic in range(c):
                M_roll[:, ir, ic] = M[vec_indices, (-roll_indices + ir) % r, ic]
    else:
        for ir in range(r):
            for ic in range(c):
                M_roll[:, ir, ic] = M[vec_indices, ir, (-roll_indices + ic) % c]
    return M_roll

def _to_matrix_vectorized(M):
    if False:
        print('Hello World!')
    '\n    Build an array of matrices from individuals np.arrays of identical shapes.\n\n    Parameters\n    ----------\n    M\n        ncols-list of nrows-lists of shape sh.\n\n    Returns\n    -------\n    M_res : np.array of shape (sh, nrow, ncols)\n        *M_res* satisfies ``M_res[..., i, j] = M[i][j]``.\n    '
    assert isinstance(M, (tuple, list))
    assert all((isinstance(item, (tuple, list)) for item in M))
    c_vec = np.asarray([len(item) for item in M])
    assert np.all(c_vec - c_vec[0] == 0)
    r = len(M)
    c = c_vec[0]
    M00 = np.asarray(M[0][0])
    dt = M00.dtype
    sh = [M00.shape[0], r, c]
    M_ret = np.empty(sh, dtype=dt)
    for irow in range(r):
        for icol in range(c):
            M_ret[:, irow, icol] = np.asarray(M[irow][icol])
    return M_ret

def _extract_submatrices(M, block_indices, block_size, axis):
    if False:
        while True:
            i = 10
    '\n    Extract selected blocks of a matrices *M* depending on parameters\n    *block_indices* and *block_size*.\n\n    Returns the array of extracted matrices *Mres* so that ::\n\n        M_res[..., ir, :] = M[(block_indices*block_size+ir), :]\n    '
    assert block_indices.ndim == 1
    assert axis in [0, 1]
    (r, c) = M.shape
    if axis == 0:
        sh = [block_indices.shape[0], block_size, c]
    else:
        sh = [block_indices.shape[0], r, block_size]
    dt = M.dtype
    M_res = np.empty(sh, dtype=dt)
    if axis == 0:
        for ir in range(block_size):
            M_res[:, ir, :] = M[block_indices * block_size + ir, :]
    else:
        for ic in range(block_size):
            M_res[:, :, ic] = M[:, block_indices * block_size + ic]
    return M_res