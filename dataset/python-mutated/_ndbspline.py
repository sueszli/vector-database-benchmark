import operator
import numpy as np
from math import prod
from . import _bspl
__all__ = ['NdBSpline']

def _get_dtype(dtype):
    if False:
        i = 10
        return i + 15
    'Return np.complex128 for complex dtypes, np.float64 otherwise.'
    if np.issubdtype(dtype, np.complexfloating):
        return np.complex128
    else:
        return np.float64

class NdBSpline:
    """Tensor product spline object.

    The value at point ``xp = (x1, x2, ..., xN)`` is evaluated as a linear
    combination of products of one-dimensional b-splines in each of the ``N``
    dimensions::

       c[i1, i2, ..., iN] * B(x1; i1, t1) * B(x2; i2, t2) * ... * B(xN; iN, tN)


    Here ``B(x; i, t)`` is the ``i``-th b-spline defined by the knot vector
    ``t`` evaluated at ``x``.

    Parameters
    ----------
    t : tuple of 1D ndarrays
        knot vectors in directions 1, 2, ... N,
        ``len(t[i]) == n[i] + k + 1``
    c : ndarray, shape (n1, n2, ..., nN, ...)
        b-spline coefficients
    k : int or length-d tuple of integers
        spline degrees.
        A single integer is interpreted as having this degree for
        all dimensions.
    extrapolate : bool, optional
        Whether to extrapolate out-of-bounds inputs, or return `nan`.
        Default is to extrapolate.

    Attributes
    ----------
    t : tuple of ndarrays
        Knots vectors.
    c : ndarray
        Coefficients of the tensor-produce spline.
    k : tuple of integers
        Degrees for each dimension.
    extrapolate : bool, optional
        Whether to extrapolate or return nans for out-of-bounds inputs.
        Defaults to true.

    Methods
    -------
    __call__

    See Also
    --------
    BSpline : a one-dimensional B-spline object
    NdPPoly : an N-dimensional piecewise tensor product polynomial

    """

    def __init__(self, t, c, k, *, extrapolate=None):
        if False:
            print('Hello World!')
        ndim = len(t)
        try:
            len(k)
        except TypeError:
            k = (k,) * ndim
        if len(k) != ndim:
            raise ValueError(f'len(t) = {len(t)!r} != len(k) = {len(k)!r}.')
        self.k = tuple((operator.index(ki) for ki in k))
        self.t = tuple((np.ascontiguousarray(ti, dtype=float) for ti in t))
        self.c = np.asarray(c)
        if extrapolate is None:
            extrapolate = True
        self.extrapolate = bool(extrapolate)
        self.c = np.asarray(c)
        for d in range(ndim):
            td = self.t[d]
            kd = self.k[d]
            n = td.shape[0] - kd - 1
            if kd < 0:
                raise ValueError(f'Spline degree in dimension {d} cannot be negative.')
            if td.ndim != 1:
                raise ValueError(f'Knot vector in dimension {d} must be one-dimensional.')
            if n < kd + 1:
                raise ValueError(f'Need at least {2 * kd + 2} knots for degree {kd} in dimension {d}.')
            if (np.diff(td) < 0).any():
                raise ValueError(f'Knots in dimension {d} must be in a non-decreasing order.')
            if len(np.unique(td[kd:n + 1])) < 2:
                raise ValueError(f'Need at least two internal knots in dimension {d}.')
            if not np.isfinite(td).all():
                raise ValueError(f'Knots in dimension {d} should not have nans or infs.')
            if self.c.ndim < ndim:
                raise ValueError(f'Coefficients must be at least {d}-dimensional.')
            if self.c.shape[d] != n:
                raise ValueError(f'Knots, coefficients and degree in dimension {d} are inconsistent: got {self.c.shape[d]} coefficients for {len(td)} knots, need at least {n} for k={k}.')
        dt = _get_dtype(self.c.dtype)
        self.c = np.ascontiguousarray(self.c, dtype=dt)

    def __call__(self, xi, *, nu=None, extrapolate=None):
        if False:
            return 10
        'Evaluate the tensor product b-spline at ``xi``.\n\n        Parameters\n        ----------\n        xi : array_like, shape(..., ndim)\n            The coordinates to evaluate the interpolator at.\n            This can be a list or tuple of ndim-dimensional points\n            or an array with the shape (num_points, ndim).\n        nu : array_like, optional, shape (ndim,)\n            Orders of derivatives to evaluate. Each must be non-negative. Defaults to the zeroth derivivative.\n        extrapolate : bool, optional\n            Whether to exrapolate based on first and last intervals in each\n            dimension, or return `nan`. Default is to ``self.extrapolate``.\n\n        Returns\n        -------\n        values : ndarray, shape ``xi.shape[:-1] + self.c.shape[ndim:]``\n            Interpolated values at ``xi``\n        '
        ndim = len(self.t)
        if extrapolate is None:
            extrapolate = self.extrapolate
        extrapolate = bool(extrapolate)
        if nu is None:
            nu = np.zeros((ndim,), dtype=np.intc)
        else:
            nu = np.asarray(nu, dtype=np.intc)
            if nu.ndim != 1 or nu.shape[0] != ndim:
                raise ValueError('invalid number of derivative orders nu')
        xi = np.asarray(xi, dtype=float)
        xi_shape = xi.shape
        xi = xi.reshape(-1, xi_shape[-1])
        xi = np.ascontiguousarray(xi)
        if xi_shape[-1] != ndim:
            raise ValueError(f'Shapes: xi.shape={xi_shape} and ndim={ndim}')
        _k = np.asarray(self.k)
        len_t = [len(ti) for ti in self.t]
        _t = np.empty((ndim, max(len_t)), dtype=float)
        _t.fill(np.nan)
        for d in range(ndim):
            _t[d, :len(self.t[d])] = self.t[d]
        shape = tuple((kd + 1 for kd in self.k))
        indices = np.unravel_index(np.arange(prod(shape)), shape)
        _indices_k1d = np.asarray(indices, dtype=np.intp).T
        c1 = self.c.reshape(self.c.shape[:ndim] + (-1,))
        c1r = c1.ravel()
        _strides_c1 = np.asarray([s // c1.dtype.itemsize for s in c1.strides], dtype=np.intp)
        num_c_tr = c1.shape[-1]
        out = np.empty(xi.shape[:-1] + (num_c_tr,), dtype=c1.dtype)
        _bspl.evaluate_ndbspline(xi, _t, _k, nu, extrapolate, c1r, num_c_tr, _strides_c1, _indices_k1d, out)
        return out.reshape(xi_shape[:-1] + self.c.shape[ndim:])