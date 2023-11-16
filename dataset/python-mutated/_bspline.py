import operator
import cupy
from cupy._core import internal
from cupy._core._scalar import get_typename
from cupyx.scipy.sparse import csr_matrix
import numpy as np
TYPES = ['double', 'thrust::complex<double>']
INT_TYPES = ['int', 'long long']
INTERVAL_KERNEL = '\n#include <cupy/complex.cuh>\nextern "C" {\n__global__ void find_interval(\n        const double* t, const double* x, long long* out,\n        int k, int n, bool extrapolate, int total_x) {\n\n    int idx = blockDim.x * blockIdx.x + threadIdx.x;\n    if(idx >= total_x) {\n        return;\n    }\n\n    double xp = *&x[idx];\n    double tb = *&t[k];\n    double te = *&t[n];\n\n    if(isnan(xp)) {\n        out[idx] = -1;\n        return;\n    }\n\n    if((xp < tb || xp > te) && !extrapolate) {\n        out[idx] = -1;\n        return;\n    }\n\n    int left = k;\n    int right = n;\n    int mid;\n    bool found = false;\n\n    while(left < right && !found) {\n        mid = ((right + left) / 2);\n        if(xp > *&t[mid]) {\n            left = mid + 1;\n        } else if (xp < *&t[mid]) {\n            right = mid - 1;\n        } else {\n            found = true;\n        }\n    }\n\n    int default_value = left - 1 < k ? k : left - 1;\n    int result = found ? mid + 1 : default_value + 1;\n\n    while(xp >= *&t[result] && result != n) {\n        result++;\n    }\n\n    out[idx] = result - 1;\n}\n}\n'
INTERVAL_MODULE = cupy.RawModule(code=INTERVAL_KERNEL, options=('-std=c++11',))
D_BOOR_KERNEL = '\n#include <cupy/complex.cuh>\n#include <cupy/math_constants.h>\n#define COMPUTE_LINEAR 0x1\n\ntemplate<typename T>\n__global__ void d_boor(\n        const double* t, const T* c, const int k, const int mu,\n        const double* x, const long long* intervals, T* out,\n        double* temp, int num_c, int mode, int num_x) {\n\n    int idx = blockDim.x * blockIdx.x + threadIdx.x;\n\n    if(idx >= num_x) {\n        return;\n    }\n\n    double xp = *&x[idx];\n    long long interval = *&intervals[idx];\n\n    double* h = temp + idx * (2 * k + 1);\n    double* hh = h + k + 1;\n\n    int ind, j, n;\n    double xa, xb, w;\n\n    if(mode == COMPUTE_LINEAR && interval < 0) {\n        for(j = 0; j < num_c; j++) {\n            out[num_c * idx + j] = CUDART_NAN;\n        }\n        return;\n    }\n\n    /*\n     * Perform k-m "standard" deBoor iterations\n     * so that h contains the k+1 non-zero values of beta_{ell,k-m}(x)\n     * needed to calculate the remaining derivatives.\n     */\n    h[0] = 1.0;\n    for (j = 1; j <= k - mu; j++) {\n        for(int p = 0; p < j; p++) {\n            hh[p] = h[p];\n        }\n        h[0] = 0.0;\n        for (n = 1; n <= j; n++) {\n            ind = interval + n;\n            xb = t[ind];\n            xa = t[ind - j];\n            if (xb == xa) {\n                h[n] = 0.0;\n                continue;\n            }\n            w = hh[n - 1]/(xb - xa);\n            h[n - 1] += w*(xb - xp);\n            h[n] = w*(xp - xa);\n        }\n    }\n\n    /*\n     * Now do m "derivative" recursions\n     * to convert the values of beta into the mth derivative\n     */\n    for (j = k - mu + 1; j <= k; j++) {\n        for(int p = 0; p < j; p++) {\n            hh[p] = h[p];\n        }\n        h[0] = 0.0;\n        for (n = 1; n <= j; n++) {\n            ind = interval + n;\n            xb = t[ind];\n            xa = t[ind - j];\n            if (xb == xa) {\n                h[mu] = 0.0;\n                continue;\n            }\n            w = ((double) j) * hh[n - 1]/(xb - xa);\n            h[n - 1] -= w;\n            h[n] = w;\n        }\n    }\n\n    if(mode != COMPUTE_LINEAR) {\n        return;\n    }\n\n    // Compute linear combinations\n    for(j = 0; j < num_c; j++) {\n        out[num_c * idx + j] = 0;\n        for(n = 0; n < k + 1; n++) {\n            out[num_c * idx + j] = (\n                out[num_c * idx + j] +\n                c[(interval + n - k) * num_c + j] * ((T) h[n]));\n        }\n    }\n\n}\n'
D_BOOR_MODULE = cupy.RawModule(code=D_BOOR_KERNEL, options=('-std=c++11',), name_expressions=[f'd_boor<{type_name}>' for type_name in TYPES])
DESIGN_MAT_KERNEL = '\n#include <cupy/complex.cuh>\n\ntemplate<typename U>\n__global__ void compute_design_matrix(\n        const int k, const long long* intervals, double* bspline_basis,\n        double* data, U* indices, int num_intervals) {\n\n    int idx = blockDim.x * blockIdx.x + threadIdx.x;\n    if(idx >= num_intervals) {\n        return;\n    }\n\n    long long interval = *&intervals[idx];\n\n    double* work = bspline_basis + idx * (2 * k + 1);\n\n    for(int j = 0; j <= k; j++) {\n        int m = (k + 1) * idx + j;\n        data[m] = work[j];\n        indices[m] = (U) (interval - k + j);\n    }\n}\n'
DESIGN_MAT_MODULE = cupy.RawModule(code=DESIGN_MAT_KERNEL, options=('-std=c++11',), name_expressions=[f'compute_design_matrix<{itype}>' for itype in INT_TYPES])

def _get_module_func(module, func_name, *template_args):
    if False:
        for i in range(10):
            print('nop')

    def _get_typename(dtype):
        if False:
            return 10
        typename = get_typename(dtype)
        if dtype.kind == 'c':
            typename = 'thrust::' + typename
        return typename
    args_dtypes = [_get_typename(arg.dtype) for arg in template_args]
    template = ', '.join(args_dtypes)
    kernel_name = f'{func_name}<{template}>' if template_args else func_name
    kernel = module.get_function(kernel_name)
    return kernel

def _get_dtype(dtype):
    if False:
        return 10
    'Return np.complex128 for complex dtypes, np.float64 otherwise.'
    if cupy.issubdtype(dtype, cupy.complexfloating):
        return cupy.complex_
    else:
        return cupy.float_

def _as_float_array(x, check_finite=False):
    if False:
        return 10
    'Convert the input into a C contiguous float array.\n    NB: Upcasts half- and single-precision floats to double precision.\n    '
    x = cupy.ascontiguousarray(x)
    dtyp = _get_dtype(x.dtype)
    x = x.astype(dtyp, copy=False)
    if check_finite and (not cupy.isfinite(x).all()):
        raise ValueError('Array must not contain infs or nans.')
    return x

def _evaluate_spline(t, c, k, xp, nu, extrapolate, out):
    if False:
        return 10
    '\n    Evaluate a spline in the B-spline basis.\n\n    Parameters\n    ----------\n    t : ndarray, shape (n+k+1)\n        knots\n    c : ndarray, shape (n, m)\n        B-spline coefficients\n    xp : ndarray, shape (s,)\n        Points to evaluate the spline at.\n    nu : int\n        Order of derivative to evaluate.\n    extrapolate : int, optional\n        Whether to extrapolate to ouf-of-bounds points, or to return NaNs.\n    out : ndarray, shape (s, m)\n        Computed values of the spline at each of the input points.\n        This argument is modified in-place.\n    '
    n = t.shape[0] - k - 1
    intervals = cupy.empty_like(xp, dtype=cupy.int64)
    interval_kernel = _get_module_func(INTERVAL_MODULE, 'find_interval')
    interval_kernel(((xp.shape[0] + 128 - 1) // 128,), (128,), (t, xp, intervals, k, n, extrapolate, xp.shape[0]))
    num_c = int(np.prod(c.shape[1:]))
    temp = cupy.empty(xp.shape[0] * (2 * k + 1))
    d_boor_kernel = _get_module_func(D_BOOR_MODULE, 'd_boor', c)
    d_boor_kernel(((xp.shape[0] + 128 - 1) // 128,), (128,), (t, c, k, nu, xp, intervals, out, temp, num_c, 1, xp.shape[0]))

def _make_design_matrix(x, t, k, extrapolate, indices):
    if False:
        while True:
            i = 10
    '\n    Returns a design matrix in CSR format.\n    Note that only indices is passed, but not indptr because indptr is already\n    precomputed in the calling Python function design_matrix.\n\n    Parameters\n    ----------\n    x : array_like, shape (n,)\n        Points to evaluate the spline at.\n    t : array_like, shape (nt,)\n        Sorted 1D array of knots.\n    k : int\n        B-spline degree.\n    extrapolate : bool, optional\n        Whether to extrapolate to ouf-of-bounds points.\n    indices : ndarray, shape (n * (k + 1),)\n        Preallocated indices of the final CSR array.\n    Returns\n    -------\n    data\n        The data array of a CSR array of the b-spline design matrix.\n        In each row all the basis elements are evaluated at the certain point\n        (first row - x[0], ..., last row - x[-1]).\n\n    indices\n        The indices array of a CSR array of the b-spline design matrix.\n    '
    n = t.shape[0] - k - 1
    intervals = cupy.empty_like(x, dtype=cupy.int64)
    interval_kernel = _get_module_func(INTERVAL_MODULE, 'find_interval')
    interval_kernel(((x.shape[0] + 128 - 1) // 128,), (128,), (t, x, intervals, k, n, extrapolate, x.shape[0]))
    bspline_basis = cupy.empty(x.shape[0] * (2 * k + 1))
    d_boor_kernel = _get_module_func(D_BOOR_MODULE, 'd_boor', x)
    d_boor_kernel(((x.shape[0] + 128 - 1) // 128,), (128,), (t, None, k, 0, x, intervals, None, bspline_basis, 0, 0, x.shape[0]))
    data = cupy.zeros(x.shape[0] * (k + 1), dtype=cupy.float_)
    design_mat_kernel = _get_module_func(DESIGN_MAT_MODULE, 'compute_design_matrix', indices)
    design_mat_kernel(((x.shape[0] + 128 - 1) // 128,), (128,), (k, intervals, bspline_basis, data, indices, x.shape[0]))
    return (data, indices)

def splder(tck, n=1):
    if False:
        for i in range(10):
            print('nop')
    '\n    Compute the spline representation of the derivative of a given spline\n\n    Parameters\n    ----------\n    tck : tuple of (t, c, k)\n        Spline whose derivative to compute\n    n : int, optional\n        Order of derivative to evaluate. Default: 1\n\n    Returns\n    -------\n    tck_der : tuple of (t2, c2, k2)\n        Spline of order k2=k-n representing the derivative\n        of the input spline.\n\n    Notes\n    -----\n    .. seealso:: :class:`scipy.interpolate.splder`\n\n    See Also\n    --------\n    splantider, splev, spalde\n    '
    if n < 0:
        return splantider(tck, -n)
    (t, c, k) = tck
    if n > k:
        raise ValueError('Order of derivative (n = %r) must be <= order of spline (k = %r)' % (n, tck[2]))
    sh = (slice(None),) + (None,) * len(c.shape[1:])
    try:
        for j in range(n):
            dt = t[k + 1:-1] - t[1:-k - 1]
            dt = dt[sh]
            c = (c[1:-1 - k] - c[:-2 - k]) * k / dt
            c = cupy.r_[c, np.zeros((k,) + c.shape[1:])]
            t = t[1:-1]
            k -= 1
    except FloatingPointError as e:
        raise ValueError('The spline has internal repeated knots and is not differentiable %d times' % n) from e
    return (t, c, k)

def splantider(tck, n=1):
    if False:
        for i in range(10):
            print('nop')
    '\n    Compute the spline for the antiderivative (integral) of a given spline.\n\n    Parameters\n    ----------\n    tck : tuple of (t, c, k)\n        Spline whose antiderivative to compute\n    n : int, optional\n        Order of antiderivative to evaluate. Default: 1\n\n    Returns\n    -------\n    tck_ader : tuple of (t2, c2, k2)\n        Spline of order k2=k+n representing the antiderivative of the input\n        spline.\n\n    See Also\n    --------\n    splder, splev, spalde\n\n    Notes\n    -----\n    The `splder` function is the inverse operation of this function.\n    Namely, ``splder(splantider(tck))`` is identical to `tck`, modulo\n    rounding error.\n\n    .. seealso:: :class:`scipy.interpolate.splantider`\n    '
    if n < 0:
        return splder(tck, -n)
    (t, c, k) = tck
    sh = (slice(None),) + (None,) * len(c.shape[1:])
    for j in range(n):
        dt = t[k + 1:] - t[:-k - 1]
        dt = dt[sh]
        c = cupy.cumsum(c[:-k - 1] * dt, axis=0) / (k + 1)
        c = cupy.r_[cupy.zeros((1,) + c.shape[1:]), c, [c[-1]] * (k + 2)]
        t = cupy.r_[t[0], t, t[-1]]
        k += 1
    return (t, c, k)

class BSpline:
    """Univariate spline in the B-spline basis.

    .. math::
        S(x) = \\sum_{j=0}^{n-1} c_j  B_{j, k; t}(x)

    where :math:`B_{j, k; t}` are B-spline basis functions of degree `k`
    and knots `t`.

    Parameters
    ----------
    t : ndarray, shape (n+k+1,)
        knots
    c : ndarray, shape (>=n, ...)
        spline coefficients
    k : int
        B-spline degree
    extrapolate : bool or 'periodic', optional
        whether to extrapolate beyond the base interval, ``t[k] .. t[n]``,
        or to return nans.
        If True, extrapolates the first and last polynomial pieces of b-spline
        functions active on the base interval.
        If 'periodic', periodic extrapolation is used.
        Default is True.
    axis : int, optional
        Interpolation axis. Default is zero.

    Attributes
    ----------
    t : ndarray
        knot vector
    c : ndarray
        spline coefficients
    k : int
        spline degree
    extrapolate : bool
        If True, extrapolates the first and last polynomial pieces of b-spline
        functions active on the base interval.
    axis : int
        Interpolation axis.
    tck : tuple
        A read-only equivalent of ``(self.t, self.c, self.k)``

    Notes
    -----
    B-spline basis elements are defined via

    .. math::
        B_{i, 0}(x) = 1, \\textrm{if $t_i \\le x < t_{i+1}$, otherwise $0$,}

        B_{i, k}(x) = \\frac{x - t_i}{t_{i+k} - t_i} B_{i, k-1}(x)
                 + \\frac{t_{i+k+1} - x}{t_{i+k+1} - t_{i+1}} B_{i+1, k-1}(x)

    **Implementation details**

    - At least ``k+1`` coefficients are required for a spline of degree `k`,
      so that ``n >= k+1``. Additional coefficients, ``c[j]`` with
      ``j > n``, are ignored.

    - B-spline basis elements of degree `k` form a partition of unity on the
      *base interval*, ``t[k] <= x <= t[n]``.

    - Based on [1]_ and [2]_

    .. seealso:: :class:`scipy.interpolate.BSpline`

    References
    ----------
    .. [1] Tom Lyche and Knut Morken, Spline methods,
        http://www.uio.no/studier/emner/matnat/ifi/INF-MAT5340/v05/undervisningsmateriale/
    .. [2] Carl de Boor, A practical guide to splines, Springer, 2001.
    """

    def __init__(self, t, c, k, extrapolate=True, axis=0):
        if False:
            for i in range(10):
                print('nop')
        self.k = operator.index(k)
        self.c = cupy.asarray(c)
        self.t = cupy.ascontiguousarray(t, dtype=cupy.float64)
        if extrapolate == 'periodic':
            self.extrapolate = extrapolate
        else:
            self.extrapolate = bool(extrapolate)
        n = self.t.shape[0] - self.k - 1
        axis = internal._normalize_axis_index(axis, self.c.ndim)
        self.axis = axis
        if axis != 0:
            self.c = cupy.moveaxis(self.c, axis, 0)
        if k < 0:
            raise ValueError('Spline order cannot be negative.')
        if self.t.ndim != 1:
            raise ValueError('Knot vector must be one-dimensional.')
        if n < self.k + 1:
            raise ValueError('Need at least %d knots for degree %d' % (2 * k + 2, k))
        if (cupy.diff(self.t) < 0).any():
            raise ValueError('Knots must be in a non-decreasing order.')
        if len(cupy.unique(self.t[k:n + 1])) < 2:
            raise ValueError('Need at least two internal knots.')
        if not cupy.isfinite(self.t).all():
            raise ValueError('Knots should not have nans or infs.')
        if self.c.ndim < 1:
            raise ValueError('Coefficients must be at least 1-dimensional.')
        if self.c.shape[0] < n:
            raise ValueError('Knots, coefficients and degree are inconsistent.')
        dt = _get_dtype(self.c.dtype)
        self.c = cupy.ascontiguousarray(self.c, dtype=dt)

    @classmethod
    def construct_fast(cls, t, c, k, extrapolate=True, axis=0):
        if False:
            for i in range(10):
                print('nop')
        'Construct a spline without making checks.\n        Accepts same parameters as the regular constructor. Input arrays\n        `t` and `c` must of correct shape and dtype.\n        '
        self = object.__new__(cls)
        (self.t, self.c, self.k) = (t, c, k)
        self.extrapolate = extrapolate
        self.axis = axis
        return self

    @property
    def tck(self):
        if False:
            i = 10
            return i + 15
        'Equivalent to ``(self.t, self.c, self.k)`` (read-only).\n        '
        return (self.t, self.c, self.k)

    @classmethod
    def basis_element(cls, t, extrapolate=True):
        if False:
            print('Hello World!')
        "Return a B-spline basis element ``B(x | t[0], ..., t[k+1])``.\n\n        Parameters\n        ----------\n        t : ndarray, shape (k+2,)\n            internal knots\n        extrapolate : bool or 'periodic', optional\n            whether to extrapolate beyond the base interval,\n            ``t[0] .. t[k+1]``, or to return nans.\n            If 'periodic', periodic extrapolation is used.\n            Default is True.\n\n        Returns\n        -------\n        basis_element : callable\n            A callable representing a B-spline basis element for the knot\n            vector `t`.\n\n        Notes\n        -----\n        The degree of the B-spline, `k`, is inferred from the length of `t` as\n        ``len(t)-2``. The knot vector is constructed by appending and\n        prepending ``k+1`` elements to internal knots `t`.\n\n        .. seealso:: :class:`scipy.interpolate.BSpline`\n        "
        k = len(t) - 2
        t = _as_float_array(t)
        t = cupy.r_[(t[0] - 1,) * k, t, (t[-1] + 1,) * k]
        c = cupy.zeros_like(t)
        c[k] = 1.0
        return cls.construct_fast(t, c, k, extrapolate)

    @classmethod
    def design_matrix(cls, x, t, k, extrapolate=False):
        if False:
            for i in range(10):
                print('nop')
        "\n        Returns a design matrix as a CSR format sparse array.\n\n        Parameters\n        ----------\n        x : array_like, shape (n,)\n            Points to evaluate the spline at.\n        t : array_like, shape (nt,)\n            Sorted 1D array of knots.\n        k : int\n            B-spline degree.\n        extrapolate : bool or 'periodic', optional\n            Whether to extrapolate based on the first and last intervals\n            or raise an error. If 'periodic', periodic extrapolation is used.\n            Default is False.\n\n        Returns\n        -------\n        design_matrix : `csr_matrix` object\n            Sparse matrix in CSR format where each row contains all the basis\n            elements of the input row (first row = basis elements of x[0],\n            ..., last row = basis elements x[-1]).\n\n        Notes\n        -----\n        In each row of the design matrix all the basis elements are evaluated\n        at the certain point (first row - x[0], ..., last row - x[-1]).\n        `nt` is a length of the vector of knots: as far as there are\n        `nt - k - 1` basis elements, `nt` should be not less than `2 * k + 2`\n        to have at least `k + 1` basis element.\n\n        Out of bounds `x` raises a ValueError.\n\n        .. note::\n            This method returns a `csr_matrix` instance as CuPy still does not\n            have `csr_array`.\n\n        .. seealso:: :class:`scipy.interpolate.BSpline`\n        "
        x = _as_float_array(x, True)
        t = _as_float_array(t, True)
        if extrapolate != 'periodic':
            extrapolate = bool(extrapolate)
        if k < 0:
            raise ValueError('Spline order cannot be negative.')
        if t.ndim != 1 or np.any(t[1:] < t[:-1]):
            raise ValueError(f'Expect t to be a 1-D sorted array_like, but got t={t}.')
        if len(t) < 2 * k + 2:
            raise ValueError(f'Length t is not enough for k={k}.')
        if extrapolate == 'periodic':
            n = t.size - k - 1
            x = t[k] + (x - t[k]) % (t[n] - t[k])
            extrapolate = False
        elif not extrapolate and (min(x) < t[k] or max(x) > t[t.shape[0] - k - 1]):
            raise ValueError(f'Out of bounds w/ x = {x}.')
        n = x.shape[0]
        nnz = n * (k + 1)
        if nnz < cupy.iinfo(cupy.int32).max:
            int_dtype = cupy.int32
        else:
            int_dtype = cupy.int64
        indices = cupy.empty(n * (k + 1), dtype=int_dtype)
        indptr = cupy.arange(0, (n + 1) * (k + 1), k + 1, dtype=int_dtype)
        (data, indices) = _make_design_matrix(x, t, k, extrapolate, indices)
        return csr_matrix((data, indices, indptr), shape=(x.shape[0], t.shape[0] - k - 1))

    def __call__(self, x, nu=0, extrapolate=None):
        if False:
            i = 10
            return i + 15
        "\n        Evaluate a spline function.\n\n        Parameters\n        ----------\n        x : array_like\n            points to evaluate the spline at.\n        nu : int, optional\n            derivative to evaluate (default is 0).\n        extrapolate : bool or 'periodic', optional\n            whether to extrapolate based on the first and last intervals\n            or return nans. If 'periodic', periodic extrapolation is used.\n            Default is `self.extrapolate`.\n\n        Returns\n        -------\n        y : array_like\n            Shape is determined by replacing the interpolation axis\n            in the coefficient array with the shape of `x`.\n        "
        if extrapolate is None:
            extrapolate = self.extrapolate
        x = cupy.asarray(x)
        (x_shape, x_ndim) = (x.shape, x.ndim)
        x = cupy.ascontiguousarray(cupy.ravel(x), dtype=cupy.float_)
        if extrapolate == 'periodic':
            n = self.t.size - self.k - 1
            x = self.t[self.k] + (x - self.t[self.k]) % (self.t[n] - self.t[self.k])
            extrapolate = False
        out = cupy.empty((len(x), int(np.prod(self.c.shape[1:]))), dtype=self.c.dtype)
        self._evaluate(x, nu, extrapolate, out)
        out = out.reshape(x_shape + self.c.shape[1:])
        if self.axis != 0:
            dim_order = list(range(out.ndim))
            dim_order = dim_order[x_ndim:x_ndim + self.axis] + dim_order[:x_ndim] + dim_order[x_ndim + self.axis:]
            out = out.transpose(dim_order)
        return out

    def _ensure_c_contiguous(self):
        if False:
            return 10
        if not self.t.flags.c_contiguous:
            self.t = self.t.copy()
        if not self.c.flags.c_contiguous:
            self.c = self.c.copy()

    def _evaluate(self, xp, nu, extrapolate, out):
        if False:
            print('Hello World!')
        _evaluate_spline(self.t, self.c.reshape(self.c.shape[0], -1), self.k, xp, nu, extrapolate, out)

    def derivative(self, nu=1):
        if False:
            print('Hello World!')
        '\n        Return a B-spline representing the derivative.\n\n        Parameters\n        ----------\n        nu : int, optional\n            Derivative order.\n            Default is 1.\n\n        Returns\n        -------\n        b : BSpline object\n            A new instance representing the derivative.\n\n        See Also\n        --------\n        splder, splantider\n        '
        c = self.c
        ct = len(self.t) - len(c)
        if ct > 0:
            c = cupy.r_[c, cupy.zeros((ct,) + c.shape[1:])]
        tck = splder((self.t, c, self.k), nu)
        return self.construct_fast(*tck, extrapolate=self.extrapolate, axis=self.axis)

    def antiderivative(self, nu=1):
        if False:
            while True:
                i = 10
        "\n        Return a B-spline representing the antiderivative.\n\n        Parameters\n        ----------\n        nu : int, optional\n            Antiderivative order. Default is 1.\n\n        Returns\n        -------\n        b : BSpline object\n            A new instance representing the antiderivative.\n\n        Notes\n        -----\n        If antiderivative is computed and ``self.extrapolate='periodic'``,\n        it will be set to False for the returned instance. This is done because\n        the antiderivative is no longer periodic and its correct evaluation\n        outside of the initially given x interval is difficult.\n\n        See Also\n        --------\n        splder, splantider\n        "
        c = self.c
        ct = len(self.t) - len(c)
        if ct > 0:
            c = cupy.r_[c, cupy.zeros((ct,) + c.shape[1:])]
        tck = splantider((self.t, c, self.k), nu)
        if self.extrapolate == 'periodic':
            extrapolate = False
        else:
            extrapolate = self.extrapolate
        return self.construct_fast(*tck, extrapolate=extrapolate, axis=self.axis)

    def integrate(self, a, b, extrapolate=None):
        if False:
            return 10
        "\n        Compute a definite integral of the spline.\n\n        Parameters\n        ----------\n        a : float\n            Lower limit of integration.\n        b : float\n            Upper limit of integration.\n        extrapolate : bool or 'periodic', optional\n            whether to extrapolate beyond the base interval,\n            ``t[k] .. t[-k-1]``, or take the spline to be zero outside of the\n            base interval. If 'periodic', periodic extrapolation is used.\n            If None (default), use `self.extrapolate`.\n\n        Returns\n        -------\n        I : array_like\n            Definite integral of the spline over the interval ``[a, b]``.\n        "
        if extrapolate is None:
            extrapolate = self.extrapolate
        self._ensure_c_contiguous()
        sign = 1
        if b < a:
            (a, b) = (b, a)
            sign = -1
        n = self.t.size - self.k - 1
        if extrapolate != 'periodic' and (not extrapolate):
            a = max(a, self.t[self.k].item())
            b = min(b, self.t[n].item())
        out = cupy.empty((2, int(np.prod(self.c.shape[1:]))), dtype=self.c.dtype)
        c = self.c
        ct = len(self.t) - len(c)
        if ct > 0:
            c = cupy.r_[c, cupy.zeros((ct,) + c.shape[1:])]
        (ta, ca, ka) = splantider((self.t, c, self.k), 1)
        if extrapolate == 'periodic':
            (ts, te) = (self.t[self.k], self.t[n])
            period = te - ts
            interval = b - a
            (n_periods, left) = divmod(interval, period)
            if n_periods > 0:
                x = cupy.asarray([ts, te], dtype=cupy.float_)
                _evaluate_spline(ta, ca.reshape(ca.shape[0], -1), ka, x, 0, False, out)
                integral = out[1] - out[0]
                integral *= n_periods
            else:
                integral = cupy.zeros((1, int(np.prod(self.c.shape[1:]))), dtype=self.c.dtype)
            a = ts + (a - ts) % period
            b = a + left
            if b <= te:
                x = cupy.asarray([a, b], dtype=cupy.float_)
                _evaluate_spline(ta, ca.reshape(ca.shape[0], -1), ka, x, 0, False, out)
                integral += out[1] - out[0]
            else:
                x = cupy.asarray([a, te], dtype=cupy.float_)
                _evaluate_spline(ta, ca.reshape(ca.shape[0], -1), ka, x, 0, False, out)
                integral += out[1] - out[0]
                x = cupy.asarray([ts, ts + b - te], dtype=cupy.float_)
                _evaluate_spline(ta, ca.reshape(ca.shape[0], -1), ka, x, 0, False, out)
                integral += out[1] - out[0]
        else:
            x = cupy.asarray([a, b], dtype=cupy.float_)
            _evaluate_spline(ta, ca.reshape(ca.shape[0], -1), ka, x, 0, extrapolate, out)
            integral = out[1] - out[0]
        integral *= sign
        return integral.reshape(ca.shape[1:])