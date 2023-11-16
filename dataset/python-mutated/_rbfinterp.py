"""Module for RBF interpolation."""
import math
import warnings
from itertools import combinations_with_replacement
import cupy as cp
kernel_definitions = '\nstatic __device__ double linear(double r)\n{\n     return -r;\n}\n\nstatic __device__ float linear_f(float r)\n{\n    return -r;\n}\n\n\nstatic __device__ double cubic(double r)\n{\n     return r*r*r;\n}\n\nstatic __device__ float cubic_f(float r)\n{\n    return r*r*r;\n}\n\n\nstatic __device__ double thin_plate_spline(double r)\n{\n    if (r == 0.0) {\n        return 0.0;\n    }\n    else {\n        return r*r*log(r);\n    }\n}\n\nstatic __device__ float thin_plate_spline_f(float r)\n{\n    if (r == 0.0) {\n        return 0.0;\n    }\n    else {\n        return r*r*log(r);\n    }\n}\n\n\nstatic __device__ double multiquadric(double r)\n{\n    return -sqrt(r*r + 1);\n}\n\nstatic __device__ float multiquadric_f(float r)\n{\n    return -sqrt(r*r + 1);\n}\n\n\nstatic __device__ double inverse_multiquadric(double r)\n{\n    return 1.0 / sqrt(r*r + 1);\n}\n\nstatic __device__ float inverse_multiquadric_f(float r)\n{\n    return 1.0 / sqrt(r*r + 1);\n}\n\n\nstatic __device__ double inverse_quadratic(double r)\n{\n    return 1.0 / (r*r + 1);\n}\n\nstatic __device__ float inverse_quadrtic_f(float r)\n{\n    return 1.0 / (r*r + 1);\n}\n\n\nstatic __device__ double gaussian(double r)\n{\n    return exp(-r*r);\n}\n\nstatic __device__ float gaussian_f(float r)\n{\n    return exp(-r*r);\n}\n\n\nstatic __device__ double quintic(double r)\n{\n    double r2 = r*r;\n    return -r2*r2*r;\n}\n\nstatic __device__ float qunitic_f(float r)\n{\n    float r2 = r*r;\n    return -r2*r2*r;\n}\n\n'
linear = cp._core.create_ufunc('cupyx_scipy_interpolate_linear', (('f->f', 'out0 = linear_f(in0)'), 'd->d'), 'out0 = linear(in0)', preamble=kernel_definitions, doc='Linear kernel function.\n\n    ``-r``\n    ')
cubic = cp._core.create_ufunc('cupyx_scipy_interpolate_cubic', (('f->f', 'out0 = cubic_f(in0)'), 'd->d'), 'out0 = cubic(in0)', preamble=kernel_definitions, doc='Cubic kernel function.\n\n    ``r**3``\n    ')
thin_plate_spline = cp._core.create_ufunc('cupyx_scipy_interpolate_thin_plate_spline', (('f->f', 'out0 = thin_plate_spline_f(in0)'), 'd->d'), 'out0 = thin_plate_spline(in0)', preamble=kernel_definitions, doc='Thin-plate spline kernel function.\n\n    ``r**2 * log(r) if r != 0 else 0``\n    ')
multiquadric = cp._core.create_ufunc('cupyx_scipy_interpolate_multiquadric', (('f->f', 'out0 = multiquadric_f(in0)'), 'd->d'), 'out0 = multiquadric(in0)', preamble=kernel_definitions, doc='Multiquadric kernel function.\n\n    ``-sqrt(r**2 + 1)``\n    ')
inverse_multiquadric = cp._core.create_ufunc('cupyx_scipy_interpolate_inverse_multiquadric', (('f->f', 'out0 = inverse_multiquadric_f(in0)'), 'd->d'), 'out0 = inverse_multiquadric(in0)', preamble=kernel_definitions, doc='Inverse multiquadric kernel function.\n\n    ``1 / sqrt(r**2 + 1)``\n    ')
inverse_quadratic = cp._core.create_ufunc('cupyx_scipy_interpolate_inverse_quadratic', (('f->f', 'out0 = inverse_quadratic_f(in0)'), 'd->d'), 'out0 = inverse_quadratic(in0)', preamble=kernel_definitions, doc='Inverse quadratic kernel function.\n\n    ``1 / (r**2 + 1)``\n    ')
gaussian = cp._core.create_ufunc('cupyx_scipy_interpolate_gaussian', (('f->f', 'out0 = gaussian_f(in0)'), 'd->d'), 'out0 = gaussian(in0)', preamble=kernel_definitions, doc='Gaussian kernel function.\n\n    ``exp(-r**2)``\n    ')
quintic = cp._core.create_ufunc('cupyx_scipy_interpolate_quintic', (('f->f', 'out0 = quintic_f(in0)'), 'd->d'), 'out0 = quintic(in0)', preamble=kernel_definitions, doc='Quintic kernel function.\n\n    ``-r**5``\n    ')
NAME_TO_FUNC = {'linear': linear, 'thin_plate_spline': thin_plate_spline, 'cubic': cubic, 'quintic': quintic, 'multiquadric': multiquadric, 'inverse_multiquadric': inverse_multiquadric, 'inverse_quadratic': inverse_quadratic, 'gaussian': gaussian}

def kernel_matrix(x, kernel_func, out):
    if False:
        return 10
    'Evaluate RBFs, with centers at `x`, at `x`.'
    delta = x[None, :, :] - x[:, None, :]
    out[...] = kernel_func(cp.linalg.norm(delta, axis=-1))

def polynomial_matrix(x, powers, out):
    if False:
        while True:
            i = 10
    'Evaluate monomials, with exponents from `powers`, at `x`.'
    pwr = x[:, None, :] ** powers[None, :, :]
    cp.prod(pwr, axis=-1, out=out)

def _build_system(y, d, smoothing, kernel, epsilon, powers):
    if False:
        return 10
    'Build the system used to solve for the RBF interpolant coefficients.\n\n    Parameters\n    ----------\n    y : (P, N) float ndarray\n        Data point coordinates.\n    d : (P, S) float ndarray\n        Data values at `y`.\n    smoothing : (P,) float ndarray\n        Smoothing parameter for each data point.\n    kernel : str\n        Name of the RBF.\n    epsilon : float\n        Shape parameter.\n    powers : (R, N) int ndarray\n        The exponents for each monomial in the polynomial.\n\n    Returns\n    -------\n    lhs : (P + R, P + R) float ndarray\n        Left-hand side matrix.\n    rhs : (P + R, S) float ndarray\n        Right-hand side matrix.\n    shift : (N,) float ndarray\n        Domain shift used to create the polynomial matrix.\n    scale : (N,) float ndarray\n        Domain scaling used to create the polynomial matrix.\n\n    '
    p = d.shape[0]
    s = d.shape[1]
    r = powers.shape[0]
    kernel_func = NAME_TO_FUNC[kernel]
    mins = cp.min(y, axis=0)
    maxs = cp.max(y, axis=0)
    shift = (maxs + mins) / 2
    scale = (maxs - mins) / 2
    scale[scale == 0.0] = 1.0
    yeps = y * epsilon
    yhat = (y - shift) / scale
    lhs = cp.empty((p + r, p + r), dtype=float).T
    kernel_matrix(yeps, kernel_func, lhs[:p, :p])
    polynomial_matrix(yhat, powers, lhs[:p, p:])
    lhs[p:, :p] = lhs[:p, p:].T
    lhs[p:, p:] = 0.0
    for i in range(p):
        lhs[i, i] += smoothing[i]
    rhs = cp.empty((s, p + r), dtype=float).T
    rhs[:p] = d
    rhs[p:] = 0.0
    return (lhs, rhs, shift, scale)

def _build_evaluation_coefficients(x, y, kernel, epsilon, powers, shift, scale):
    if False:
        i = 10
        return i + 15
    'Construct the coefficients needed to evaluate\n    the RBF.\n\n    Parameters\n    ----------\n    x : (Q, N) float ndarray\n        Evaluation point coordinates.\n    y : (P, N) float ndarray\n        Data point coordinates.\n    kernel : str\n        Name of the RBF.\n    epsilon : float\n        Shape parameter.\n    powers : (R, N) int ndarray\n        The exponents for each monomial in the polynomial.\n    shift : (N,) float ndarray\n        Shifts the polynomial domain for numerical stability.\n    scale : (N,) float ndarray\n        Scales the polynomial domain for numerical stability.\n\n    Returns\n    -------\n    (Q, P + R) float ndarray\n\n    '
    q = x.shape[0]
    p = y.shape[0]
    r = powers.shape[0]
    kernel_func = NAME_TO_FUNC[kernel]
    yeps = y * epsilon
    xeps = x * epsilon
    xhat = (x - shift) / scale
    vec = cp.empty((q, p + r), dtype=float)
    delta = xeps[:, None, :] - yeps[None, :, :]
    vec[:, :p] = kernel_func(cp.linalg.norm(delta, axis=-1))
    pwr = xhat[:, None, :] ** powers[None, :, :]
    vec[:, p:] = cp.prod(pwr, axis=-1)
    return vec
_AVAILABLE = {'linear', 'thin_plate_spline', 'cubic', 'quintic', 'multiquadric', 'inverse_multiquadric', 'inverse_quadratic', 'gaussian'}
_SCALE_INVARIANT = {'linear', 'thin_plate_spline', 'cubic', 'quintic'}
_NAME_TO_MIN_DEGREE = {'multiquadric': 0, 'linear': 0, 'thin_plate_spline': 1, 'cubic': 1, 'quintic': 2}
try:
    _comb = math.comb
except AttributeError:

    def _comb(n, k):
        if False:
            i = 10
            return i + 15
        return math.factorial(n) // (math.factorial(n - k) * math.factorial(k))

def _monomial_powers(ndim, degree):
    if False:
        print('Hello World!')
    'Return the powers for each monomial in a polynomial.\n\n    Parameters\n    ----------\n    ndim : int\n        Number of variables in the polynomial.\n    degree : int\n        Degree of the polynomial.\n\n    Returns\n    -------\n    (nmonos, ndim) int ndarray\n        Array where each row contains the powers for each variable in a\n        monomial.\n\n    '
    nmonos = _comb(degree + ndim, ndim)
    out = cp.zeros((nmonos, ndim), dtype=int)
    count = 0
    for deg in range(degree + 1):
        for mono in combinations_with_replacement(range(ndim), deg):
            for var in mono:
                out[count, var] += 1
            count += 1
    return out

def _build_and_solve_system(y, d, smoothing, kernel, epsilon, powers):
    if False:
        return 10
    'Build and solve the RBF interpolation system of equations.\n\n    Parameters\n    ----------\n    y : (P, N) float ndarray\n        Data point coordinates.\n    d : (P, S) float ndarray\n        Data values at `y`.\n    smoothing : (P,) float ndarray\n        Smoothing parameter for each data point.\n    kernel : str\n        Name of the RBF.\n    epsilon : float\n        Shape parameter.\n    powers : (R, N) int ndarray\n        The exponents for each monomial in the polynomial.\n\n    Returns\n    -------\n    coeffs : (P + R, S) float ndarray\n        Coefficients for each RBF and monomial.\n    shift : (N,) float ndarray\n        Domain shift used to create the polynomial matrix.\n    scale : (N,) float ndarray\n        Domain scaling used to create the polynomial matrix.\n\n    '
    (lhs, rhs, shift, scale) = _build_system(y, d, smoothing, kernel, epsilon, powers)
    coeffs = cp.linalg.solve(lhs, rhs)
    return (shift, scale, coeffs)

class RBFInterpolator:
    """Radial basis function (RBF) interpolation in N dimensions.

    Parameters
    ----------
    y : (P, N) array_like
        Data point coordinates.
    d : (P, ...) array_like
        Data values at `y`.
    neighbors : int, optional
        If specified, the value of the interpolant at each evaluation point
        will be computed using only this many nearest data points. All the data
        points are used by default.
    smoothing : float or (P,) array_like, optional
        Smoothing parameter. The interpolant perfectly fits the data when this
        is set to 0. For large values, the interpolant approaches a least
        squares fit of a polynomial with the specified degree. Default is 0.
    kernel : str, optional
        Type of RBF. This should be one of

            - 'linear'               : ``-r``
            - 'thin_plate_spline'    : ``r**2 * log(r)``
            - 'cubic'                : ``r**3``
            - 'quintic'              : ``-r**5``
            - 'multiquadric'         : ``-sqrt(1 + r**2)``
            - 'inverse_multiquadric' : ``1/sqrt(1 + r**2)``
            - 'inverse_quadratic'    : ``1/(1 + r**2)``
            - 'gaussian'             : ``exp(-r**2)``

        Default is 'thin_plate_spline'.
    epsilon : float, optional
        Shape parameter that scales the input to the RBF. If `kernel` is
        'linear', 'thin_plate_spline', 'cubic', or 'quintic', this defaults to
        1 and can be ignored because it has the same effect as scaling the
        smoothing parameter. Otherwise, this must be specified.
    degree : int, optional
        Degree of the added polynomial. For some RBFs the interpolant may not
        be well-posed if the polynomial degree is too small. Those RBFs and
        their corresponding minimum degrees are

            - 'multiquadric'      : 0
            - 'linear'            : 0
            - 'thin_plate_spline' : 1
            - 'cubic'             : 1
            - 'quintic'           : 2

        The default value is the minimum degree for `kernel` or 0 if there is
        no minimum degree. Set this to -1 for no added polynomial.

    Notes
    -----
    An RBF is a scalar valued function in N-dimensional space whose value at
    :math:`x` can be expressed in terms of :math:`r=||x - c||`, where :math:`c`
    is the center of the RBF.

    An RBF interpolant for the vector of data values :math:`d`, which are from
    locations :math:`y`, is a linear combination of RBFs centered at :math:`y`
    plus a polynomial with a specified degree. The RBF interpolant is written
    as

    .. math::
        f(x) = K(x, y) a + P(x) b,

    where :math:`K(x, y)` is a matrix of RBFs with centers at :math:`y`
    evaluated at the points :math:`x`, and :math:`P(x)` is a matrix of
    monomials, which span polynomials with the specified degree, evaluated at
    :math:`x`. The coefficients :math:`a` and :math:`b` are the solution to the
    linear equations

    .. math::
        (K(y, y) + \\lambda I) a + P(y) b = d

    and

    .. math::
        P(y)^T a = 0,

    where :math:`\\lambda` is a non-negative smoothing parameter that controls
    how well we want to fit the data. The data are fit exactly when the
    smoothing parameter is 0.

    The above system is uniquely solvable if the following requirements are
    met:

        - :math:`P(y)` must have full column rank. :math:`P(y)` always has full
          column rank when `degree` is -1 or 0. When `degree` is 1,
          :math:`P(y)` has full column rank if the data point locations are not
          all collinear (N=2), coplanar (N=3), etc.
        - If `kernel` is 'multiquadric', 'linear', 'thin_plate_spline',
          'cubic', or 'quintic', then `degree` must not be lower than the
          minimum value listed above.
        - If `smoothing` is 0, then each data point location must be distinct.

    When using an RBF that is not scale invariant ('multiquadric',
    'inverse_multiquadric', 'inverse_quadratic', or 'gaussian'), an appropriate
    shape parameter must be chosen (e.g., through cross validation). Smaller
    values for the shape parameter correspond to wider RBFs. The problem can
    become ill-conditioned or singular when the shape parameter is too small.

    The memory required to solve for the RBF interpolation coefficients
    increases quadratically with the number of data points, which can become
    impractical when interpolating more than about a thousand data points.
    To overcome memory limitations for large interpolation problems, the
    `neighbors` argument can be specified to compute an RBF interpolant for
    each evaluation point using only the nearest data points.

    See Also
    --------
    scipy.interpolate.RBFInterpolator

    """

    def __init__(self, y, d, neighbors=None, smoothing=0.0, kernel='thin_plate_spline', epsilon=None, degree=None):
        if False:
            while True:
                i = 10
        y = cp.asarray(y, dtype=float, order='C')
        if y.ndim != 2:
            raise ValueError('`y` must be a 2-dimensional array.')
        (ny, ndim) = y.shape
        d_dtype = complex if cp.iscomplexobj(d) else float
        d = cp.asarray(d, dtype=d_dtype, order='C')
        if d.shape[0] != ny:
            raise ValueError(f'Expected the first axis of `d` to have length {ny}.')
        d_shape = d.shape[1:]
        d = d.reshape((ny, -1))
        d = d.view(float)
        isscalar = cp.isscalar(smoothing) or smoothing.shape == ()
        if isscalar:
            smoothing = cp.full(ny, smoothing, dtype=float)
        else:
            smoothing = cp.asarray(smoothing, dtype=float, order='C')
            if smoothing.shape != (ny,):
                raise ValueError(f'Expected `smoothing` to be a scalar or have shape ({ny},).')
        kernel = kernel.lower()
        if kernel not in _AVAILABLE:
            raise ValueError(f'`kernel` must be one of {_AVAILABLE}.')
        if epsilon is None:
            if kernel in _SCALE_INVARIANT:
                epsilon = 1.0
            else:
                raise ValueError(f'`epsilon` must be specified if `kernel` is not one of {_SCALE_INVARIANT}.')
        else:
            epsilon = float(epsilon)
        min_degree = _NAME_TO_MIN_DEGREE.get(kernel, -1)
        if degree is None:
            degree = max(min_degree, 0)
        else:
            degree = int(degree)
            if degree < -1:
                raise ValueError('`degree` must be at least -1.')
            elif degree < min_degree:
                warnings.warn(f"`degree` should not be below {min_degree} when `kernel` is '{kernel}'. The interpolant may not be uniquely solvable, and the smoothing parameter may have an unintuitive effect.", UserWarning)
        if neighbors is None:
            nobs = ny
        else:
            raise NotImplementedError('neighbors is not implemented yet')
            neighbors = int(min(neighbors, ny))
            nobs = neighbors
        powers = _monomial_powers(ndim, degree)
        if powers.shape[0] > nobs:
            raise ValueError(f'At least {powers.shape[0]} data points are required when `degree` is {degree} and the number of dimensions is {ndim}.')
        if neighbors is None:
            (shift, scale, coeffs) = _build_and_solve_system(y, d, smoothing, kernel, epsilon, powers)
            self._shift = shift
            self._scale = scale
            self._coeffs = coeffs
        else:
            raise NotImplementedError
        self.y = y
        self.d = d
        self.d_shape = d_shape
        self.d_dtype = d_dtype
        self.neighbors = neighbors
        self.smoothing = smoothing
        self.kernel = kernel
        self.epsilon = epsilon
        self.powers = powers

    def _chunk_evaluator(self, x, y, shift, scale, coeffs, memory_budget=1000000):
        if False:
            print('Hello World!')
        '\n        Evaluate the interpolation.\n\n        Parameters\n        ----------\n        x : (Q, N) float ndarray\n            array of points on which to evaluate\n        y: (P, N) float ndarray\n            array of points on which we know function values\n        shift: (N, ) ndarray\n            Domain shift used to create the polynomial matrix.\n        scale : (N,) float ndarray\n            Domain scaling used to create the polynomial matrix.\n        coeffs: (P+R, S) float ndarray\n            Coefficients in front of basis functions\n\n        Returns\n        -------\n        (Q, S) float ndarray\n        Interpolated array\n        '
        (nx, ndim) = x.shape
        nnei = len(y)
        chunksize = memory_budget // (self.powers.shape[0] + nnei) + 1
        if chunksize <= nx:
            out = cp.empty((nx, self.d.shape[1]), dtype=float)
            for i in range(0, nx, chunksize):
                vec = _build_evaluation_coefficients(x[i:i + chunksize, :], y, self.kernel, self.epsilon, self.powers, shift, scale)
                out[i:i + chunksize, :] = cp.dot(vec, coeffs)
        else:
            vec = _build_evaluation_coefficients(x, y, self.kernel, self.epsilon, self.powers, shift, scale)
            out = cp.dot(vec, coeffs)
        return out

    def __call__(self, x):
        if False:
            return 10
        'Evaluate the interpolant at `x`.\n\n        Parameters\n        ----------\n        x : (Q, N) array_like\n            Evaluation point coordinates.\n\n        Returns\n        -------\n        (Q, ...) ndarray\n            Values of the interpolant at `x`.\n\n        '
        x = cp.asarray(x, dtype=float, order='C')
        if x.ndim != 2:
            raise ValueError('`x` must be a 2-dimensional array.')
        (nx, ndim) = x.shape
        if ndim != self.y.shape[1]:
            raise ValueError(f'Expected the second axis of `x` to have length {self.y.shape[1]}.')
        memory_budget = max(x.size + self.y.size + self.d.size, 1000000)
        if self.neighbors is None:
            out = self._chunk_evaluator(x, self.y, self._shift, self._scale, self._coeffs, memory_budget=memory_budget)
        else:
            raise NotImplementedError
        out = out.view(self.d_dtype)
        out = out.reshape((nx,) + self.d_shape)
        return out