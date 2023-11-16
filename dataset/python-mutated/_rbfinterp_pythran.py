import numpy as np

def linear(r):
    if False:
        return 10
    return -r

def thin_plate_spline(r):
    if False:
        for i in range(10):
            print('nop')
    if r == 0:
        return 0.0
    else:
        return r ** 2 * np.log(r)

def cubic(r):
    if False:
        for i in range(10):
            print('nop')
    return r ** 3

def quintic(r):
    if False:
        while True:
            i = 10
    return -r ** 5

def multiquadric(r):
    if False:
        for i in range(10):
            print('nop')
    return -np.sqrt(r ** 2 + 1)

def inverse_multiquadric(r):
    if False:
        print('Hello World!')
    return 1 / np.sqrt(r ** 2 + 1)

def inverse_quadratic(r):
    if False:
        print('Hello World!')
    return 1 / (r ** 2 + 1)

def gaussian(r):
    if False:
        print('Hello World!')
    return np.exp(-r ** 2)
NAME_TO_FUNC = {'linear': linear, 'thin_plate_spline': thin_plate_spline, 'cubic': cubic, 'quintic': quintic, 'multiquadric': multiquadric, 'inverse_multiquadric': inverse_multiquadric, 'inverse_quadratic': inverse_quadratic, 'gaussian': gaussian}

def kernel_vector(x, y, kernel_func, out):
    if False:
        i = 10
        return i + 15
    'Evaluate RBFs, with centers at `y`, at the point `x`.'
    for i in range(y.shape[0]):
        out[i] = kernel_func(np.linalg.norm(x - y[i]))

def polynomial_vector(x, powers, out):
    if False:
        return 10
    'Evaluate monomials, with exponents from `powers`, at the point `x`.'
    for i in range(powers.shape[0]):
        out[i] = np.prod(x ** powers[i])

def kernel_matrix(x, kernel_func, out):
    if False:
        i = 10
        return i + 15
    'Evaluate RBFs, with centers at `x`, at `x`.'
    for i in range(x.shape[0]):
        for j in range(i + 1):
            out[i, j] = kernel_func(np.linalg.norm(x[i] - x[j]))
            out[j, i] = out[i, j]

def polynomial_matrix(x, powers, out):
    if False:
        return 10
    'Evaluate monomials, with exponents from `powers`, at `x`.'
    for i in range(x.shape[0]):
        for j in range(powers.shape[0]):
            out[i, j] = np.prod(x[i] ** powers[j])

def _kernel_matrix(x, kernel):
    if False:
        return 10
    'Return RBFs, with centers at `x`, evaluated at `x`.'
    out = np.empty((x.shape[0], x.shape[0]), dtype=float)
    kernel_func = NAME_TO_FUNC[kernel]
    kernel_matrix(x, kernel_func, out)
    return out

def _polynomial_matrix(x, powers):
    if False:
        print('Hello World!')
    'Return monomials, with exponents from `powers`, evaluated at `x`.'
    out = np.empty((x.shape[0], powers.shape[0]), dtype=float)
    polynomial_matrix(x, powers, out)
    return out

def _build_system(y, d, smoothing, kernel, epsilon, powers):
    if False:
        return 10
    'Build the system used to solve for the RBF interpolant coefficients.\n\n    Parameters\n    ----------\n    y : (P, N) float ndarray\n        Data point coordinates.\n    d : (P, S) float ndarray\n        Data values at `y`.\n    smoothing : (P,) float ndarray\n        Smoothing parameter for each data point.\n    kernel : str\n        Name of the RBF.\n    epsilon : float\n        Shape parameter.\n    powers : (R, N) int ndarray\n        The exponents for each monomial in the polynomial.\n\n    Returns\n    -------\n    lhs : (P + R, P + R) float ndarray\n        Left-hand side matrix.\n    rhs : (P + R, S) float ndarray\n        Right-hand side matrix.\n    shift : (N,) float ndarray\n        Domain shift used to create the polynomial matrix.\n    scale : (N,) float ndarray\n        Domain scaling used to create the polynomial matrix.\n\n    '
    p = d.shape[0]
    s = d.shape[1]
    r = powers.shape[0]
    kernel_func = NAME_TO_FUNC[kernel]
    mins = np.min(y, axis=0)
    maxs = np.max(y, axis=0)
    shift = (maxs + mins) / 2
    scale = (maxs - mins) / 2
    scale[scale == 0.0] = 1.0
    yeps = y * epsilon
    yhat = (y - shift) / scale
    lhs = np.empty((p + r, p + r), dtype=float).T
    kernel_matrix(yeps, kernel_func, lhs[:p, :p])
    polynomial_matrix(yhat, powers, lhs[:p, p:])
    lhs[p:, :p] = lhs[:p, p:].T
    lhs[p:, p:] = 0.0
    for i in range(p):
        lhs[i, i] += smoothing[i]
    rhs = np.empty((s, p + r), dtype=float).T
    rhs[:p] = d
    rhs[p:] = 0.0
    return (lhs, rhs, shift, scale)

def _build_evaluation_coefficients(x, y, kernel, epsilon, powers, shift, scale):
    if False:
        print('Hello World!')
    'Construct the coefficients needed to evaluate\n    the RBF.\n\n    Parameters\n    ----------\n    x : (Q, N) float ndarray\n        Evaluation point coordinates.\n    y : (P, N) float ndarray\n        Data point coordinates.\n    kernel : str\n        Name of the RBF.\n    epsilon : float\n        Shape parameter.\n    powers : (R, N) int ndarray\n        The exponents for each monomial in the polynomial.\n    shift : (N,) float ndarray\n        Shifts the polynomial domain for numerical stability.\n    scale : (N,) float ndarray\n        Scales the polynomial domain for numerical stability.\n\n    Returns\n    -------\n    (Q, P + R) float ndarray\n\n    '
    q = x.shape[0]
    p = y.shape[0]
    r = powers.shape[0]
    kernel_func = NAME_TO_FUNC[kernel]
    yeps = y * epsilon
    xeps = x * epsilon
    xhat = (x - shift) / scale
    vec = np.empty((q, p + r), dtype=float)
    for i in range(q):
        kernel_vector(xeps[i], yeps, kernel_func, vec[i, :p])
        polynomial_vector(xhat[i], powers, vec[i, p:])
    return vec