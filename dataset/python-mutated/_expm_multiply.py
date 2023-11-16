"""Compute the action of the matrix exponential."""
from warnings import warn
import numpy as np
import scipy.linalg
import scipy.sparse.linalg
from scipy.linalg._decomp_qr import qr
from scipy.sparse._sputils import is_pydata_spmatrix
from scipy.sparse.linalg import aslinearoperator
from scipy.sparse.linalg._interface import IdentityOperator
from scipy.sparse.linalg._onenormest import onenormest
__all__ = ['expm_multiply']

def _exact_inf_norm(A):
    if False:
        while True:
            i = 10
    if scipy.sparse.issparse(A):
        return max(abs(A).sum(axis=1).flat)
    elif is_pydata_spmatrix(A):
        return max(abs(A).sum(axis=1))
    else:
        return np.linalg.norm(A, np.inf)

def _exact_1_norm(A):
    if False:
        return 10
    if scipy.sparse.issparse(A):
        return max(abs(A).sum(axis=0).flat)
    elif is_pydata_spmatrix(A):
        return max(abs(A).sum(axis=0))
    else:
        return np.linalg.norm(A, 1)

def _trace(A):
    if False:
        for i in range(10):
            print('nop')
    if is_pydata_spmatrix(A):
        return A.to_scipy_sparse().trace()
    else:
        return A.trace()

def traceest(A, m3, seed=None):
    if False:
        for i in range(10):
            print('nop')
    'Estimate `np.trace(A)` using `3*m3` matrix-vector products.\n\n    The result is not deterministic.\n\n    Parameters\n    ----------\n    A : LinearOperator\n        Linear operator whose trace will be estimated. Has to be square.\n    m3 : int\n        Number of matrix-vector products divided by 3 used to estimate the\n        trace.\n    seed : optional\n        Seed for `numpy.random.default_rng`.\n        Can be provided to obtain deterministic results.\n\n    Returns\n    -------\n    trace : LinearOperator.dtype\n        Estimate of the trace\n\n    Notes\n    -----\n    This is the Hutch++ algorithm given in [1]_.\n\n    References\n    ----------\n    .. [1] Meyer, Raphael A., Cameron Musco, Christopher Musco, and David P.\n       Woodruff. "Hutch++: Optimal Stochastic Trace Estimation." In Symposium\n       on Simplicity in Algorithms (SOSA), pp. 142-155. Society for Industrial\n       and Applied Mathematics, 2021\n       https://doi.org/10.1137/1.9781611976496.16\n\n    '
    rng = np.random.default_rng(seed)
    if len(A.shape) != 2 or A.shape[-1] != A.shape[-2]:
        raise ValueError('Expected A to be like a square matrix.')
    n = A.shape[-1]
    S = rng.choice([-1.0, +1.0], [n, m3])
    (Q, _) = qr(A.matmat(S), overwrite_a=True, mode='economic')
    trQAQ = np.trace(Q.conj().T @ A.matmat(Q))
    G = rng.choice([-1, +1], [n, m3])
    right = G - Q @ (Q.conj().T @ G)
    trGAG = np.trace(right.conj().T @ A.matmat(right))
    return trQAQ + trGAG / m3

def _ident_like(A):
    if False:
        i = 10
        return i + 15
    if scipy.sparse.issparse(A):
        out = scipy.sparse.eye(A.shape[0], A.shape[1], dtype=A.dtype)
        if isinstance(A, scipy.sparse.spmatrix):
            return out.asformat(A.format)
        return scipy.sparse.dia_array(out).asformat(A.format)
    elif is_pydata_spmatrix(A):
        import sparse
        return sparse.eye(A.shape[0], A.shape[1], dtype=A.dtype)
    elif isinstance(A, scipy.sparse.linalg.LinearOperator):
        return IdentityOperator(A.shape, dtype=A.dtype)
    else:
        return np.eye(A.shape[0], A.shape[1], dtype=A.dtype)

def expm_multiply(A, B, start=None, stop=None, num=None, endpoint=None, traceA=None):
    if False:
        i = 10
        return i + 15
    '\n    Compute the action of the matrix exponential of A on B.\n\n    Parameters\n    ----------\n    A : transposable linear operator\n        The operator whose exponential is of interest.\n    B : ndarray\n        The matrix or vector to be multiplied by the matrix exponential of A.\n    start : scalar, optional\n        The starting time point of the sequence.\n    stop : scalar, optional\n        The end time point of the sequence, unless `endpoint` is set to False.\n        In that case, the sequence consists of all but the last of ``num + 1``\n        evenly spaced time points, so that `stop` is excluded.\n        Note that the step size changes when `endpoint` is False.\n    num : int, optional\n        Number of time points to use.\n    endpoint : bool, optional\n        If True, `stop` is the last time point.  Otherwise, it is not included.\n    traceA : scalar, optional\n        Trace of `A`. If not given the trace is estimated for linear operators,\n        or calculated exactly for sparse matrices. It is used to precondition\n        `A`, thus an approximate trace is acceptable.\n        For linear operators, `traceA` should be provided to ensure performance\n        as the estimation is not guaranteed to be reliable for all cases.\n\n        .. versionadded:: 1.9.0\n\n    Returns\n    -------\n    expm_A_B : ndarray\n         The result of the action :math:`e^{t_k A} B`.\n\n    Warns\n    -----\n    UserWarning\n        If `A` is a linear operator and ``traceA=None`` (default).\n\n    Notes\n    -----\n    The optional arguments defining the sequence of evenly spaced time points\n    are compatible with the arguments of `numpy.linspace`.\n\n    The output ndarray shape is somewhat complicated so I explain it here.\n    The ndim of the output could be either 1, 2, or 3.\n    It would be 1 if you are computing the expm action on a single vector\n    at a single time point.\n    It would be 2 if you are computing the expm action on a vector\n    at multiple time points, or if you are computing the expm action\n    on a matrix at a single time point.\n    It would be 3 if you want the action on a matrix with multiple\n    columns at multiple time points.\n    If multiple time points are requested, expm_A_B[0] will always\n    be the action of the expm at the first time point,\n    regardless of whether the action is on a vector or a matrix.\n\n    References\n    ----------\n    .. [1] Awad H. Al-Mohy and Nicholas J. Higham (2011)\n           "Computing the Action of the Matrix Exponential,\n           with an Application to Exponential Integrators."\n           SIAM Journal on Scientific Computing,\n           33 (2). pp. 488-511. ISSN 1064-8275\n           http://eprints.ma.man.ac.uk/1591/\n\n    .. [2] Nicholas J. Higham and Awad H. Al-Mohy (2010)\n           "Computing Matrix Functions."\n           Acta Numerica,\n           19. 159-208. ISSN 0962-4929\n           http://eprints.ma.man.ac.uk/1451/\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy.sparse import csc_matrix\n    >>> from scipy.sparse.linalg import expm, expm_multiply\n    >>> A = csc_matrix([[1, 0], [0, 1]])\n    >>> A.toarray()\n    array([[1, 0],\n           [0, 1]], dtype=int64)\n    >>> B = np.array([np.exp(-1.), np.exp(-2.)])\n    >>> B\n    array([ 0.36787944,  0.13533528])\n    >>> expm_multiply(A, B, start=1, stop=2, num=3, endpoint=True)\n    array([[ 1.        ,  0.36787944],\n           [ 1.64872127,  0.60653066],\n           [ 2.71828183,  1.        ]])\n    >>> expm(A).dot(B)                  # Verify 1st timestep\n    array([ 1.        ,  0.36787944])\n    >>> expm(1.5*A).dot(B)              # Verify 2nd timestep\n    array([ 1.64872127,  0.60653066])\n    >>> expm(2*A).dot(B)                # Verify 3rd timestep\n    array([ 2.71828183,  1.        ])\n    '
    if all((arg is None for arg in (start, stop, num, endpoint))):
        X = _expm_multiply_simple(A, B, traceA=traceA)
    else:
        (X, status) = _expm_multiply_interval(A, B, start, stop, num, endpoint, traceA=traceA)
    return X

def _expm_multiply_simple(A, B, t=1.0, traceA=None, balance=False):
    if False:
        return 10
    '\n    Compute the action of the matrix exponential at a single time point.\n\n    Parameters\n    ----------\n    A : transposable linear operator\n        The operator whose exponential is of interest.\n    B : ndarray\n        The matrix to be multiplied by the matrix exponential of A.\n    t : float\n        A time point.\n    traceA : scalar, optional\n        Trace of `A`. If not given the trace is estimated for linear operators,\n        or calculated exactly for sparse matrices. It is used to precondition\n        `A`, thus an approximate trace is acceptable\n    balance : bool\n        Indicates whether or not to apply balancing.\n\n    Returns\n    -------\n    F : ndarray\n        :math:`e^{t A} B`\n\n    Notes\n    -----\n    This is algorithm (3.2) in Al-Mohy and Higham (2011).\n\n    '
    if balance:
        raise NotImplementedError
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected A to be like a square matrix')
    if A.shape[1] != B.shape[0]:
        raise ValueError('shapes of matrices A {} and B {} are incompatible'.format(A.shape, B.shape))
    ident = _ident_like(A)
    is_linear_operator = isinstance(A, scipy.sparse.linalg.LinearOperator)
    n = A.shape[0]
    if len(B.shape) == 1:
        n0 = 1
    elif len(B.shape) == 2:
        n0 = B.shape[1]
    else:
        raise ValueError('expected B to be like a matrix or a vector')
    u_d = 2 ** (-53)
    tol = u_d
    if traceA is None:
        if is_linear_operator:
            warn('Trace of LinearOperator not available, it will be estimated. Provide `traceA` to ensure performance.', stacklevel=3)
        traceA = traceest(A, m3=1) if is_linear_operator else _trace(A)
    mu = traceA / float(n)
    A = A - mu * ident
    A_1_norm = onenormest(A) if is_linear_operator else _exact_1_norm(A)
    if t * A_1_norm == 0:
        (m_star, s) = (0, 1)
    else:
        ell = 2
        norm_info = LazyOperatorNormInfo(t * A, A_1_norm=t * A_1_norm, ell=ell)
        (m_star, s) = _fragment_3_1(norm_info, n0, tol, ell=ell)
    return _expm_multiply_simple_core(A, B, t, mu, m_star, s, tol, balance)

def _expm_multiply_simple_core(A, B, t, mu, m_star, s, tol=None, balance=False):
    if False:
        while True:
            i = 10
    '\n    A helper function.\n    '
    if balance:
        raise NotImplementedError
    if tol is None:
        u_d = 2 ** (-53)
        tol = u_d
    F = B
    eta = np.exp(t * mu / float(s))
    for i in range(s):
        c1 = _exact_inf_norm(B)
        for j in range(m_star):
            coeff = t / float(s * (j + 1))
            B = coeff * A.dot(B)
            c2 = _exact_inf_norm(B)
            F = F + B
            if c1 + c2 <= tol * _exact_inf_norm(F):
                break
            c1 = c2
        F = eta * F
        B = F
    return F
_theta = {1: 2.29e-16, 2: 2.58e-08, 3: 1.39e-05, 4: 0.00034, 5: 0.0024, 6: 0.00907, 7: 0.0238, 8: 0.05, 9: 0.0896, 10: 0.144, 11: 0.214, 12: 0.3, 13: 0.4, 14: 0.514, 15: 0.641, 16: 0.781, 17: 0.931, 18: 1.09, 19: 1.26, 20: 1.44, 21: 1.62, 22: 1.82, 23: 2.01, 24: 2.22, 25: 2.43, 26: 2.64, 27: 2.86, 28: 3.08, 29: 3.31, 30: 3.54, 35: 4.7, 40: 6.0, 45: 7.2, 50: 8.5, 55: 9.9}

def _onenormest_matrix_power(A, p, t=2, itmax=5, compute_v=False, compute_w=False):
    if False:
        while True:
            i = 10
    '\n    Efficiently estimate the 1-norm of A^p.\n\n    Parameters\n    ----------\n    A : ndarray\n        Matrix whose 1-norm of a power is to be computed.\n    p : int\n        Non-negative integer power.\n    t : int, optional\n        A positive parameter controlling the tradeoff between\n        accuracy versus time and memory usage.\n        Larger values take longer and use more memory\n        but give more accurate output.\n    itmax : int, optional\n        Use at most this many iterations.\n    compute_v : bool, optional\n        Request a norm-maximizing linear operator input vector if True.\n    compute_w : bool, optional\n        Request a norm-maximizing linear operator output vector if True.\n\n    Returns\n    -------\n    est : float\n        An underestimate of the 1-norm of the sparse matrix.\n    v : ndarray, optional\n        The vector such that ||Av||_1 == est*||v||_1.\n        It can be thought of as an input to the linear operator\n        that gives an output with particularly large norm.\n    w : ndarray, optional\n        The vector Av which has relatively large 1-norm.\n        It can be thought of as an output of the linear operator\n        that is relatively large in norm compared to the input.\n\n    '
    from scipy.sparse.linalg._onenormest import onenormest
    return onenormest(aslinearoperator(A) ** p)

class LazyOperatorNormInfo:
    """
    Information about an operator is lazily computed.

    The information includes the exact 1-norm of the operator,
    in addition to estimates of 1-norms of powers of the operator.
    This uses the notation of Computing the Action (2011).
    This class is specialized enough to probably not be of general interest
    outside of this module.

    """

    def __init__(self, A, A_1_norm=None, ell=2, scale=1):
        if False:
            for i in range(10):
                print('nop')
        '\n        Provide the operator and some norm-related information.\n\n        Parameters\n        ----------\n        A : linear operator\n            The operator of interest.\n        A_1_norm : float, optional\n            The exact 1-norm of A.\n        ell : int, optional\n            A technical parameter controlling norm estimation quality.\n        scale : int, optional\n            If specified, return the norms of scale*A instead of A.\n\n        '
        self._A = A
        self._A_1_norm = A_1_norm
        self._ell = ell
        self._d = {}
        self._scale = scale

    def set_scale(self, scale):
        if False:
            i = 10
            return i + 15
        '\n        Set the scale parameter.\n        '
        self._scale = scale

    def onenorm(self):
        if False:
            i = 10
            return i + 15
        '\n        Compute the exact 1-norm.\n        '
        if self._A_1_norm is None:
            self._A_1_norm = _exact_1_norm(self._A)
        return self._scale * self._A_1_norm

    def d(self, p):
        if False:
            print('Hello World!')
        '\n        Lazily estimate d_p(A) ~= || A^p ||^(1/p) where ||.|| is the 1-norm.\n        '
        if p not in self._d:
            est = _onenormest_matrix_power(self._A, p, self._ell)
            self._d[p] = est ** (1.0 / p)
        return self._scale * self._d[p]

    def alpha(self, p):
        if False:
            while True:
                i = 10
        '\n        Lazily compute max(d(p), d(p+1)).\n        '
        return max(self.d(p), self.d(p + 1))

def _compute_cost_div_m(m, p, norm_info):
    if False:
        print('Hello World!')
    '\n    A helper function for computing bounds.\n\n    This is equation (3.10).\n    It measures cost in terms of the number of required matrix products.\n\n    Parameters\n    ----------\n    m : int\n        A valid key of _theta.\n    p : int\n        A matrix power.\n    norm_info : LazyOperatorNormInfo\n        Information about 1-norms of related operators.\n\n    Returns\n    -------\n    cost_div_m : int\n        Required number of matrix products divided by m.\n\n    '
    return int(np.ceil(norm_info.alpha(p) / _theta[m]))

def _compute_p_max(m_max):
    if False:
        print('Hello World!')
    '\n    Compute the largest positive integer p such that p*(p-1) <= m_max + 1.\n\n    Do this in a slightly dumb way, but safe and not too slow.\n\n    Parameters\n    ----------\n    m_max : int\n        A count related to bounds.\n\n    '
    sqrt_m_max = np.sqrt(m_max)
    p_low = int(np.floor(sqrt_m_max))
    p_high = int(np.ceil(sqrt_m_max + 1))
    return max((p for p in range(p_low, p_high + 1) if p * (p - 1) <= m_max + 1))

def _fragment_3_1(norm_info, n0, tol, m_max=55, ell=2):
    if False:
        return 10
    '\n    A helper function for the _expm_multiply_* functions.\n\n    Parameters\n    ----------\n    norm_info : LazyOperatorNormInfo\n        Information about norms of certain linear operators of interest.\n    n0 : int\n        Number of columns in the _expm_multiply_* B matrix.\n    tol : float\n        Expected to be\n        :math:`2^{-24}` for single precision or\n        :math:`2^{-53}` for double precision.\n    m_max : int\n        A value related to a bound.\n    ell : int\n        The number of columns used in the 1-norm approximation.\n        This is usually taken to be small, maybe between 1 and 5.\n\n    Returns\n    -------\n    best_m : int\n        Related to bounds for error control.\n    best_s : int\n        Amount of scaling.\n\n    Notes\n    -----\n    This is code fragment (3.1) in Al-Mohy and Higham (2011).\n    The discussion of default values for m_max and ell\n    is given between the definitions of equation (3.11)\n    and the definition of equation (3.12).\n\n    '
    if ell < 1:
        raise ValueError('expected ell to be a positive integer')
    best_m = None
    best_s = None
    if _condition_3_13(norm_info.onenorm(), n0, m_max, ell):
        for (m, theta) in _theta.items():
            s = int(np.ceil(norm_info.onenorm() / theta))
            if best_m is None or m * s < best_m * best_s:
                best_m = m
                best_s = s
    else:
        for p in range(2, _compute_p_max(m_max) + 1):
            for m in range(p * (p - 1) - 1, m_max + 1):
                if m in _theta:
                    s = _compute_cost_div_m(m, p, norm_info)
                    if best_m is None or m * s < best_m * best_s:
                        best_m = m
                        best_s = s
        best_s = max(best_s, 1)
    return (best_m, best_s)

def _condition_3_13(A_1_norm, n0, m_max, ell):
    if False:
        for i in range(10):
            print('nop')
    '\n    A helper function for the _expm_multiply_* functions.\n\n    Parameters\n    ----------\n    A_1_norm : float\n        The precomputed 1-norm of A.\n    n0 : int\n        Number of columns in the _expm_multiply_* B matrix.\n    m_max : int\n        A value related to a bound.\n    ell : int\n        The number of columns used in the 1-norm approximation.\n        This is usually taken to be small, maybe between 1 and 5.\n\n    Returns\n    -------\n    value : bool\n        Indicates whether or not the condition has been met.\n\n    Notes\n    -----\n    This is condition (3.13) in Al-Mohy and Higham (2011).\n\n    '
    p_max = _compute_p_max(m_max)
    a = 2 * ell * p_max * (p_max + 3)
    b = _theta[m_max] / float(n0 * m_max)
    return A_1_norm <= a * b

def _expm_multiply_interval(A, B, start=None, stop=None, num=None, endpoint=None, traceA=None, balance=False, status_only=False):
    if False:
        return 10
    '\n    Compute the action of the matrix exponential at multiple time points.\n\n    Parameters\n    ----------\n    A : transposable linear operator\n        The operator whose exponential is of interest.\n    B : ndarray\n        The matrix to be multiplied by the matrix exponential of A.\n    start : scalar, optional\n        The starting time point of the sequence.\n    stop : scalar, optional\n        The end time point of the sequence, unless `endpoint` is set to False.\n        In that case, the sequence consists of all but the last of ``num + 1``\n        evenly spaced time points, so that `stop` is excluded.\n        Note that the step size changes when `endpoint` is False.\n    num : int, optional\n        Number of time points to use.\n    traceA : scalar, optional\n        Trace of `A`. If not given the trace is estimated for linear operators,\n        or calculated exactly for sparse matrices. It is used to precondition\n        `A`, thus an approximate trace is acceptable\n    endpoint : bool, optional\n        If True, `stop` is the last time point. Otherwise, it is not included.\n    balance : bool\n        Indicates whether or not to apply balancing.\n    status_only : bool\n        A flag that is set to True for some debugging and testing operations.\n\n    Returns\n    -------\n    F : ndarray\n        :math:`e^{t_k A} B`\n    status : int\n        An integer status for testing and debugging.\n\n    Notes\n    -----\n    This is algorithm (5.2) in Al-Mohy and Higham (2011).\n\n    There seems to be a typo, where line 15 of the algorithm should be\n    moved to line 6.5 (between lines 6 and 7).\n\n    '
    if balance:
        raise NotImplementedError
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected A to be like a square matrix')
    if A.shape[1] != B.shape[0]:
        raise ValueError('shapes of matrices A {} and B {} are incompatible'.format(A.shape, B.shape))
    ident = _ident_like(A)
    is_linear_operator = isinstance(A, scipy.sparse.linalg.LinearOperator)
    n = A.shape[0]
    if len(B.shape) == 1:
        n0 = 1
    elif len(B.shape) == 2:
        n0 = B.shape[1]
    else:
        raise ValueError('expected B to be like a matrix or a vector')
    u_d = 2 ** (-53)
    tol = u_d
    if traceA is None:
        if is_linear_operator:
            warn('Trace of LinearOperator not available, it will be estimated. Provide `traceA` to ensure performance.', stacklevel=3)
        traceA = traceest(A, m3=5) if is_linear_operator else _trace(A)
    mu = traceA / float(n)
    linspace_kwargs = {'retstep': True}
    if num is not None:
        linspace_kwargs['num'] = num
    if endpoint is not None:
        linspace_kwargs['endpoint'] = endpoint
    (samples, step) = np.linspace(start, stop, **linspace_kwargs)
    nsamples = len(samples)
    if nsamples < 2:
        raise ValueError('at least two time points are required')
    q = nsamples - 1
    h = step
    t_0 = samples[0]
    t_q = samples[q]
    X_shape = (nsamples,) + B.shape
    X = np.empty(X_shape, dtype=np.result_type(A.dtype, B.dtype, float))
    t = t_q - t_0
    A = A - mu * ident
    A_1_norm = onenormest(A) if is_linear_operator else _exact_1_norm(A)
    ell = 2
    norm_info = LazyOperatorNormInfo(t * A, A_1_norm=t * A_1_norm, ell=ell)
    if t * A_1_norm == 0:
        (m_star, s) = (0, 1)
    else:
        (m_star, s) = _fragment_3_1(norm_info, n0, tol, ell=ell)
    X[0] = _expm_multiply_simple_core(A, B, t_0, mu, m_star, s)
    if q <= s:
        if status_only:
            return 0
        else:
            return _expm_multiply_interval_core_0(A, X, h, mu, q, norm_info, tol, ell, n0)
    elif not q % s:
        if status_only:
            return 1
        else:
            return _expm_multiply_interval_core_1(A, X, h, mu, m_star, s, q, tol)
    elif q % s:
        if status_only:
            return 2
        else:
            return _expm_multiply_interval_core_2(A, X, h, mu, m_star, s, q, tol)
    else:
        raise Exception('internal error')

def _expm_multiply_interval_core_0(A, X, h, mu, q, norm_info, tol, ell, n0):
    if False:
        return 10
    '\n    A helper function, for the case q <= s.\n    '
    if norm_info.onenorm() == 0:
        (m_star, s) = (0, 1)
    else:
        norm_info.set_scale(1.0 / q)
        (m_star, s) = _fragment_3_1(norm_info, n0, tol, ell=ell)
        norm_info.set_scale(1)
    for k in range(q):
        X[k + 1] = _expm_multiply_simple_core(A, X[k], h, mu, m_star, s)
    return (X, 0)

def _expm_multiply_interval_core_1(A, X, h, mu, m_star, s, q, tol):
    if False:
        while True:
            i = 10
    '\n    A helper function, for the case q > s and q % s == 0.\n    '
    d = q // s
    input_shape = X.shape[1:]
    K_shape = (m_star + 1,) + input_shape
    K = np.empty(K_shape, dtype=X.dtype)
    for i in range(s):
        Z = X[i * d]
        K[0] = Z
        high_p = 0
        for k in range(1, d + 1):
            F = K[0]
            c1 = _exact_inf_norm(F)
            for p in range(1, m_star + 1):
                if p > high_p:
                    K[p] = h * A.dot(K[p - 1]) / float(p)
                coeff = float(pow(k, p))
                F = F + coeff * K[p]
                inf_norm_K_p_1 = _exact_inf_norm(K[p])
                c2 = coeff * inf_norm_K_p_1
                if c1 + c2 <= tol * _exact_inf_norm(F):
                    break
                c1 = c2
            X[k + i * d] = np.exp(k * h * mu) * F
    return (X, 1)

def _expm_multiply_interval_core_2(A, X, h, mu, m_star, s, q, tol):
    if False:
        while True:
            i = 10
    '\n    A helper function, for the case q > s and q % s > 0.\n    '
    d = q // s
    j = q // d
    r = q - d * j
    input_shape = X.shape[1:]
    K_shape = (m_star + 1,) + input_shape
    K = np.empty(K_shape, dtype=X.dtype)
    for i in range(j + 1):
        Z = X[i * d]
        K[0] = Z
        high_p = 0
        if i < j:
            effective_d = d
        else:
            effective_d = r
        for k in range(1, effective_d + 1):
            F = K[0]
            c1 = _exact_inf_norm(F)
            for p in range(1, m_star + 1):
                if p == high_p + 1:
                    K[p] = h * A.dot(K[p - 1]) / float(p)
                    high_p = p
                coeff = float(pow(k, p))
                F = F + coeff * K[p]
                inf_norm_K_p_1 = _exact_inf_norm(K[p])
                c2 = coeff * inf_norm_K_p_1
                if c1 + c2 <= tol * _exact_inf_norm(F):
                    break
                c1 = c2
            X[k + i * d] = np.exp(k * h * mu) * F
    return (X, 2)