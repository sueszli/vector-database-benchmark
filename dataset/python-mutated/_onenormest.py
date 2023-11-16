"""Sparse block 1-norm estimator.
"""
import numpy as np
from scipy.sparse.linalg import aslinearoperator
__all__ = ['onenormest']

def onenormest(A, t=2, itmax=5, compute_v=False, compute_w=False):
    if False:
        return 10
    '\n    Compute a lower bound of the 1-norm of a sparse matrix.\n\n    Parameters\n    ----------\n    A : ndarray or other linear operator\n        A linear operator that can be transposed and that can\n        produce matrix products.\n    t : int, optional\n        A positive parameter controlling the tradeoff between\n        accuracy versus time and memory usage.\n        Larger values take longer and use more memory\n        but give more accurate output.\n    itmax : int, optional\n        Use at most this many iterations.\n    compute_v : bool, optional\n        Request a norm-maximizing linear operator input vector if True.\n    compute_w : bool, optional\n        Request a norm-maximizing linear operator output vector if True.\n\n    Returns\n    -------\n    est : float\n        An underestimate of the 1-norm of the sparse matrix.\n    v : ndarray, optional\n        The vector such that ||Av||_1 == est*||v||_1.\n        It can be thought of as an input to the linear operator\n        that gives an output with particularly large norm.\n    w : ndarray, optional\n        The vector Av which has relatively large 1-norm.\n        It can be thought of as an output of the linear operator\n        that is relatively large in norm compared to the input.\n\n    Notes\n    -----\n    This is algorithm 2.4 of [1].\n\n    In [2] it is described as follows.\n    "This algorithm typically requires the evaluation of\n    about 4t matrix-vector products and almost invariably\n    produces a norm estimate (which is, in fact, a lower\n    bound on the norm) correct to within a factor 3."\n\n    .. versionadded:: 0.13.0\n\n    References\n    ----------\n    .. [1] Nicholas J. Higham and Francoise Tisseur (2000),\n           "A Block Algorithm for Matrix 1-Norm Estimation,\n           with an Application to 1-Norm Pseudospectra."\n           SIAM J. Matrix Anal. Appl. Vol. 21, No. 4, pp. 1185-1201.\n\n    .. [2] Awad H. Al-Mohy and Nicholas J. Higham (2009),\n           "A new scaling and squaring algorithm for the matrix exponential."\n           SIAM J. Matrix Anal. Appl. Vol. 31, No. 3, pp. 970-989.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy.sparse import csc_matrix\n    >>> from scipy.sparse.linalg import onenormest\n    >>> A = csc_matrix([[1., 0., 0.], [5., 8., 2.], [0., -1., 0.]], dtype=float)\n    >>> A.toarray()\n    array([[ 1.,  0.,  0.],\n           [ 5.,  8.,  2.],\n           [ 0., -1.,  0.]])\n    >>> onenormest(A)\n    9.0\n    >>> np.linalg.norm(A.toarray(), ord=1)\n    9.0\n    '
    A = aslinearoperator(A)
    if A.shape[0] != A.shape[1]:
        raise ValueError('expected the operator to act like a square matrix')
    n = A.shape[1]
    if t >= n:
        A_explicit = np.asarray(aslinearoperator(A).matmat(np.identity(n)))
        if A_explicit.shape != (n, n):
            raise Exception('internal error: ', 'unexpected shape ' + str(A_explicit.shape))
        col_abs_sums = abs(A_explicit).sum(axis=0)
        if col_abs_sums.shape != (n,):
            raise Exception('internal error: ', 'unexpected shape ' + str(col_abs_sums.shape))
        argmax_j = np.argmax(col_abs_sums)
        v = elementary_vector(n, argmax_j)
        w = A_explicit[:, argmax_j]
        est = col_abs_sums[argmax_j]
    else:
        (est, v, w, nmults, nresamples) = _onenormest_core(A, A.H, t, itmax)
    if compute_v or compute_w:
        result = (est,)
        if compute_v:
            result += (v,)
        if compute_w:
            result += (w,)
        return result
    else:
        return est

def _blocked_elementwise(func):
    if False:
        i = 10
        return i + 15
    '\n    Decorator for an elementwise function, to apply it blockwise along\n    first dimension, to avoid excessive memory usage in temporaries.\n    '
    block_size = 2 ** 20

    def wrapper(x):
        if False:
            for i in range(10):
                print('nop')
        if x.shape[0] < block_size:
            return func(x)
        else:
            y0 = func(x[:block_size])
            y = np.zeros((x.shape[0],) + y0.shape[1:], dtype=y0.dtype)
            y[:block_size] = y0
            del y0
            for j in range(block_size, x.shape[0], block_size):
                y[j:j + block_size] = func(x[j:j + block_size])
            return y
    return wrapper

@_blocked_elementwise
def sign_round_up(X):
    if False:
        i = 10
        return i + 15
    '\n    This should do the right thing for both real and complex matrices.\n\n    From Higham and Tisseur:\n    "Everything in this section remains valid for complex matrices\n    provided that sign(A) is redefined as the matrix (aij / |aij|)\n    (and sign(0) = 1) transposes are replaced by conjugate transposes."\n\n    '
    Y = X.copy()
    Y[Y == 0] = 1
    Y /= np.abs(Y)
    return Y

@_blocked_elementwise
def _max_abs_axis1(X):
    if False:
        for i in range(10):
            print('nop')
    return np.max(np.abs(X), axis=1)

def _sum_abs_axis0(X):
    if False:
        while True:
            i = 10
    block_size = 2 ** 20
    r = None
    for j in range(0, X.shape[0], block_size):
        y = np.sum(np.abs(X[j:j + block_size]), axis=0)
        if r is None:
            r = y
        else:
            r += y
    return r

def elementary_vector(n, i):
    if False:
        return 10
    v = np.zeros(n, dtype=float)
    v[i] = 1
    return v

def vectors_are_parallel(v, w):
    if False:
        return 10
    if v.ndim != 1 or v.shape != w.shape:
        raise ValueError('expected conformant vectors with entries in {-1,1}')
    n = v.shape[0]
    return np.dot(v, w) == n

def every_col_of_X_is_parallel_to_a_col_of_Y(X, Y):
    if False:
        i = 10
        return i + 15
    for v in X.T:
        if not any((vectors_are_parallel(v, w) for w in Y.T)):
            return False
    return True

def column_needs_resampling(i, X, Y=None):
    if False:
        while True:
            i = 10
    (n, t) = X.shape
    v = X[:, i]
    if any((vectors_are_parallel(v, X[:, j]) for j in range(i))):
        return True
    if Y is not None:
        if any((vectors_are_parallel(v, w) for w in Y.T)):
            return True
    return False

def resample_column(i, X):
    if False:
        return 10
    X[:, i] = np.random.randint(0, 2, size=X.shape[0]) * 2 - 1

def less_than_or_close(a, b):
    if False:
        return 10
    return np.allclose(a, b) or a < b

def _algorithm_2_2(A, AT, t):
    if False:
        i = 10
        return i + 15
    "\n    This is Algorithm 2.2.\n\n    Parameters\n    ----------\n    A : ndarray or other linear operator\n        A linear operator that can produce matrix products.\n    AT : ndarray or other linear operator\n        The transpose of A.\n    t : int, optional\n        A positive parameter controlling the tradeoff between\n        accuracy versus time and memory usage.\n\n    Returns\n    -------\n    g : sequence\n        A non-negative decreasing vector\n        such that g[j] is a lower bound for the 1-norm\n        of the column of A of jth largest 1-norm.\n        The first entry of this vector is therefore a lower bound\n        on the 1-norm of the linear operator A.\n        This sequence has length t.\n    ind : sequence\n        The ith entry of ind is the index of the column A whose 1-norm\n        is given by g[i].\n        This sequence of indices has length t, and its entries are\n        chosen from range(n), possibly with repetition,\n        where n is the order of the operator A.\n\n    Notes\n    -----\n    This algorithm is mainly for testing.\n    It uses the 'ind' array in a way that is similar to\n    its usage in algorithm 2.4. This algorithm 2.2 may be easier to test,\n    so it gives a chance of uncovering bugs related to indexing\n    which could have propagated less noticeably to algorithm 2.4.\n\n    "
    A_linear_operator = aslinearoperator(A)
    AT_linear_operator = aslinearoperator(AT)
    n = A_linear_operator.shape[0]
    X = np.ones((n, t))
    if t > 1:
        X[:, 1:] = np.random.randint(0, 2, size=(n, t - 1)) * 2 - 1
    X /= float(n)
    g_prev = None
    h_prev = None
    k = 1
    ind = range(t)
    while True:
        Y = np.asarray(A_linear_operator.matmat(X))
        g = _sum_abs_axis0(Y)
        best_j = np.argmax(g)
        g.sort()
        g = g[::-1]
        S = sign_round_up(Y)
        Z = np.asarray(AT_linear_operator.matmat(S))
        h = _max_abs_axis1(Z)
        if k >= 2:
            if less_than_or_close(max(h), np.dot(Z[:, best_j], X[:, best_j])):
                break
        ind = np.argsort(h)[::-1][:t]
        h = h[ind]
        for j in range(t):
            X[:, j] = elementary_vector(n, ind[j])
        if k >= 2:
            if not less_than_or_close(g_prev[0], h_prev[0]):
                raise Exception('invariant (2.2) is violated')
            if not less_than_or_close(h_prev[0], g[0]):
                raise Exception('invariant (2.2) is violated')
        if k >= 3:
            for j in range(t):
                if not less_than_or_close(g[j], g_prev[j]):
                    raise Exception('invariant (2.3) is violated')
        g_prev = g
        h_prev = h
        k += 1
    return (g, ind)

def _onenormest_core(A, AT, t, itmax):
    if False:
        while True:
            i = 10
    '\n    Compute a lower bound of the 1-norm of a sparse matrix.\n\n    Parameters\n    ----------\n    A : ndarray or other linear operator\n        A linear operator that can produce matrix products.\n    AT : ndarray or other linear operator\n        The transpose of A.\n    t : int, optional\n        A positive parameter controlling the tradeoff between\n        accuracy versus time and memory usage.\n    itmax : int, optional\n        Use at most this many iterations.\n\n    Returns\n    -------\n    est : float\n        An underestimate of the 1-norm of the sparse matrix.\n    v : ndarray, optional\n        The vector such that ||Av||_1 == est*||v||_1.\n        It can be thought of as an input to the linear operator\n        that gives an output with particularly large norm.\n    w : ndarray, optional\n        The vector Av which has relatively large 1-norm.\n        It can be thought of as an output of the linear operator\n        that is relatively large in norm compared to the input.\n    nmults : int, optional\n        The number of matrix products that were computed.\n    nresamples : int, optional\n        The number of times a parallel column was observed,\n        necessitating a re-randomization of the column.\n\n    Notes\n    -----\n    This is algorithm 2.4.\n\n    '
    A_linear_operator = aslinearoperator(A)
    AT_linear_operator = aslinearoperator(AT)
    if itmax < 2:
        raise ValueError('at least two iterations are required')
    if t < 1:
        raise ValueError('at least one column is required')
    n = A.shape[0]
    if t >= n:
        raise ValueError('t should be smaller than the order of A')
    nmults = 0
    nresamples = 0
    X = np.ones((n, t), dtype=float)
    if t > 1:
        for i in range(1, t):
            resample_column(i, X)
        for i in range(t):
            while column_needs_resampling(i, X):
                resample_column(i, X)
                nresamples += 1
    X /= float(n)
    ind_hist = np.zeros(0, dtype=np.intp)
    est_old = 0
    S = np.zeros((n, t), dtype=float)
    k = 1
    ind = None
    while True:
        Y = np.asarray(A_linear_operator.matmat(X))
        nmults += 1
        mags = _sum_abs_axis0(Y)
        est = np.max(mags)
        best_j = np.argmax(mags)
        if est > est_old or k == 2:
            if k >= 2:
                ind_best = ind[best_j]
            w = Y[:, best_j]
        if k >= 2 and est <= est_old:
            est = est_old
            break
        est_old = est
        S_old = S
        if k > itmax:
            break
        S = sign_round_up(Y)
        del Y
        if every_col_of_X_is_parallel_to_a_col_of_Y(S, S_old):
            break
        if t > 1:
            for i in range(t):
                while column_needs_resampling(i, S, S_old):
                    resample_column(i, S)
                    nresamples += 1
        del S_old
        Z = np.asarray(AT_linear_operator.matmat(S))
        nmults += 1
        h = _max_abs_axis1(Z)
        del Z
        if k >= 2 and max(h) == h[ind_best]:
            break
        ind = np.argsort(h)[::-1][:t + len(ind_hist)].copy()
        del h
        if t > 1:
            if np.isin(ind[:t], ind_hist).all():
                break
            seen = np.isin(ind, ind_hist)
            ind = np.concatenate((ind[~seen], ind[seen]))
        for j in range(t):
            X[:, j] = elementary_vector(n, ind[j])
        new_ind = ind[:t][~np.isin(ind[:t], ind_hist)]
        ind_hist = np.concatenate((ind_hist, new_ind))
        k += 1
    v = elementary_vector(n, ind_best)
    return (est, v, w, nmults, nresamples)