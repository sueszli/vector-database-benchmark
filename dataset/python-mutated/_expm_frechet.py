"""Frechet derivative of the matrix exponential."""
import numpy as np
import scipy.linalg
__all__ = ['expm_frechet', 'expm_cond']

def expm_frechet(A, E, method=None, compute_expm=True, check_finite=True):
    if False:
        for i in range(10):
            print('nop')
    '\n    Frechet derivative of the matrix exponential of A in the direction E.\n\n    Parameters\n    ----------\n    A : (N, N) array_like\n        Matrix of which to take the matrix exponential.\n    E : (N, N) array_like\n        Matrix direction in which to take the Frechet derivative.\n    method : str, optional\n        Choice of algorithm. Should be one of\n\n        - `SPS` (default)\n        - `blockEnlarge`\n\n    compute_expm : bool, optional\n        Whether to compute also `expm_A` in addition to `expm_frechet_AE`.\n        Default is True.\n    check_finite : bool, optional\n        Whether to check that the input matrix contains only finite numbers.\n        Disabling may give a performance gain, but may result in problems\n        (crashes, non-termination) if the inputs do contain infinities or NaNs.\n\n    Returns\n    -------\n    expm_A : ndarray\n        Matrix exponential of A.\n    expm_frechet_AE : ndarray\n        Frechet derivative of the matrix exponential of A in the direction E.\n    For ``compute_expm = False``, only `expm_frechet_AE` is returned.\n\n    See Also\n    --------\n    expm : Compute the exponential of a matrix.\n\n    Notes\n    -----\n    This section describes the available implementations that can be selected\n    by the `method` parameter. The default method is *SPS*.\n\n    Method *blockEnlarge* is a naive algorithm.\n\n    Method *SPS* is Scaling-Pade-Squaring [1]_.\n    It is a sophisticated implementation which should take\n    only about 3/8 as much time as the naive implementation.\n    The asymptotics are the same.\n\n    .. versionadded:: 0.13.0\n\n    References\n    ----------\n    .. [1] Awad H. Al-Mohy and Nicholas J. Higham (2009)\n           Computing the Frechet Derivative of the Matrix Exponential,\n           with an application to Condition Number Estimation.\n           SIAM Journal On Matrix Analysis and Applications.,\n           30 (4). pp. 1639-1657. ISSN 1095-7162\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy import linalg\n    >>> rng = np.random.default_rng()\n\n    >>> A = rng.standard_normal((3, 3))\n    >>> E = rng.standard_normal((3, 3))\n    >>> expm_A, expm_frechet_AE = linalg.expm_frechet(A, E)\n    >>> expm_A.shape, expm_frechet_AE.shape\n    ((3, 3), (3, 3))\n\n    Create a 6x6 matrix containing [[A, E], [0, A]]:\n\n    >>> M = np.zeros((6, 6))\n    >>> M[:3, :3] = A\n    >>> M[:3, 3:] = E\n    >>> M[3:, 3:] = A\n\n    >>> expm_M = linalg.expm(M)\n    >>> np.allclose(expm_A, expm_M[:3, :3])\n    True\n    >>> np.allclose(expm_frechet_AE, expm_M[:3, 3:])\n    True\n\n    '
    if check_finite:
        A = np.asarray_chkfinite(A)
        E = np.asarray_chkfinite(E)
    else:
        A = np.asarray(A)
        E = np.asarray(E)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected A to be a square matrix')
    if E.ndim != 2 or E.shape[0] != E.shape[1]:
        raise ValueError('expected E to be a square matrix')
    if A.shape != E.shape:
        raise ValueError('expected A and E to be the same shape')
    if method is None:
        method = 'SPS'
    if method == 'SPS':
        (expm_A, expm_frechet_AE) = expm_frechet_algo_64(A, E)
    elif method == 'blockEnlarge':
        (expm_A, expm_frechet_AE) = expm_frechet_block_enlarge(A, E)
    else:
        raise ValueError('Unknown implementation %s' % method)
    if compute_expm:
        return (expm_A, expm_frechet_AE)
    else:
        return expm_frechet_AE

def expm_frechet_block_enlarge(A, E):
    if False:
        i = 10
        return i + 15
    '\n    This is a helper function, mostly for testing and profiling.\n    Return expm(A), frechet(A, E)\n    '
    n = A.shape[0]
    M = np.vstack([np.hstack([A, E]), np.hstack([np.zeros_like(A), A])])
    expm_M = scipy.linalg.expm(M)
    return (expm_M[:n, :n], expm_M[:n, n:])
'\nMaximal values ell_m of ||2**-s A|| such that the backward error bound\ndoes not exceed 2**-53.\n'
ell_table_61 = (None, 2.11e-08, 0.000356, 0.0108, 0.0649, 0.2, 0.437, 0.783, 1.23, 1.78, 2.42, 3.13, 3.9, 4.74, 5.63, 6.56, 7.52, 8.53, 9.56, 10.6, 11.7)

def _diff_pade3(A, E, ident):
    if False:
        print('Hello World!')
    b = (120.0, 60.0, 12.0, 1.0)
    A2 = A.dot(A)
    M2 = np.dot(A, E) + np.dot(E, A)
    U = A.dot(b[3] * A2 + b[1] * ident)
    V = b[2] * A2 + b[0] * ident
    Lu = A.dot(b[3] * M2) + E.dot(b[3] * A2 + b[1] * ident)
    Lv = b[2] * M2
    return (U, V, Lu, Lv)

def _diff_pade5(A, E, ident):
    if False:
        for i in range(10):
            print('nop')
    b = (30240.0, 15120.0, 3360.0, 420.0, 30.0, 1.0)
    A2 = A.dot(A)
    M2 = np.dot(A, E) + np.dot(E, A)
    A4 = np.dot(A2, A2)
    M4 = np.dot(A2, M2) + np.dot(M2, A2)
    U = A.dot(b[5] * A4 + b[3] * A2 + b[1] * ident)
    V = b[4] * A4 + b[2] * A2 + b[0] * ident
    Lu = A.dot(b[5] * M4 + b[3] * M2) + E.dot(b[5] * A4 + b[3] * A2 + b[1] * ident)
    Lv = b[4] * M4 + b[2] * M2
    return (U, V, Lu, Lv)

def _diff_pade7(A, E, ident):
    if False:
        for i in range(10):
            print('nop')
    b = (17297280.0, 8648640.0, 1995840.0, 277200.0, 25200.0, 1512.0, 56.0, 1.0)
    A2 = A.dot(A)
    M2 = np.dot(A, E) + np.dot(E, A)
    A4 = np.dot(A2, A2)
    M4 = np.dot(A2, M2) + np.dot(M2, A2)
    A6 = np.dot(A2, A4)
    M6 = np.dot(A4, M2) + np.dot(M4, A2)
    U = A.dot(b[7] * A6 + b[5] * A4 + b[3] * A2 + b[1] * ident)
    V = b[6] * A6 + b[4] * A4 + b[2] * A2 + b[0] * ident
    Lu = A.dot(b[7] * M6 + b[5] * M4 + b[3] * M2) + E.dot(b[7] * A6 + b[5] * A4 + b[3] * A2 + b[1] * ident)
    Lv = b[6] * M6 + b[4] * M4 + b[2] * M2
    return (U, V, Lu, Lv)

def _diff_pade9(A, E, ident):
    if False:
        i = 10
        return i + 15
    b = (17643225600.0, 8821612800.0, 2075673600.0, 302702400.0, 30270240.0, 2162160.0, 110880.0, 3960.0, 90.0, 1.0)
    A2 = A.dot(A)
    M2 = np.dot(A, E) + np.dot(E, A)
    A4 = np.dot(A2, A2)
    M4 = np.dot(A2, M2) + np.dot(M2, A2)
    A6 = np.dot(A2, A4)
    M6 = np.dot(A4, M2) + np.dot(M4, A2)
    A8 = np.dot(A4, A4)
    M8 = np.dot(A4, M4) + np.dot(M4, A4)
    U = A.dot(b[9] * A8 + b[7] * A6 + b[5] * A4 + b[3] * A2 + b[1] * ident)
    V = b[8] * A8 + b[6] * A6 + b[4] * A4 + b[2] * A2 + b[0] * ident
    Lu = A.dot(b[9] * M8 + b[7] * M6 + b[5] * M4 + b[3] * M2) + E.dot(b[9] * A8 + b[7] * A6 + b[5] * A4 + b[3] * A2 + b[1] * ident)
    Lv = b[8] * M8 + b[6] * M6 + b[4] * M4 + b[2] * M2
    return (U, V, Lu, Lv)

def expm_frechet_algo_64(A, E):
    if False:
        print('Hello World!')
    n = A.shape[0]
    s = None
    ident = np.identity(n)
    A_norm_1 = scipy.linalg.norm(A, 1)
    m_pade_pairs = ((3, _diff_pade3), (5, _diff_pade5), (7, _diff_pade7), (9, _diff_pade9))
    for (m, pade) in m_pade_pairs:
        if A_norm_1 <= ell_table_61[m]:
            (U, V, Lu, Lv) = pade(A, E, ident)
            s = 0
            break
    if s is None:
        s = max(0, int(np.ceil(np.log2(A_norm_1 / ell_table_61[13]))))
        A = A * 2.0 ** (-s)
        E = E * 2.0 ** (-s)
        A2 = np.dot(A, A)
        M2 = np.dot(A, E) + np.dot(E, A)
        A4 = np.dot(A2, A2)
        M4 = np.dot(A2, M2) + np.dot(M2, A2)
        A6 = np.dot(A2, A4)
        M6 = np.dot(A4, M2) + np.dot(M4, A2)
        b = (6.476475253248e+16, 3.238237626624e+16, 7771770303897600.0, 1187353796428800.0, 129060195264000.0, 10559470521600.0, 670442572800.0, 33522128640.0, 1323241920.0, 40840800.0, 960960.0, 16380.0, 182.0, 1.0)
        W1 = b[13] * A6 + b[11] * A4 + b[9] * A2
        W2 = b[7] * A6 + b[5] * A4 + b[3] * A2 + b[1] * ident
        Z1 = b[12] * A6 + b[10] * A4 + b[8] * A2
        Z2 = b[6] * A6 + b[4] * A4 + b[2] * A2 + b[0] * ident
        W = np.dot(A6, W1) + W2
        U = np.dot(A, W)
        V = np.dot(A6, Z1) + Z2
        Lw1 = b[13] * M6 + b[11] * M4 + b[9] * M2
        Lw2 = b[7] * M6 + b[5] * M4 + b[3] * M2
        Lz1 = b[12] * M6 + b[10] * M4 + b[8] * M2
        Lz2 = b[6] * M6 + b[4] * M4 + b[2] * M2
        Lw = np.dot(A6, Lw1) + np.dot(M6, W1) + Lw2
        Lu = np.dot(A, Lw) + np.dot(E, W)
        Lv = np.dot(A6, Lz1) + np.dot(M6, Z1) + Lz2
    lu_piv = scipy.linalg.lu_factor(-U + V)
    R = scipy.linalg.lu_solve(lu_piv, U + V)
    L = scipy.linalg.lu_solve(lu_piv, Lu + Lv + np.dot(Lu - Lv, R))
    for k in range(s):
        L = np.dot(R, L) + np.dot(L, R)
        R = np.dot(R, R)
    return (R, L)

def vec(M):
    if False:
        print('Hello World!')
    '\n    Stack columns of M to construct a single vector.\n\n    This is somewhat standard notation in linear algebra.\n\n    Parameters\n    ----------\n    M : 2-D array_like\n        Input matrix\n\n    Returns\n    -------\n    v : 1-D ndarray\n        Output vector\n\n    '
    return M.T.ravel()

def expm_frechet_kronform(A, method=None, check_finite=True):
    if False:
        print('Hello World!')
    "\n    Construct the Kronecker form of the Frechet derivative of expm.\n\n    Parameters\n    ----------\n    A : array_like with shape (N, N)\n        Matrix to be expm'd.\n    method : str, optional\n        Extra keyword to be passed to expm_frechet.\n    check_finite : bool, optional\n        Whether to check that the input matrix contains only finite numbers.\n        Disabling may give a performance gain, but may result in problems\n        (crashes, non-termination) if the inputs do contain infinities or NaNs.\n\n    Returns\n    -------\n    K : 2-D ndarray with shape (N*N, N*N)\n        Kronecker form of the Frechet derivative of the matrix exponential.\n\n    Notes\n    -----\n    This function is used to help compute the condition number\n    of the matrix exponential.\n\n    See Also\n    --------\n    expm : Compute a matrix exponential.\n    expm_frechet : Compute the Frechet derivative of the matrix exponential.\n    expm_cond : Compute the relative condition number of the matrix exponential\n                in the Frobenius norm.\n\n    "
    if check_finite:
        A = np.asarray_chkfinite(A)
    else:
        A = np.asarray(A)
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected a square matrix')
    n = A.shape[0]
    ident = np.identity(n)
    cols = []
    for i in range(n):
        for j in range(n):
            E = np.outer(ident[i], ident[j])
            F = expm_frechet(A, E, method=method, compute_expm=False, check_finite=False)
            cols.append(vec(F))
    return np.vstack(cols).T

def expm_cond(A, check_finite=True):
    if False:
        while True:
            i = 10
    '\n    Relative condition number of the matrix exponential in the Frobenius norm.\n\n    Parameters\n    ----------\n    A : 2-D array_like\n        Square input matrix with shape (N, N).\n    check_finite : bool, optional\n        Whether to check that the input matrix contains only finite numbers.\n        Disabling may give a performance gain, but may result in problems\n        (crashes, non-termination) if the inputs do contain infinities or NaNs.\n\n    Returns\n    -------\n    kappa : float\n        The relative condition number of the matrix exponential\n        in the Frobenius norm\n\n    See Also\n    --------\n    expm : Compute the exponential of a matrix.\n    expm_frechet : Compute the Frechet derivative of the matrix exponential.\n\n    Notes\n    -----\n    A faster estimate for the condition number in the 1-norm\n    has been published but is not yet implemented in SciPy.\n\n    .. versionadded:: 0.14.0\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy.linalg import expm_cond\n    >>> A = np.array([[-0.3, 0.2, 0.6], [0.6, 0.3, -0.1], [-0.7, 1.2, 0.9]])\n    >>> k = expm_cond(A)\n    >>> k\n    1.7787805864469866\n\n    '
    if check_finite:
        A = np.asarray_chkfinite(A)
    else:
        A = np.asarray(A)
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected a square matrix')
    X = scipy.linalg.expm(A)
    K = expm_frechet_kronform(A, check_finite=False)
    A_norm = scipy.linalg.norm(A, 'fro')
    X_norm = scipy.linalg.norm(X, 'fro')
    K_norm = scipy.linalg.norm(K, 2)
    kappa = K_norm * A_norm / X_norm
    return kappa