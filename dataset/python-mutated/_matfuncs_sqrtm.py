"""
Matrix square root for general matrices and for upper triangular matrices.

This module exists to avoid cyclic imports.

"""
__all__ = ['sqrtm']
import numpy as np
from scipy._lib._util import _asarray_validated
from ._misc import norm
from .lapack import ztrsyl, dtrsyl
from ._decomp_schur import schur, rsf2csf

class SqrtmError(np.linalg.LinAlgError):
    pass
from ._matfuncs_sqrtm_triu import within_block_loop

def _sqrtm_triu(T, blocksize=64):
    if False:
        for i in range(10):
            print('nop')
    '\n    Matrix square root of an upper triangular matrix.\n\n    This is a helper function for `sqrtm` and `logm`.\n\n    Parameters\n    ----------\n    T : (N, N) array_like upper triangular\n        Matrix whose square root to evaluate\n    blocksize : int, optional\n        If the blocksize is not degenerate with respect to the\n        size of the input array, then use a blocked algorithm. (Default: 64)\n\n    Returns\n    -------\n    sqrtm : (N, N) ndarray\n        Value of the sqrt function at `T`\n\n    References\n    ----------\n    .. [1] Edvin Deadman, Nicholas J. Higham, Rui Ralha (2013)\n           "Blocked Schur Algorithms for Computing the Matrix Square Root,\n           Lecture Notes in Computer Science, 7782. pp. 171-182.\n\n    '
    T_diag = np.diag(T)
    keep_it_real = np.isrealobj(T) and np.min(T_diag) >= 0
    if not keep_it_real:
        T = np.asarray(T, dtype=np.complex128, order='C')
        T_diag = np.asarray(T_diag, dtype=np.complex128)
    else:
        T = np.asarray(T, dtype=np.float64, order='C')
        T_diag = np.asarray(T_diag, dtype=np.float64)
    R = np.diag(np.sqrt(T_diag))
    (n, n) = T.shape
    nblocks = max(n // blocksize, 1)
    (bsmall, nlarge) = divmod(n, nblocks)
    blarge = bsmall + 1
    nsmall = nblocks - nlarge
    if nsmall * bsmall + nlarge * blarge != n:
        raise Exception('internal inconsistency')
    start_stop_pairs = []
    start = 0
    for (count, size) in ((nsmall, bsmall), (nlarge, blarge)):
        for i in range(count):
            start_stop_pairs.append((start, start + size))
            start += size
    try:
        within_block_loop(R, T, start_stop_pairs, nblocks)
    except RuntimeError as e:
        raise SqrtmError(*e.args) from e
    for j in range(nblocks):
        (jstart, jstop) = start_stop_pairs[j]
        for i in range(j - 1, -1, -1):
            (istart, istop) = start_stop_pairs[i]
            S = T[istart:istop, jstart:jstop]
            if j - i > 1:
                S = S - R[istart:istop, istop:jstart].dot(R[istop:jstart, jstart:jstop])
            Rii = R[istart:istop, istart:istop]
            Rjj = R[jstart:jstop, jstart:jstop]
            if keep_it_real:
                (x, scale, info) = dtrsyl(Rii, Rjj, S)
            else:
                (x, scale, info) = ztrsyl(Rii, Rjj, S)
            R[istart:istop, jstart:jstop] = x * scale
    return R

def sqrtm(A, disp=True, blocksize=64):
    if False:
        print('Hello World!')
    '\n    Matrix square root.\n\n    Parameters\n    ----------\n    A : (N, N) array_like\n        Matrix whose square root to evaluate\n    disp : bool, optional\n        Print warning if error in the result is estimated large\n        instead of returning estimated error. (Default: True)\n    blocksize : integer, optional\n        If the blocksize is not degenerate with respect to the\n        size of the input array, then use a blocked algorithm. (Default: 64)\n\n    Returns\n    -------\n    sqrtm : (N, N) ndarray\n        Value of the sqrt function at `A`. The dtype is float or complex.\n        The precision (data size) is determined based on the precision of\n        input `A`. When the dtype is float, the precision is the same as `A`.\n        When the dtype is complex, the precision is double that of `A`. The\n        precision might be clipped by each dtype precision range.\n\n    errest : float\n        (if disp == False)\n\n        Frobenius norm of the estimated error, ||err||_F / ||A||_F\n\n    References\n    ----------\n    .. [1] Edvin Deadman, Nicholas J. Higham, Rui Ralha (2013)\n           "Blocked Schur Algorithms for Computing the Matrix Square Root,\n           Lecture Notes in Computer Science, 7782. pp. 171-182.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy.linalg import sqrtm\n    >>> a = np.array([[1.0, 3.0], [1.0, 4.0]])\n    >>> r = sqrtm(a)\n    >>> r\n    array([[ 0.75592895,  1.13389342],\n           [ 0.37796447,  1.88982237]])\n    >>> r.dot(r)\n    array([[ 1.,  3.],\n           [ 1.,  4.]])\n\n    '
    byte_size = np.asarray(A).dtype.itemsize
    A = _asarray_validated(A, check_finite=True, as_inexact=True)
    if len(A.shape) != 2:
        raise ValueError('Non-matrix input to matrix function.')
    if blocksize < 1:
        raise ValueError('The blocksize should be at least 1.')
    keep_it_real = np.isrealobj(A)
    if keep_it_real:
        (T, Z) = schur(A)
        if not np.allclose(T, np.triu(T)):
            (T, Z) = rsf2csf(T, Z)
    else:
        (T, Z) = schur(A, output='complex')
    failflag = False
    try:
        R = _sqrtm_triu(T, blocksize=blocksize)
        ZH = np.conjugate(Z).T
        X = Z.dot(R).dot(ZH)
        if not np.iscomplexobj(X):
            X = X.astype(f'f{np.clip(byte_size, 2, 16)}', copy=False)
        elif hasattr(np, 'complex256'):
            X = X.astype(f'c{np.clip(byte_size * 2, 8, 32)}', copy=False)
        else:
            X = X.astype(f'c{np.clip(byte_size * 2, 8, 16)}', copy=False)
    except SqrtmError:
        failflag = True
        X = np.empty_like(A)
        X.fill(np.nan)
    if disp:
        if failflag:
            print('Failed to find a square root.')
        return X
    else:
        try:
            arg2 = norm(X.dot(X) - A, 'fro') ** 2 / norm(A, 'fro')
        except ValueError:
            arg2 = np.inf
        return (X, arg2)