import warnings
import numpy as np
from numpy.linalg import LinAlgError
from scipy.linalg import get_blas_funcs, qr, solve, svd, qr_insert, lstsq
from scipy.sparse.linalg._isolve.utils import make_system
from scipy._lib.deprecation import _deprecate_positional_args
__all__ = ['gcrotmk']

def _fgmres(matvec, v0, m, atol, lpsolve=None, rpsolve=None, cs=(), outer_v=(), prepend_outer_v=False):
    if False:
        i = 10
        return i + 15
    '\n    FGMRES Arnoldi process, with optional projection or augmentation\n\n    Parameters\n    ----------\n    matvec : callable\n        Operation A*x\n    v0 : ndarray\n        Initial vector, normalized to nrm2(v0) == 1\n    m : int\n        Number of GMRES rounds\n    atol : float\n        Absolute tolerance for early exit\n    lpsolve : callable\n        Left preconditioner L\n    rpsolve : callable\n        Right preconditioner R\n    cs : list of (ndarray, ndarray)\n        Columns of matrices C and U in GCROT\n    outer_v : list of ndarrays\n        Augmentation vectors in LGMRES\n    prepend_outer_v : bool, optional\n        Whether augmentation vectors come before or after\n        Krylov iterates\n\n    Raises\n    ------\n    LinAlgError\n        If nans encountered\n\n    Returns\n    -------\n    Q, R : ndarray\n        QR decomposition of the upper Hessenberg H=QR\n    B : ndarray\n        Projections corresponding to matrix C\n    vs : list of ndarray\n        Columns of matrix V\n    zs : list of ndarray\n        Columns of matrix Z\n    y : ndarray\n        Solution to ||H y - e_1||_2 = min!\n    res : float\n        The final (preconditioned) residual norm\n\n    '
    if lpsolve is None:

        def lpsolve(x):
            if False:
                i = 10
                return i + 15
            return x
    if rpsolve is None:

        def rpsolve(x):
            if False:
                i = 10
                return i + 15
            return x
    (axpy, dot, scal, nrm2) = get_blas_funcs(['axpy', 'dot', 'scal', 'nrm2'], (v0,))
    vs = [v0]
    zs = []
    y = None
    res = np.nan
    m = m + len(outer_v)
    B = np.zeros((len(cs), m), dtype=v0.dtype)
    Q = np.ones((1, 1), dtype=v0.dtype)
    R = np.zeros((1, 0), dtype=v0.dtype)
    eps = np.finfo(v0.dtype).eps
    breakdown = False
    for j in range(m):
        if prepend_outer_v and j < len(outer_v):
            (z, w) = outer_v[j]
        elif prepend_outer_v and j == len(outer_v):
            z = rpsolve(v0)
            w = None
        elif not prepend_outer_v and j >= m - len(outer_v):
            (z, w) = outer_v[j - (m - len(outer_v))]
        else:
            z = rpsolve(vs[-1])
            w = None
        if w is None:
            w = lpsolve(matvec(z))
        else:
            w = w.copy()
        w_norm = nrm2(w)
        for (i, c) in enumerate(cs):
            alpha = dot(c, w)
            B[i, j] = alpha
            w = axpy(c, w, c.shape[0], -alpha)
        hcur = np.zeros(j + 2, dtype=Q.dtype)
        for (i, v) in enumerate(vs):
            alpha = dot(v, w)
            hcur[i] = alpha
            w = axpy(v, w, v.shape[0], -alpha)
        hcur[i + 1] = nrm2(w)
        with np.errstate(over='ignore', divide='ignore'):
            alpha = 1 / hcur[-1]
        if np.isfinite(alpha):
            w = scal(alpha, w)
        if not hcur[-1] > eps * w_norm:
            breakdown = True
        vs.append(w)
        zs.append(z)
        Q2 = np.zeros((j + 2, j + 2), dtype=Q.dtype, order='F')
        Q2[:j + 1, :j + 1] = Q
        Q2[j + 1, j + 1] = 1
        R2 = np.zeros((j + 2, j), dtype=R.dtype, order='F')
        R2[:j + 1, :] = R
        (Q, R) = qr_insert(Q2, R2, hcur, j, which='col', overwrite_qru=True, check_finite=False)
        res = abs(Q[0, -1])
        if res < atol or breakdown:
            break
    if not np.isfinite(R[j, j]):
        raise LinAlgError()
    (y, _, _, _) = lstsq(R[:j + 1, :j + 1], Q[0, :j + 1].conj())
    B = B[:, :j + 1]
    return (Q, R, B, vs, zs, y, res)

@_deprecate_positional_args(version='1.14.0')
def gcrotmk(A, b, x0=None, *, tol=1e-05, maxiter=1000, M=None, callback=None, m=20, k=None, CU=None, discard_C=False, truncate='oldest', atol=None):
    if False:
        while True:
            i = 10
    "\n    Solve a matrix equation using flexible GCROT(m,k) algorithm.\n\n    Parameters\n    ----------\n    A : {sparse matrix, ndarray, LinearOperator}\n        The real or complex N-by-N matrix of the linear system.\n        Alternatively, ``A`` can be a linear operator which can\n        produce ``Ax`` using, e.g.,\n        ``scipy.sparse.linalg.LinearOperator``.\n    b : ndarray\n        Right hand side of the linear system. Has shape (N,) or (N,1).\n    x0 : ndarray\n        Starting guess for the solution.\n    tol, atol : float, optional\n        Tolerances for convergence, ``norm(residual) <= max(tol*norm(b), atol)``.\n        The default for ``atol`` is `tol`.\n\n        .. warning::\n\n           The default value for `atol` will be changed in a future release.\n           For future compatibility, specify `atol` explicitly.\n    maxiter : int, optional\n        Maximum number of iterations.  Iteration will stop after maxiter\n        steps even if the specified tolerance has not been achieved.\n    M : {sparse matrix, ndarray, LinearOperator}, optional\n        Preconditioner for A.  The preconditioner should approximate the\n        inverse of A. gcrotmk is a 'flexible' algorithm and the preconditioner\n        can vary from iteration to iteration. Effective preconditioning\n        dramatically improves the rate of convergence, which implies that\n        fewer iterations are needed to reach a given error tolerance.\n    callback : function, optional\n        User-supplied function to call after each iteration.  It is called\n        as callback(xk), where xk is the current solution vector.\n    m : int, optional\n        Number of inner FGMRES iterations per each outer iteration.\n        Default: 20\n    k : int, optional\n        Number of vectors to carry between inner FGMRES iterations.\n        According to [2]_, good values are around m.\n        Default: m\n    CU : list of tuples, optional\n        List of tuples ``(c, u)`` which contain the columns of the matrices\n        C and U in the GCROT(m,k) algorithm. For details, see [2]_.\n        The list given and vectors contained in it are modified in-place.\n        If not given, start from empty matrices. The ``c`` elements in the\n        tuples can be ``None``, in which case the vectors are recomputed\n        via ``c = A u`` on start and orthogonalized as described in [3]_.\n    discard_C : bool, optional\n        Discard the C-vectors at the end. Useful if recycling Krylov subspaces\n        for different linear systems.\n    truncate : {'oldest', 'smallest'}, optional\n        Truncation scheme to use. Drop: oldest vectors, or vectors with\n        smallest singular values using the scheme discussed in [1,2].\n        See [2]_ for detailed comparison.\n        Default: 'oldest'\n\n    Returns\n    -------\n    x : ndarray\n        The solution found.\n    info : int\n        Provides convergence information:\n\n        * 0  : successful exit\n        * >0 : convergence to tolerance not achieved, number of iterations\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy.sparse import csc_matrix\n    >>> from scipy.sparse.linalg import gcrotmk\n    >>> R = np.random.randn(5, 5)\n    >>> A = csc_matrix(R)\n    >>> b = np.random.randn(5)\n    >>> x, exit_code = gcrotmk(A, b, atol=1e-5)\n    >>> print(exit_code)\n    0\n    >>> np.allclose(A.dot(x), b)\n    True\n\n    References\n    ----------\n    .. [1] E. de Sturler, ''Truncation strategies for optimal Krylov subspace\n           methods'', SIAM J. Numer. Anal. 36, 864 (1999).\n    .. [2] J.E. Hicken and D.W. Zingg, ''A simplified and flexible variant\n           of GCROT for solving nonsymmetric linear systems'',\n           SIAM J. Sci. Comput. 32, 172 (2010).\n    .. [3] M.L. Parks, E. de Sturler, G. Mackey, D.D. Johnson, S. Maiti,\n           ''Recycling Krylov subspaces for sequences of linear systems'',\n           SIAM J. Sci. Comput. 28, 1651 (2006).\n\n    "
    (A, M, x, b, postprocess) = make_system(A, M, x0, b)
    if not np.isfinite(b).all():
        raise ValueError('RHS must contain only finite numbers')
    if truncate not in ('oldest', 'smallest'):
        raise ValueError(f"Invalid value for 'truncate': {truncate!r}")
    if atol is None:
        warnings.warn('scipy.sparse.linalg.gcrotmk called without specifying `atol`. The default value will change in the future. To preserve current behavior, set ``atol=tol``.', category=DeprecationWarning, stacklevel=2)
        atol = tol
    matvec = A.matvec
    psolve = M.matvec
    if CU is None:
        CU = []
    if k is None:
        k = m
    (axpy, dot, scal) = (None, None, None)
    if x0 is None:
        r = b.copy()
    else:
        r = b - matvec(x)
    (axpy, dot, scal, nrm2) = get_blas_funcs(['axpy', 'dot', 'scal', 'nrm2'], (x, r))
    b_norm = nrm2(b)
    if b_norm == 0:
        x = b
        return (postprocess(x), 0)
    if discard_C:
        CU[:] = [(None, u) for (c, u) in CU]
    if CU:
        CU.sort(key=lambda cu: cu[0] is not None)
        C = np.empty((A.shape[0], len(CU)), dtype=r.dtype, order='F')
        us = []
        j = 0
        while CU:
            (c, u) = CU.pop(0)
            if c is None:
                c = matvec(u)
            C[:, j] = c
            j += 1
            us.append(u)
        (Q, R, P) = qr(C, overwrite_a=True, mode='economic', pivoting=True)
        del C
        cs = list(Q.T)
        new_us = []
        for j in range(len(cs)):
            u = us[P[j]]
            for i in range(j):
                u = axpy(us[P[i]], u, u.shape[0], -R[i, j])
            if abs(R[j, j]) < 1e-12 * abs(R[0, 0]):
                break
            u = scal(1.0 / R[j, j], u)
            new_us.append(u)
        CU[:] = list(zip(cs, new_us))[::-1]
    if CU:
        (axpy, dot) = get_blas_funcs(['axpy', 'dot'], (r,))
        for (c, u) in CU:
            yc = dot(c, r)
            x = axpy(u, x, x.shape[0], yc)
            r = axpy(c, r, r.shape[0], -yc)
    for j_outer in range(maxiter):
        if callback is not None:
            callback(x)
        beta = nrm2(r)
        beta_tol = max(atol, tol * b_norm)
        if beta <= beta_tol and (j_outer > 0 or CU):
            r = b - matvec(x)
            beta = nrm2(r)
        if beta <= beta_tol:
            j_outer = -1
            break
        ml = m + max(k - len(CU), 0)
        cs = [c for (c, u) in CU]
        try:
            (Q, R, B, vs, zs, y, pres) = _fgmres(matvec, r / beta, ml, rpsolve=psolve, atol=max(atol, tol * b_norm) / beta, cs=cs)
            y *= beta
        except LinAlgError:
            break
        ux = zs[0] * y[0]
        for (z, yc) in zip(zs[1:], y[1:]):
            ux = axpy(z, ux, ux.shape[0], yc)
        by = B.dot(y)
        for (cu, byc) in zip(CU, by):
            (c, u) = cu
            ux = axpy(u, ux, ux.shape[0], -byc)
        hy = Q.dot(R.dot(y))
        cx = vs[0] * hy[0]
        for (v, hyc) in zip(vs[1:], hy[1:]):
            cx = axpy(v, cx, cx.shape[0], hyc)
        try:
            alpha = 1 / nrm2(cx)
            if not np.isfinite(alpha):
                raise FloatingPointError()
        except (FloatingPointError, ZeroDivisionError):
            continue
        cx = scal(alpha, cx)
        ux = scal(alpha, ux)
        gamma = dot(cx, r)
        r = axpy(cx, r, r.shape[0], -gamma)
        x = axpy(ux, x, x.shape[0], gamma)
        if truncate == 'oldest':
            while len(CU) >= k and CU:
                del CU[0]
        elif truncate == 'smallest':
            if len(CU) >= k and CU:
                D = solve(R[:-1, :].T, B.T).T
                (W, sigma, V) = svd(D)
                new_CU = []
                for (j, w) in enumerate(W[:, :k - 1].T):
                    (c, u) = CU[0]
                    c = c * w[0]
                    u = u * w[0]
                    for (cup, wp) in zip(CU[1:], w[1:]):
                        (cp, up) = cup
                        c = axpy(cp, c, c.shape[0], wp)
                        u = axpy(up, u, u.shape[0], wp)
                    for (cp, up) in new_CU:
                        alpha = dot(cp, c)
                        c = axpy(cp, c, c.shape[0], -alpha)
                        u = axpy(up, u, u.shape[0], -alpha)
                    alpha = nrm2(c)
                    c = scal(1.0 / alpha, c)
                    u = scal(1.0 / alpha, u)
                    new_CU.append((c, u))
                CU[:] = new_CU
        CU.append((cx, ux))
    CU.append((None, x.copy()))
    if discard_C:
        CU[:] = [(None, uz) for (cz, uz) in CU]
    return (postprocess(x), j_outer + 1)