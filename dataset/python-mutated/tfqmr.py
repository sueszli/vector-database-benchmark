import numpy as np
from .utils import make_system
from scipy._lib.deprecation import _deprecate_positional_args
__all__ = ['tfqmr']

@_deprecate_positional_args(version='1.14.0')
def tfqmr(A, b, x0=None, *, tol=1e-05, maxiter=None, M=None, callback=None, atol=None, show=False):
    if False:
        i = 10
        return i + 15
    '\n    Use Transpose-Free Quasi-Minimal Residual iteration to solve ``Ax = b``.\n\n    Parameters\n    ----------\n    A : {sparse matrix, ndarray, LinearOperator}\n        The real or complex N-by-N matrix of the linear system.\n        Alternatively, `A` can be a linear operator which can\n        produce ``Ax`` using, e.g.,\n        `scipy.sparse.linalg.LinearOperator`.\n    b : {ndarray}\n        Right hand side of the linear system. Has shape (N,) or (N,1).\n    x0 : {ndarray}\n        Starting guess for the solution.\n    tol, atol : float, optional\n        Tolerances for convergence, ``norm(residual) <= max(tol*norm(b-Ax0), atol)``.\n        The default for `tol` is 1.0e-5.\n        The default for `atol` is ``tol * norm(b-Ax0)``.\n\n        .. warning::\n\n           The default value for `atol` will be changed in a future release.\n           For future compatibility, specify `atol` explicitly.\n    maxiter : int, optional\n        Maximum number of iterations.  Iteration will stop after maxiter\n        steps even if the specified tolerance has not been achieved.\n        Default is ``min(10000, ndofs * 10)``, where ``ndofs = A.shape[0]``.\n    M : {sparse matrix, ndarray, LinearOperator}\n        Inverse of the preconditioner of A.  M should approximate the\n        inverse of A and be easy to solve for (see Notes).  Effective\n        preconditioning dramatically improves the rate of convergence,\n        which implies that fewer iterations are needed to reach a given\n        error tolerance.  By default, no preconditioner is used.\n    callback : function, optional\n        User-supplied function to call after each iteration.  It is called\n        as `callback(xk)`, where `xk` is the current solution vector.\n    show : bool, optional\n        Specify ``show = True`` to show the convergence, ``show = False`` is\n        to close the output of the convergence.\n        Default is `False`.\n\n    Returns\n    -------\n    x : ndarray\n        The converged solution.\n    info : int\n        Provides convergence information:\n\n            - 0  : successful exit\n            - >0 : convergence to tolerance not achieved, number of iterations\n            - <0 : illegal input or breakdown\n\n    Notes\n    -----\n    The Transpose-Free QMR algorithm is derived from the CGS algorithm.\n    However, unlike CGS, the convergence curves for the TFQMR method is\n    smoothed by computing a quasi minimization of the residual norm. The\n    implementation supports left preconditioner, and the "residual norm"\n    to compute in convergence criterion is actually an upper bound on the\n    actual residual norm ``||b - Axk||``.\n\n    References\n    ----------\n    .. [1] R. W. Freund, A Transpose-Free Quasi-Minimal Residual Algorithm for\n           Non-Hermitian Linear Systems, SIAM J. Sci. Comput., 14(2), 470-482,\n           1993.\n    .. [2] Y. Saad, Iterative Methods for Sparse Linear Systems, 2nd edition,\n           SIAM, Philadelphia, 2003.\n    .. [3] C. T. Kelley, Iterative Methods for Linear and Nonlinear Equations,\n           number 16 in Frontiers in Applied Mathematics, SIAM, Philadelphia,\n           1995.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy.sparse import csc_matrix\n    >>> from scipy.sparse.linalg import tfqmr\n    >>> A = csc_matrix([[3, 2, 0], [1, -1, 0], [0, 5, 1]], dtype=float)\n    >>> b = np.array([2, 4, -1], dtype=float)\n    >>> x, exitCode = tfqmr(A, b)\n    >>> print(exitCode)            # 0 indicates successful convergence\n    0\n    >>> np.allclose(A.dot(x), b)\n    True\n    '
    dtype = A.dtype
    if np.issubdtype(dtype, np.int64):
        dtype = float
        A = A.astype(dtype)
    if np.issubdtype(b.dtype, np.int64):
        b = b.astype(dtype)
    (A, M, x, b, postprocess) = make_system(A, M, x0, b)
    if np.linalg.norm(b) == 0.0:
        x = b.copy()
        return (postprocess(x), 0)
    ndofs = A.shape[0]
    if maxiter is None:
        maxiter = min(10000, ndofs * 10)
    if x0 is None:
        r = b.copy()
    else:
        r = b - A.matvec(x)
    u = r
    w = r.copy()
    rstar = r
    v = M.matvec(A.matvec(r))
    uhat = v
    d = theta = eta = 0.0
    rho = np.inner(rstar.conjugate(), r)
    rhoLast = rho
    r0norm = np.sqrt(rho)
    tau = r0norm
    if r0norm == 0:
        return (postprocess(x), 0)
    if atol is None:
        atol = tol * r0norm
    else:
        atol = max(atol, tol * r0norm)
    for iter in range(maxiter):
        even = iter % 2 == 0
        if even:
            vtrstar = np.inner(rstar.conjugate(), v)
            if vtrstar == 0.0:
                return (postprocess(x), -1)
            alpha = rho / vtrstar
            uNext = u - alpha * v
        w -= alpha * uhat
        d = u + theta ** 2 / alpha * eta * d
        theta = np.linalg.norm(w) / tau
        c = np.sqrt(1.0 / (1 + theta ** 2))
        tau *= theta * c
        eta = c ** 2 * alpha
        z = M.matvec(d)
        x += eta * z
        if callback is not None:
            callback(x)
        if tau * np.sqrt(iter + 1) < atol:
            if show:
                print('TFQMR: Linear solve converged due to reach TOL iterations {}'.format(iter + 1))
            return (postprocess(x), 0)
        if not even:
            rho = np.inner(rstar.conjugate(), w)
            beta = rho / rhoLast
            u = w + beta * u
            v = beta * uhat + beta ** 2 * v
            uhat = M.matvec(A.matvec(u))
            v += uhat
        else:
            uhat = M.matvec(A.matvec(uNext))
            u = uNext
            rhoLast = rho
    if show:
        print('TFQMR: Linear solve not converged due to reach MAXIT iterations {}'.format(iter + 1))
    return (postprocess(x), maxiter)