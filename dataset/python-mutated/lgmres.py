import warnings
import numpy as np
from numpy.linalg import LinAlgError
from scipy.linalg import get_blas_funcs
from .utils import make_system
from scipy._lib.deprecation import _deprecate_positional_args
from ._gcrotmk import _fgmres
__all__ = ['lgmres']

@_deprecate_positional_args(version='1.14.0')
def lgmres(A, b, x0=None, *, tol=1e-05, maxiter=1000, M=None, callback=None, inner_m=30, outer_k=3, outer_v=None, store_outer_Av=True, prepend_outer_v=False, atol=None):
    if False:
        while True:
            i = 10
    '\n    Solve a matrix equation using the LGMRES algorithm.\n\n    The LGMRES algorithm [1]_ [2]_ is designed to avoid some problems\n    in the convergence in restarted GMRES, and often converges in fewer\n    iterations.\n\n    Parameters\n    ----------\n    A : {sparse matrix, ndarray, LinearOperator}\n        The real or complex N-by-N matrix of the linear system.\n        Alternatively, ``A`` can be a linear operator which can\n        produce ``Ax`` using, e.g.,\n        ``scipy.sparse.linalg.LinearOperator``.\n    b : ndarray\n        Right hand side of the linear system. Has shape (N,) or (N,1).\n    x0 : ndarray\n        Starting guess for the solution.\n    tol, atol : float, optional\n        Tolerances for convergence, ``norm(residual) <= max(tol*norm(b), atol)``.\n        The default for ``atol`` is `tol`.\n\n        .. warning::\n\n           The default value for `atol` will be changed in a future release.\n           For future compatibility, specify `atol` explicitly.\n    maxiter : int, optional\n        Maximum number of iterations.  Iteration will stop after maxiter\n        steps even if the specified tolerance has not been achieved.\n    M : {sparse matrix, ndarray, LinearOperator}, optional\n        Preconditioner for A.  The preconditioner should approximate the\n        inverse of A.  Effective preconditioning dramatically improves the\n        rate of convergence, which implies that fewer iterations are needed\n        to reach a given error tolerance.\n    callback : function, optional\n        User-supplied function to call after each iteration.  It is called\n        as callback(xk), where xk is the current solution vector.\n    inner_m : int, optional\n        Number of inner GMRES iterations per each outer iteration.\n    outer_k : int, optional\n        Number of vectors to carry between inner GMRES iterations.\n        According to [1]_, good values are in the range of 1...3.\n        However, note that if you want to use the additional vectors to\n        accelerate solving multiple similar problems, larger values may\n        be beneficial.\n    outer_v : list of tuples, optional\n        List containing tuples ``(v, Av)`` of vectors and corresponding\n        matrix-vector products, used to augment the Krylov subspace, and\n        carried between inner GMRES iterations. The element ``Av`` can\n        be `None` if the matrix-vector product should be re-evaluated.\n        This parameter is modified in-place by `lgmres`, and can be used\n        to pass "guess" vectors in and out of the algorithm when solving\n        similar problems.\n    store_outer_Av : bool, optional\n        Whether LGMRES should store also A@v in addition to vectors `v`\n        in the `outer_v` list. Default is True.\n    prepend_outer_v : bool, optional\n        Whether to put outer_v augmentation vectors before Krylov iterates.\n        In standard LGMRES, prepend_outer_v=False.\n\n    Returns\n    -------\n    x : ndarray\n        The converged solution.\n    info : int\n        Provides convergence information:\n\n            - 0  : successful exit\n            - >0 : convergence to tolerance not achieved, number of iterations\n            - <0 : illegal input or breakdown\n\n    Notes\n    -----\n    The LGMRES algorithm [1]_ [2]_ is designed to avoid the\n    slowing of convergence in restarted GMRES, due to alternating\n    residual vectors. Typically, it often outperforms GMRES(m) of\n    comparable memory requirements by some measure, or at least is not\n    much worse.\n\n    Another advantage in this algorithm is that you can supply it with\n    \'guess\' vectors in the `outer_v` argument that augment the Krylov\n    subspace. If the solution lies close to the span of these vectors,\n    the algorithm converges faster. This can be useful if several very\n    similar matrices need to be inverted one after another, such as in\n    Newton-Krylov iteration where the Jacobian matrix often changes\n    little in the nonlinear steps.\n\n    References\n    ----------\n    .. [1] A.H. Baker and E.R. Jessup and T. Manteuffel, "A Technique for\n             Accelerating the Convergence of Restarted GMRES", SIAM J. Matrix\n             Anal. Appl. 26, 962 (2005).\n    .. [2] A.H. Baker, "On Improving the Performance of the Linear Solver\n             restarted GMRES", PhD thesis, University of Colorado (2003).\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy.sparse import csc_matrix\n    >>> from scipy.sparse.linalg import lgmres\n    >>> A = csc_matrix([[3, 2, 0], [1, -1, 0], [0, 5, 1]], dtype=float)\n    >>> b = np.array([2, 4, -1], dtype=float)\n    >>> x, exitCode = lgmres(A, b, atol=1e-5)\n    >>> print(exitCode)            # 0 indicates successful convergence\n    0\n    >>> np.allclose(A.dot(x), b)\n    True\n    '
    (A, M, x, b, postprocess) = make_system(A, M, x0, b)
    if not np.isfinite(b).all():
        raise ValueError('RHS must contain only finite numbers')
    if atol is None:
        warnings.warn('scipy.sparse.linalg.lgmres called without specifying `atol`. The default value will change in the future. To preserve current behavior, set ``atol=tol``.', category=DeprecationWarning, stacklevel=2)
        atol = tol
    matvec = A.matvec
    psolve = M.matvec
    if outer_v is None:
        outer_v = []
    (axpy, dot, scal) = (None, None, None)
    nrm2 = get_blas_funcs('nrm2', [b])
    b_norm = nrm2(b)
    if b_norm == 0:
        x = b
        return (postprocess(x), 0)
    ptol_max_factor = 1.0
    for k_outer in range(maxiter):
        r_outer = matvec(x) - b
        if callback is not None:
            callback(x)
        if axpy is None:
            if np.iscomplexobj(r_outer) and (not np.iscomplexobj(x)):
                x = x.astype(r_outer.dtype)
            (axpy, dot, scal, nrm2) = get_blas_funcs(['axpy', 'dot', 'scal', 'nrm2'], (x, r_outer))
        r_norm = nrm2(r_outer)
        if r_norm <= max(atol, tol * b_norm):
            break
        v0 = -psolve(r_outer)
        inner_res_0 = nrm2(v0)
        if inner_res_0 == 0:
            rnorm = nrm2(r_outer)
            raise RuntimeError('Preconditioner returned a zero vector; |v| ~ %.1g, |M v| = 0' % rnorm)
        v0 = scal(1.0 / inner_res_0, v0)
        ptol = min(ptol_max_factor, max(atol, tol * b_norm) / r_norm)
        try:
            (Q, R, B, vs, zs, y, pres) = _fgmres(matvec, v0, inner_m, lpsolve=psolve, atol=ptol, outer_v=outer_v, prepend_outer_v=prepend_outer_v)
            y *= inner_res_0
            if not np.isfinite(y).all():
                raise LinAlgError()
        except LinAlgError:
            return (postprocess(x), k_outer + 1)
        if pres > ptol:
            ptol_max_factor = min(1.0, 1.5 * ptol_max_factor)
        else:
            ptol_max_factor = max(1e-16, 0.25 * ptol_max_factor)
        dx = zs[0] * y[0]
        for (w, yc) in zip(zs[1:], y[1:]):
            dx = axpy(w, dx, dx.shape[0], yc)
        nx = nrm2(dx)
        if nx > 0:
            if store_outer_Av:
                q = Q.dot(R.dot(y))
                ax = vs[0] * q[0]
                for (v, qc) in zip(vs[1:], q[1:]):
                    ax = axpy(v, ax, ax.shape[0], qc)
                outer_v.append((dx / nx, ax / nx))
            else:
                outer_v.append((dx / nx, None))
        while len(outer_v) > outer_k:
            del outer_v[0]
        x += dx
    else:
        return (postprocess(x), maxiter)
    return (postprocess(x), 0)