from numpy import inner, zeros, inf, finfo
from numpy.linalg import norm
from math import sqrt
from .utils import make_system
from scipy._lib.deprecation import _deprecate_positional_args
__all__ = ['minres']

@_deprecate_positional_args(version='1.14.0')
def minres(A, b, x0=None, *, shift=0.0, tol=1e-05, maxiter=None, M=None, callback=None, show=False, check=False):
    if False:
        print('Hello World!')
    '\n    Use MINimum RESidual iteration to solve Ax=b\n\n    MINRES minimizes norm(Ax - b) for a real symmetric matrix A.  Unlike\n    the Conjugate Gradient method, A can be indefinite or singular.\n\n    If shift != 0 then the method solves (A - shift*I)x = b\n\n    Parameters\n    ----------\n    A : {sparse matrix, ndarray, LinearOperator}\n        The real symmetric N-by-N matrix of the linear system\n        Alternatively, ``A`` can be a linear operator which can\n        produce ``Ax`` using, e.g.,\n        ``scipy.sparse.linalg.LinearOperator``.\n    b : ndarray\n        Right hand side of the linear system. Has shape (N,) or (N,1).\n\n    Returns\n    -------\n    x : ndarray\n        The converged solution.\n    info : integer\n        Provides convergence information:\n            0  : successful exit\n            >0 : convergence to tolerance not achieved, number of iterations\n            <0 : illegal input or breakdown\n\n    Other Parameters\n    ----------------\n    x0 : ndarray\n        Starting guess for the solution.\n    shift : float\n        Value to apply to the system ``(A - shift * I)x = b``. Default is 0.\n    tol : float\n        Tolerance to achieve. The algorithm terminates when the relative\n        residual is below `tol`.\n    maxiter : integer\n        Maximum number of iterations.  Iteration will stop after maxiter\n        steps even if the specified tolerance has not been achieved.\n    M : {sparse matrix, ndarray, LinearOperator}\n        Preconditioner for A.  The preconditioner should approximate the\n        inverse of A.  Effective preconditioning dramatically improves the\n        rate of convergence, which implies that fewer iterations are needed\n        to reach a given error tolerance.\n    callback : function\n        User-supplied function to call after each iteration.  It is called\n        as callback(xk), where xk is the current solution vector.\n    show : bool\n        If ``True``, print out a summary and metrics related to the solution\n        during iterations. Default is ``False``.\n    check : bool\n        If ``True``, run additional input validation to check that `A` and\n        `M` (if specified) are symmetric. Default is ``False``.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy.sparse import csc_matrix\n    >>> from scipy.sparse.linalg import minres\n    >>> A = csc_matrix([[3, 2, 0], [1, -1, 0], [0, 5, 1]], dtype=float)\n    >>> A = A + A.T\n    >>> b = np.array([2, 4, -1], dtype=float)\n    >>> x, exitCode = minres(A, b)\n    >>> print(exitCode)            # 0 indicates successful convergence\n    0\n    >>> np.allclose(A.dot(x), b)\n    True\n\n    References\n    ----------\n    Solution of sparse indefinite systems of linear equations,\n        C. C. Paige and M. A. Saunders (1975),\n        SIAM J. Numer. Anal. 12(4), pp. 617-629.\n        https://web.stanford.edu/group/SOL/software/minres/\n\n    This file is a translation of the following MATLAB implementation:\n        https://web.stanford.edu/group/SOL/software/minres/minres-matlab.zip\n\n    '
    (A, M, x, b, postprocess) = make_system(A, M, x0, b)
    matvec = A.matvec
    psolve = M.matvec
    first = 'Enter minres.   '
    last = 'Exit  minres.   '
    n = A.shape[0]
    if maxiter is None:
        maxiter = 5 * n
    msg = [' beta2 = 0.  If M = I, b and x are eigenvectors    ', ' beta1 = 0.  The exact solution is x0          ', ' A solution to Ax = b was found, given rtol        ', ' A least-squares solution was found, given rtol    ', ' Reasonable accuracy achieved, given eps           ', ' x has converged to an eigenvector                 ', ' acond has exceeded 0.1/eps                        ', ' The iteration limit was reached                   ', ' A  does not define a symmetric matrix             ', ' M  does not define a symmetric matrix             ', ' M  does not define a pos-def preconditioner       ']
    if show:
        print(first + 'Solution of symmetric Ax = b')
        print(first + f'n      =  {n:3g}     shift  =  {shift:23.14e}')
        print(first + f'itnlim =  {maxiter:3g}     rtol   =  {tol:11.2e}')
        print()
    istop = 0
    itn = 0
    Anorm = 0
    Acond = 0
    rnorm = 0
    ynorm = 0
    xtype = x.dtype
    eps = finfo(xtype).eps
    if x0 is None:
        r1 = b.copy()
    else:
        r1 = b - A @ x
    y = psolve(r1)
    beta1 = inner(r1, y)
    if beta1 < 0:
        raise ValueError('indefinite preconditioner')
    elif beta1 == 0:
        return (postprocess(x), 0)
    bnorm = norm(b)
    if bnorm == 0:
        x = b
        return (postprocess(x), 0)
    beta1 = sqrt(beta1)
    if check:
        w = matvec(y)
        r2 = matvec(w)
        s = inner(w, w)
        t = inner(y, r2)
        z = abs(s - t)
        epsa = (s + eps) * eps ** (1.0 / 3.0)
        if z > epsa:
            raise ValueError('non-symmetric matrix')
        r2 = psolve(y)
        s = inner(y, y)
        t = inner(r1, r2)
        z = abs(s - t)
        epsa = (s + eps) * eps ** (1.0 / 3.0)
        if z > epsa:
            raise ValueError('non-symmetric preconditioner')
    oldb = 0
    beta = beta1
    dbar = 0
    epsln = 0
    qrnorm = beta1
    phibar = beta1
    rhs1 = beta1
    rhs2 = 0
    tnorm2 = 0
    gmax = 0
    gmin = finfo(xtype).max
    cs = -1
    sn = 0
    w = zeros(n, dtype=xtype)
    w2 = zeros(n, dtype=xtype)
    r2 = r1
    if show:
        print()
        print()
        print('   Itn     x(1)     Compatible    LS       norm(A)  cond(A) gbar/|A|')
    while itn < maxiter:
        itn += 1
        s = 1.0 / beta
        v = s * y
        y = matvec(v)
        y = y - shift * v
        if itn >= 2:
            y = y - beta / oldb * r1
        alfa = inner(v, y)
        y = y - alfa / beta * r2
        r1 = r2
        r2 = y
        y = psolve(r2)
        oldb = beta
        beta = inner(r2, y)
        if beta < 0:
            raise ValueError('non-symmetric matrix')
        beta = sqrt(beta)
        tnorm2 += alfa ** 2 + oldb ** 2 + beta ** 2
        if itn == 1:
            if beta / beta1 <= 10 * eps:
                istop = -1
        oldeps = epsln
        delta = cs * dbar + sn * alfa
        gbar = sn * dbar - cs * alfa
        epsln = sn * beta
        dbar = -cs * beta
        root = norm([gbar, dbar])
        Arnorm = phibar * root
        gamma = norm([gbar, beta])
        gamma = max(gamma, eps)
        cs = gbar / gamma
        sn = beta / gamma
        phi = cs * phibar
        phibar = sn * phibar
        denom = 1.0 / gamma
        w1 = w2
        w2 = w
        w = (v - oldeps * w1 - delta * w2) * denom
        x = x + phi * w
        gmax = max(gmax, gamma)
        gmin = min(gmin, gamma)
        z = rhs1 / gamma
        rhs1 = rhs2 - delta * z
        rhs2 = -epsln * z
        Anorm = sqrt(tnorm2)
        ynorm = norm(x)
        epsa = Anorm * eps
        epsx = Anorm * ynorm * eps
        epsr = Anorm * ynorm * tol
        diag = gbar
        if diag == 0:
            diag = epsa
        qrnorm = phibar
        rnorm = qrnorm
        if ynorm == 0 or Anorm == 0:
            test1 = inf
        else:
            test1 = rnorm / (Anorm * ynorm)
        if Anorm == 0:
            test2 = inf
        else:
            test2 = root / Anorm
        Acond = gmax / gmin
        if istop == 0:
            t1 = 1 + test1
            t2 = 1 + test2
            if t2 <= 1:
                istop = 2
            if t1 <= 1:
                istop = 1
            if itn >= maxiter:
                istop = 6
            if Acond >= 0.1 / eps:
                istop = 4
            if epsx >= beta1:
                istop = 3
            if test2 <= tol:
                istop = 2
            if test1 <= tol:
                istop = 1
        prnt = False
        if n <= 40:
            prnt = True
        if itn <= 10:
            prnt = True
        if itn >= maxiter - 10:
            prnt = True
        if itn % 10 == 0:
            prnt = True
        if qrnorm <= 10 * epsx:
            prnt = True
        if qrnorm <= 10 * epsr:
            prnt = True
        if Acond <= 0.01 / eps:
            prnt = True
        if istop != 0:
            prnt = True
        if show and prnt:
            str1 = f'{itn:6g} {x[0]:12.5e} {test1:10.3e}'
            str2 = f' {test2:10.3e}'
            str3 = f' {Anorm:8.1e} {Acond:8.1e} {gbar / Anorm:8.1e}'
            print(str1 + str2 + str3)
            if itn % 10 == 0:
                print()
        if callback is not None:
            callback(x)
        if istop != 0:
            break
    if show:
        print()
        print(last + f' istop   =  {istop:3g}               itn   ={itn:5g}')
        print(last + f' Anorm   =  {Anorm:12.4e}      Acond =  {Acond:12.4e}')
        print(last + f' rnorm   =  {rnorm:12.4e}      ynorm =  {ynorm:12.4e}')
        print(last + f' Arnorm  =  {Arnorm:12.4e}')
        print(last + msg[istop + 1])
    if istop == 6:
        info = maxiter
    else:
        info = 0
    return (postprocess(x), info)