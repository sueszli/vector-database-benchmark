"""
Copyright (C) 2010 David Fong and Michael Saunders

LSMR uses an iterative method.

07 Jun 2010: Documentation updated
03 Jun 2010: First release version in Python

David Chin-lung Fong            clfong@stanford.edu
Institute for Computational and Mathematical Engineering
Stanford University

Michael Saunders                saunders@stanford.edu
Systems Optimization Laboratory
Dept of MS&E, Stanford University.

"""
__all__ = ['lsmr']
from numpy import zeros, inf, atleast_1d, result_type
from numpy.linalg import norm
from math import sqrt
from scipy.sparse.linalg._interface import aslinearoperator
from scipy.sparse.linalg._isolve.lsqr import _sym_ortho

def lsmr(A, b, damp=0.0, atol=1e-06, btol=1e-06, conlim=100000000.0, maxiter=None, show=False, x0=None):
    if False:
        while True:
            i = 10
    'Iterative solver for least-squares problems.\n\n    lsmr solves the system of linear equations ``Ax = b``. If the system\n    is inconsistent, it solves the least-squares problem ``min ||b - Ax||_2``.\n    ``A`` is a rectangular matrix of dimension m-by-n, where all cases are\n    allowed: m = n, m > n, or m < n. ``b`` is a vector of length m.\n    The matrix A may be dense or sparse (usually sparse).\n\n    Parameters\n    ----------\n    A : {sparse matrix, ndarray, LinearOperator}\n        Matrix A in the linear system.\n        Alternatively, ``A`` can be a linear operator which can\n        produce ``Ax`` and ``A^H x`` using, e.g.,\n        ``scipy.sparse.linalg.LinearOperator``.\n    b : array_like, shape (m,)\n        Vector ``b`` in the linear system.\n    damp : float\n        Damping factor for regularized least-squares. `lsmr` solves\n        the regularized least-squares problem::\n\n         min ||(b) - (  A   )x||\n             ||(0)   (damp*I) ||_2\n\n        where damp is a scalar.  If damp is None or 0, the system\n        is solved without regularization. Default is 0.\n    atol, btol : float, optional\n        Stopping tolerances. `lsmr` continues iterations until a\n        certain backward error estimate is smaller than some quantity\n        depending on atol and btol.  Let ``r = b - Ax`` be the\n        residual vector for the current approximate solution ``x``.\n        If ``Ax = b`` seems to be consistent, `lsmr` terminates\n        when ``norm(r) <= atol * norm(A) * norm(x) + btol * norm(b)``.\n        Otherwise, `lsmr` terminates when ``norm(A^H r) <=\n        atol * norm(A) * norm(r)``.  If both tolerances are 1.0e-6 (default),\n        the final ``norm(r)`` should be accurate to about 6\n        digits. (The final ``x`` will usually have fewer correct digits,\n        depending on ``cond(A)`` and the size of LAMBDA.)  If `atol`\n        or `btol` is None, a default value of 1.0e-6 will be used.\n        Ideally, they should be estimates of the relative error in the\n        entries of ``A`` and ``b`` respectively.  For example, if the entries\n        of ``A`` have 7 correct digits, set ``atol = 1e-7``. This prevents\n        the algorithm from doing unnecessary work beyond the\n        uncertainty of the input data.\n    conlim : float, optional\n        `lsmr` terminates if an estimate of ``cond(A)`` exceeds\n        `conlim`.  For compatible systems ``Ax = b``, conlim could be\n        as large as 1.0e+12 (say).  For least-squares problems,\n        `conlim` should be less than 1.0e+8. If `conlim` is None, the\n        default value is 1e+8.  Maximum precision can be obtained by\n        setting ``atol = btol = conlim = 0``, but the number of\n        iterations may then be excessive. Default is 1e8.\n    maxiter : int, optional\n        `lsmr` terminates if the number of iterations reaches\n        `maxiter`.  The default is ``maxiter = min(m, n)``.  For\n        ill-conditioned systems, a larger value of `maxiter` may be\n        needed. Default is False.\n    show : bool, optional\n        Print iterations logs if ``show=True``. Default is False.\n    x0 : array_like, shape (n,), optional\n        Initial guess of ``x``, if None zeros are used. Default is None.\n\n        .. versionadded:: 1.0.0\n\n    Returns\n    -------\n    x : ndarray of float\n        Least-square solution returned.\n    istop : int\n        istop gives the reason for stopping::\n\n          istop   = 0 means x=0 is a solution.  If x0 was given, then x=x0 is a\n                      solution.\n                  = 1 means x is an approximate solution to A@x = B,\n                      according to atol and btol.\n                  = 2 means x approximately solves the least-squares problem\n                      according to atol.\n                  = 3 means COND(A) seems to be greater than CONLIM.\n                  = 4 is the same as 1 with atol = btol = eps (machine\n                      precision)\n                  = 5 is the same as 2 with atol = eps.\n                  = 6 is the same as 3 with CONLIM = 1/eps.\n                  = 7 means ITN reached maxiter before the other stopping\n                      conditions were satisfied.\n\n    itn : int\n        Number of iterations used.\n    normr : float\n        ``norm(b-Ax)``\n    normar : float\n        ``norm(A^H (b - Ax))``\n    norma : float\n        ``norm(A)``\n    conda : float\n        Condition number of A.\n    normx : float\n        ``norm(x)``\n\n    Notes\n    -----\n\n    .. versionadded:: 0.11.0\n\n    References\n    ----------\n    .. [1] D. C.-L. Fong and M. A. Saunders,\n           "LSMR: An iterative algorithm for sparse least-squares problems",\n           SIAM J. Sci. Comput., vol. 33, pp. 2950-2971, 2011.\n           :arxiv:`1006.0758`\n    .. [2] LSMR Software, https://web.stanford.edu/group/SOL/software/lsmr/\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy.sparse import csc_matrix\n    >>> from scipy.sparse.linalg import lsmr\n    >>> A = csc_matrix([[1., 0.], [1., 1.], [0., 1.]], dtype=float)\n\n    The first example has the trivial solution ``[0, 0]``\n\n    >>> b = np.array([0., 0., 0.], dtype=float)\n    >>> x, istop, itn, normr = lsmr(A, b)[:4]\n    >>> istop\n    0\n    >>> x\n    array([0., 0.])\n\n    The stopping code `istop=0` returned indicates that a vector of zeros was\n    found as a solution. The returned solution `x` indeed contains\n    ``[0., 0.]``. The next example has a non-trivial solution:\n\n    >>> b = np.array([1., 0., -1.], dtype=float)\n    >>> x, istop, itn, normr = lsmr(A, b)[:4]\n    >>> istop\n    1\n    >>> x\n    array([ 1., -1.])\n    >>> itn\n    1\n    >>> normr\n    4.440892098500627e-16\n\n    As indicated by `istop=1`, `lsmr` found a solution obeying the tolerance\n    limits. The given solution ``[1., -1.]`` obviously solves the equation. The\n    remaining return values include information about the number of iterations\n    (`itn=1`) and the remaining difference of left and right side of the solved\n    equation.\n    The final example demonstrates the behavior in the case where there is no\n    solution for the equation:\n\n    >>> b = np.array([1., 0.01, -1.], dtype=float)\n    >>> x, istop, itn, normr = lsmr(A, b)[:4]\n    >>> istop\n    2\n    >>> x\n    array([ 1.00333333, -0.99666667])\n    >>> A.dot(x)-b\n    array([ 0.00333333, -0.00333333,  0.00333333])\n    >>> normr\n    0.005773502691896255\n\n    `istop` indicates that the system is inconsistent and thus `x` is rather an\n    approximate solution to the corresponding least-squares problem. `normr`\n    contains the minimal distance that was found.\n    '
    A = aslinearoperator(A)
    b = atleast_1d(b)
    if b.ndim > 1:
        b = b.squeeze()
    msg = ('The exact solution is x = 0, or x = x0, if x0 was given  ', 'Ax - b is small enough, given atol, btol                  ', 'The least-squares solution is good enough, given atol     ', 'The estimate of cond(Abar) has exceeded conlim            ', 'Ax - b is small enough for this machine                   ', 'The least-squares solution is good enough for this machine', 'Cond(Abar) seems to be too large for this machine         ', 'The iteration limit has been reached                      ')
    hdg1 = '   itn      x(1)       norm r    norm Ar'
    hdg2 = ' compatible   LS      norm A   cond A'
    pfreq = 20
    pcount = 0
    (m, n) = A.shape
    minDim = min([m, n])
    if maxiter is None:
        maxiter = minDim
    if x0 is None:
        dtype = result_type(A, b, float)
    else:
        dtype = result_type(A, b, x0, float)
    if show:
        print(' ')
        print('LSMR            Least-squares solution of  Ax = b\n')
        print(f'The matrix A has {m} rows and {n} columns')
        print('damp = %20.14e\n' % damp)
        print(f'atol = {atol:8.2e}                 conlim = {conlim:8.2e}\n')
        print(f'btol = {btol:8.2e}             maxiter = {maxiter:8g}\n')
    u = b
    normb = norm(b)
    if x0 is None:
        x = zeros(n, dtype)
        beta = normb.copy()
    else:
        x = atleast_1d(x0.copy())
        u = u - A.matvec(x)
        beta = norm(u)
    if beta > 0:
        u = 1 / beta * u
        v = A.rmatvec(u)
        alpha = norm(v)
    else:
        v = zeros(n, dtype)
        alpha = 0
    if alpha > 0:
        v = 1 / alpha * v
    itn = 0
    zetabar = alpha * beta
    alphabar = alpha
    rho = 1
    rhobar = 1
    cbar = 1
    sbar = 0
    h = v.copy()
    hbar = zeros(n, dtype)
    betadd = beta
    betad = 0
    rhodold = 1
    tautildeold = 0
    thetatilde = 0
    zeta = 0
    d = 0
    normA2 = alpha * alpha
    maxrbar = 0
    minrbar = 1e+100
    normA = sqrt(normA2)
    condA = 1
    normx = 0
    istop = 0
    ctol = 0
    if conlim > 0:
        ctol = 1 / conlim
    normr = beta
    normar = alpha * beta
    if normar == 0:
        if show:
            print(msg[0])
        return (x, istop, itn, normr, normar, normA, condA, normx)
    if normb == 0:
        x[()] = 0
        return (x, istop, itn, normr, normar, normA, condA, normx)
    if show:
        print(' ')
        print(hdg1, hdg2)
        test1 = 1
        test2 = alpha / beta
        str1 = f'{itn:6g} {x[0]:12.5e}'
        str2 = f' {normr:10.3e} {normar:10.3e}'
        str3 = f'  {test1:8.1e} {test2:8.1e}'
        print(''.join([str1, str2, str3]))
    while itn < maxiter:
        itn = itn + 1
        u *= -alpha
        u += A.matvec(v)
        beta = norm(u)
        if beta > 0:
            u *= 1 / beta
            v *= -beta
            v += A.rmatvec(u)
            alpha = norm(v)
            if alpha > 0:
                v *= 1 / alpha
        (chat, shat, alphahat) = _sym_ortho(alphabar, damp)
        rhoold = rho
        (c, s, rho) = _sym_ortho(alphahat, beta)
        thetanew = s * alpha
        alphabar = c * alpha
        rhobarold = rhobar
        zetaold = zeta
        thetabar = sbar * rho
        rhotemp = cbar * rho
        (cbar, sbar, rhobar) = _sym_ortho(cbar * rho, thetanew)
        zeta = cbar * zetabar
        zetabar = -sbar * zetabar
        hbar *= -(thetabar * rho / (rhoold * rhobarold))
        hbar += h
        x += zeta / (rho * rhobar) * hbar
        h *= -(thetanew / rho)
        h += v
        betaacute = chat * betadd
        betacheck = -shat * betadd
        betahat = c * betaacute
        betadd = -s * betaacute
        thetatildeold = thetatilde
        (ctildeold, stildeold, rhotildeold) = _sym_ortho(rhodold, thetabar)
        thetatilde = stildeold * rhobar
        rhodold = ctildeold * rhobar
        betad = -stildeold * betad + ctildeold * betahat
        tautildeold = (zetaold - thetatildeold * tautildeold) / rhotildeold
        taud = (zeta - thetatilde * tautildeold) / rhodold
        d = d + betacheck * betacheck
        normr = sqrt(d + (betad - taud) ** 2 + betadd * betadd)
        normA2 = normA2 + beta * beta
        normA = sqrt(normA2)
        normA2 = normA2 + alpha * alpha
        maxrbar = max(maxrbar, rhobarold)
        if itn > 1:
            minrbar = min(minrbar, rhobarold)
        condA = max(maxrbar, rhotemp) / min(minrbar, rhotemp)
        normar = abs(zetabar)
        normx = norm(x)
        test1 = normr / normb
        if normA * normr != 0:
            test2 = normar / (normA * normr)
        else:
            test2 = inf
        test3 = 1 / condA
        t1 = test1 / (1 + normA * normx / normb)
        rtol = btol + atol * normA * normx / normb
        if itn >= maxiter:
            istop = 7
        if 1 + test3 <= 1:
            istop = 6
        if 1 + test2 <= 1:
            istop = 5
        if 1 + t1 <= 1:
            istop = 4
        if test3 <= ctol:
            istop = 3
        if test2 <= atol:
            istop = 2
        if test1 <= rtol:
            istop = 1
        if show:
            if n <= 40 or itn <= 10 or itn >= maxiter - 10 or (itn % 10 == 0) or (test3 <= 1.1 * ctol) or (test2 <= 1.1 * atol) or (test1 <= 1.1 * rtol) or (istop != 0):
                if pcount >= pfreq:
                    pcount = 0
                    print(' ')
                    print(hdg1, hdg2)
                pcount = pcount + 1
                str1 = f'{itn:6g} {x[0]:12.5e}'
                str2 = f' {normr:10.3e} {normar:10.3e}'
                str3 = f'  {test1:8.1e} {test2:8.1e}'
                str4 = f' {normA:8.1e} {condA:8.1e}'
                print(''.join([str1, str2, str3, str4]))
        if istop > 0:
            break
    if show:
        print(' ')
        print('LSMR finished')
        print(msg[istop])
        print(f'istop ={istop:8g}    normr ={normr:8.1e}')
        print(f'    normA ={normA:8.1e}    normAr ={normar:8.1e}')
        print(f'itn   ={itn:8g}    condA ={condA:8.1e}')
        print('    normx =%8.1e' % normx)
        print(str1, str2)
        print(str3, str4)
    return (x, istop, itn, normr, normar, normA, condA, normx)