"""Sparse Equations and Least Squares.

The original Fortran code was written by C. C. Paige and M. A. Saunders as
described in

C. C. Paige and M. A. Saunders, LSQR: An algorithm for sparse linear
equations and sparse least squares, TOMS 8(1), 43--71 (1982).

C. C. Paige and M. A. Saunders, Algorithm 583; LSQR: Sparse linear
equations and least-squares problems, TOMS 8(2), 195--209 (1982).

It is licensed under the following BSD license:

Copyright (c) 2006, Systems Optimization Laboratory
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above
      copyright notice, this list of conditions and the following
      disclaimer in the documentation and/or other materials provided
      with the distribution.

    * Neither the name of Stanford University nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The Fortran code was translated to Python for use in CVXOPT by Jeffery
Kline with contributions by Mridul Aanjaneya and Bob Myhill.

Adapted for SciPy by Stefan van der Walt.

"""
__all__ = ['lsqr']
import numpy as np
from math import sqrt
from scipy.sparse.linalg._interface import aslinearoperator
eps = np.finfo(np.float64).eps

def _sym_ortho(a, b):
    if False:
        while True:
            i = 10
    '\n    Stable implementation of Givens rotation.\n\n    Notes\n    -----\n    The routine \'SymOrtho\' was added for numerical stability. This is\n    recommended by S.-C. Choi in [1]_.  It removes the unpleasant potential of\n    ``1/eps`` in some important places (see, for example text following\n    "Compute the next plane rotation Qk" in minres.py).\n\n    References\n    ----------\n    .. [1] S.-C. Choi, "Iterative Methods for Singular Linear Equations\n           and Least-Squares Problems", Dissertation,\n           http://www.stanford.edu/group/SOL/dissertations/sou-cheng-choi-thesis.pdf\n\n    '
    if b == 0:
        return (np.sign(a), 0, abs(a))
    elif a == 0:
        return (0, np.sign(b), abs(b))
    elif abs(b) > abs(a):
        tau = a / b
        s = np.sign(b) / sqrt(1 + tau * tau)
        c = s * tau
        r = b / s
    else:
        tau = b / a
        c = np.sign(a) / sqrt(1 + tau * tau)
        s = c * tau
        r = a / c
    return (c, s, r)

def lsqr(A, b, damp=0.0, atol=1e-06, btol=1e-06, conlim=100000000.0, iter_lim=None, show=False, calc_var=False, x0=None):
    if False:
        while True:
            i = 10
    'Find the least-squares solution to a large, sparse, linear system\n    of equations.\n\n    The function solves ``Ax = b``  or  ``min ||Ax - b||^2`` or\n    ``min ||Ax - b||^2 + d^2 ||x - x0||^2``.\n\n    The matrix A may be square or rectangular (over-determined or\n    under-determined), and may have any rank.\n\n    ::\n\n      1. Unsymmetric equations --    solve  Ax = b\n\n      2. Linear least squares  --    solve  Ax = b\n                                     in the least-squares sense\n\n      3. Damped least squares  --    solve  (   A    )*x = (    b    )\n                                            ( damp*I )     ( damp*x0 )\n                                     in the least-squares sense\n\n    Parameters\n    ----------\n    A : {sparse matrix, ndarray, LinearOperator}\n        Representation of an m-by-n matrix.\n        Alternatively, ``A`` can be a linear operator which can\n        produce ``Ax`` and ``A^T x`` using, e.g.,\n        ``scipy.sparse.linalg.LinearOperator``.\n    b : array_like, shape (m,)\n        Right-hand side vector ``b``.\n    damp : float\n        Damping coefficient. Default is 0.\n    atol, btol : float, optional\n        Stopping tolerances. `lsqr` continues iterations until a\n        certain backward error estimate is smaller than some quantity\n        depending on atol and btol.  Let ``r = b - Ax`` be the\n        residual vector for the current approximate solution ``x``.\n        If ``Ax = b`` seems to be consistent, `lsqr` terminates\n        when ``norm(r) <= atol * norm(A) * norm(x) + btol * norm(b)``.\n        Otherwise, `lsqr` terminates when ``norm(A^H r) <=\n        atol * norm(A) * norm(r)``.  If both tolerances are 1.0e-6 (default),\n        the final ``norm(r)`` should be accurate to about 6\n        digits. (The final ``x`` will usually have fewer correct digits,\n        depending on ``cond(A)`` and the size of LAMBDA.)  If `atol`\n        or `btol` is None, a default value of 1.0e-6 will be used.\n        Ideally, they should be estimates of the relative error in the\n        entries of ``A`` and ``b`` respectively.  For example, if the entries\n        of ``A`` have 7 correct digits, set ``atol = 1e-7``. This prevents\n        the algorithm from doing unnecessary work beyond the\n        uncertainty of the input data.\n    conlim : float, optional\n        Another stopping tolerance.  lsqr terminates if an estimate of\n        ``cond(A)`` exceeds `conlim`.  For compatible systems ``Ax =\n        b``, `conlim` could be as large as 1.0e+12 (say).  For\n        least-squares problems, conlim should be less than 1.0e+8.\n        Maximum precision can be obtained by setting ``atol = btol =\n        conlim = zero``, but the number of iterations may then be\n        excessive. Default is 1e8.\n    iter_lim : int, optional\n        Explicit limitation on number of iterations (for safety).\n    show : bool, optional\n        Display an iteration log. Default is False.\n    calc_var : bool, optional\n        Whether to estimate diagonals of ``(A\'A + damp^2*I)^{-1}``.\n    x0 : array_like, shape (n,), optional\n        Initial guess of x, if None zeros are used. Default is None.\n\n        .. versionadded:: 1.0.0\n\n    Returns\n    -------\n    x : ndarray of float\n        The final solution.\n    istop : int\n        Gives the reason for termination.\n        1 means x is an approximate solution to Ax = b.\n        2 means x approximately solves the least-squares problem.\n    itn : int\n        Iteration number upon termination.\n    r1norm : float\n        ``norm(r)``, where ``r = b - Ax``.\n    r2norm : float\n        ``sqrt( norm(r)^2  +  damp^2 * norm(x - x0)^2 )``.  Equal to `r1norm`\n        if ``damp == 0``.\n    anorm : float\n        Estimate of Frobenius norm of ``Abar = [[A]; [damp*I]]``.\n    acond : float\n        Estimate of ``cond(Abar)``.\n    arnorm : float\n        Estimate of ``norm(A\'@r - damp^2*(x - x0))``.\n    xnorm : float\n        ``norm(x)``\n    var : ndarray of float\n        If ``calc_var`` is True, estimates all diagonals of\n        ``(A\'A)^{-1}`` (if ``damp == 0``) or more generally ``(A\'A +\n        damp^2*I)^{-1}``.  This is well defined if A has full column\n        rank or ``damp > 0``.  (Not sure what var means if ``rank(A)\n        < n`` and ``damp = 0.``)\n\n    Notes\n    -----\n    LSQR uses an iterative method to approximate the solution.  The\n    number of iterations required to reach a certain accuracy depends\n    strongly on the scaling of the problem.  Poor scaling of the rows\n    or columns of A should therefore be avoided where possible.\n\n    For example, in problem 1 the solution is unaltered by\n    row-scaling.  If a row of A is very small or large compared to\n    the other rows of A, the corresponding row of ( A  b ) should be\n    scaled up or down.\n\n    In problems 1 and 2, the solution x is easily recovered\n    following column-scaling.  Unless better information is known,\n    the nonzero columns of A should be scaled so that they all have\n    the same Euclidean norm (e.g., 1.0).\n\n    In problem 3, there is no freedom to re-scale if damp is\n    nonzero.  However, the value of damp should be assigned only\n    after attention has been paid to the scaling of A.\n\n    The parameter damp is intended to help regularize\n    ill-conditioned systems, by preventing the true solution from\n    being very large.  Another aid to regularization is provided by\n    the parameter acond, which may be used to terminate iterations\n    before the computed solution becomes very large.\n\n    If some initial estimate ``x0`` is known and if ``damp == 0``,\n    one could proceed as follows:\n\n      1. Compute a residual vector ``r0 = b - A@x0``.\n      2. Use LSQR to solve the system  ``A@dx = r0``.\n      3. Add the correction dx to obtain a final solution ``x = x0 + dx``.\n\n    This requires that ``x0`` be available before and after the call\n    to LSQR.  To judge the benefits, suppose LSQR takes k1 iterations\n    to solve A@x = b and k2 iterations to solve A@dx = r0.\n    If x0 is "good", norm(r0) will be smaller than norm(b).\n    If the same stopping tolerances atol and btol are used for each\n    system, k1 and k2 will be similar, but the final solution x0 + dx\n    should be more accurate.  The only way to reduce the total work\n    is to use a larger stopping tolerance for the second system.\n    If some value btol is suitable for A@x = b, the larger value\n    btol*norm(b)/norm(r0)  should be suitable for A@dx = r0.\n\n    Preconditioning is another way to reduce the number of iterations.\n    If it is possible to solve a related system ``M@x = b``\n    efficiently, where M approximates A in some helpful way (e.g. M -\n    A has low rank or its elements are small relative to those of A),\n    LSQR may converge more rapidly on the system ``A@M(inverse)@z =\n    b``, after which x can be recovered by solving M@x = z.\n\n    If A is symmetric, LSQR should not be used!\n\n    Alternatives are the symmetric conjugate-gradient method (cg)\n    and/or SYMMLQ.  SYMMLQ is an implementation of symmetric cg that\n    applies to any symmetric A and will converge more rapidly than\n    LSQR.  If A is positive definite, there are other implementations\n    of symmetric cg that require slightly less work per iteration than\n    SYMMLQ (but will take the same number of iterations).\n\n    References\n    ----------\n    .. [1] C. C. Paige and M. A. Saunders (1982a).\n           "LSQR: An algorithm for sparse linear equations and\n           sparse least squares", ACM TOMS 8(1), 43-71.\n    .. [2] C. C. Paige and M. A. Saunders (1982b).\n           "Algorithm 583.  LSQR: Sparse linear equations and least\n           squares problems", ACM TOMS 8(2), 195-209.\n    .. [3] M. A. Saunders (1995).  "Solution of sparse rectangular\n           systems using LSQR and CRAIG", BIT 35, 588-604.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy.sparse import csc_matrix\n    >>> from scipy.sparse.linalg import lsqr\n    >>> A = csc_matrix([[1., 0.], [1., 1.], [0., 1.]], dtype=float)\n\n    The first example has the trivial solution ``[0, 0]``\n\n    >>> b = np.array([0., 0., 0.], dtype=float)\n    >>> x, istop, itn, normr = lsqr(A, b)[:4]\n    >>> istop\n    0\n    >>> x\n    array([ 0.,  0.])\n\n    The stopping code `istop=0` returned indicates that a vector of zeros was\n    found as a solution. The returned solution `x` indeed contains\n    ``[0., 0.]``. The next example has a non-trivial solution:\n\n    >>> b = np.array([1., 0., -1.], dtype=float)\n    >>> x, istop, itn, r1norm = lsqr(A, b)[:4]\n    >>> istop\n    1\n    >>> x\n    array([ 1., -1.])\n    >>> itn\n    1\n    >>> r1norm\n    4.440892098500627e-16\n\n    As indicated by `istop=1`, `lsqr` found a solution obeying the tolerance\n    limits. The given solution ``[1., -1.]`` obviously solves the equation. The\n    remaining return values include information about the number of iterations\n    (`itn=1`) and the remaining difference of left and right side of the solved\n    equation.\n    The final example demonstrates the behavior in the case where there is no\n    solution for the equation:\n\n    >>> b = np.array([1., 0.01, -1.], dtype=float)\n    >>> x, istop, itn, r1norm = lsqr(A, b)[:4]\n    >>> istop\n    2\n    >>> x\n    array([ 1.00333333, -0.99666667])\n    >>> A.dot(x)-b\n    array([ 0.00333333, -0.00333333,  0.00333333])\n    >>> r1norm\n    0.005773502691896255\n\n    `istop` indicates that the system is inconsistent and thus `x` is rather an\n    approximate solution to the corresponding least-squares problem. `r1norm`\n    contains the norm of the minimal residual that was found.\n    '
    A = aslinearoperator(A)
    b = np.atleast_1d(b)
    if b.ndim > 1:
        b = b.squeeze()
    (m, n) = A.shape
    if iter_lim is None:
        iter_lim = 2 * n
    var = np.zeros(n)
    msg = ('The exact solution is  x = 0                              ', 'Ax - b is small enough, given atol, btol                  ', 'The least-squares solution is good enough, given atol     ', 'The estimate of cond(Abar) has exceeded conlim            ', 'Ax - b is small enough for this machine                   ', 'The least-squares solution is good enough for this machine', 'Cond(Abar) seems to be too large for this machine         ', 'The iteration limit has been reached                      ')
    if show:
        print(' ')
        print('LSQR            Least-squares solution of  Ax = b')
        str1 = f'The matrix A has {m} rows and {n} columns'
        str2 = f'damp = {damp:20.14e}   calc_var = {calc_var:8g}'
        str3 = f'atol = {atol:8.2e}                 conlim = {conlim:8.2e}'
        str4 = f'btol = {btol:8.2e}               iter_lim = {iter_lim:8g}'
        print(str1)
        print(str2)
        print(str3)
        print(str4)
    itn = 0
    istop = 0
    ctol = 0
    if conlim > 0:
        ctol = 1 / conlim
    anorm = 0
    acond = 0
    dampsq = damp ** 2
    ddnorm = 0
    res2 = 0
    xnorm = 0
    xxnorm = 0
    z = 0
    cs2 = -1
    sn2 = 0
    u = b
    bnorm = np.linalg.norm(b)
    if x0 is None:
        x = np.zeros(n)
        beta = bnorm.copy()
    else:
        x = np.asarray(x0)
        u = u - A.matvec(x)
        beta = np.linalg.norm(u)
    if beta > 0:
        u = 1 / beta * u
        v = A.rmatvec(u)
        alfa = np.linalg.norm(v)
    else:
        v = x.copy()
        alfa = 0
    if alfa > 0:
        v = 1 / alfa * v
    w = v.copy()
    rhobar = alfa
    phibar = beta
    rnorm = beta
    r1norm = rnorm
    r2norm = rnorm
    arnorm = alfa * beta
    if arnorm == 0:
        if show:
            print(msg[0])
        return (x, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm, var)
    head1 = '   Itn      x[0]       r1norm     r2norm '
    head2 = ' Compatible    LS      Norm A   Cond A'
    if show:
        print(' ')
        print(head1, head2)
        test1 = 1
        test2 = alfa / beta
        str1 = f'{itn:6g} {x[0]:12.5e}'
        str2 = f' {r1norm:10.3e} {r2norm:10.3e}'
        str3 = f'  {test1:8.1e} {test2:8.1e}'
        print(str1, str2, str3)
    while itn < iter_lim:
        itn = itn + 1
        u = A.matvec(v) - alfa * u
        beta = np.linalg.norm(u)
        if beta > 0:
            u = 1 / beta * u
            anorm = sqrt(anorm ** 2 + alfa ** 2 + beta ** 2 + dampsq)
            v = A.rmatvec(u) - beta * v
            alfa = np.linalg.norm(v)
            if alfa > 0:
                v = 1 / alfa * v
        if damp > 0:
            rhobar1 = sqrt(rhobar ** 2 + dampsq)
            cs1 = rhobar / rhobar1
            sn1 = damp / rhobar1
            psi = sn1 * phibar
            phibar = cs1 * phibar
        else:
            rhobar1 = rhobar
            psi = 0.0
        (cs, sn, rho) = _sym_ortho(rhobar1, beta)
        theta = sn * alfa
        rhobar = -cs * alfa
        phi = cs * phibar
        phibar = sn * phibar
        tau = sn * phi
        t1 = phi / rho
        t2 = -theta / rho
        dk = 1 / rho * w
        x = x + t1 * w
        w = v + t2 * w
        ddnorm = ddnorm + np.linalg.norm(dk) ** 2
        if calc_var:
            var = var + dk ** 2
        delta = sn2 * rho
        gambar = -cs2 * rho
        rhs = phi - delta * z
        zbar = rhs / gambar
        xnorm = sqrt(xxnorm + zbar ** 2)
        gamma = sqrt(gambar ** 2 + theta ** 2)
        cs2 = gambar / gamma
        sn2 = theta / gamma
        z = rhs / gamma
        xxnorm = xxnorm + z ** 2
        acond = anorm * sqrt(ddnorm)
        res1 = phibar ** 2
        res2 = res2 + psi ** 2
        rnorm = sqrt(res1 + res2)
        arnorm = alfa * abs(tau)
        if damp > 0:
            r1sq = rnorm ** 2 - dampsq * xxnorm
            r1norm = sqrt(abs(r1sq))
            if r1sq < 0:
                r1norm = -r1norm
        else:
            r1norm = rnorm
        r2norm = rnorm
        test1 = rnorm / bnorm
        test2 = arnorm / (anorm * rnorm + eps)
        test3 = 1 / (acond + eps)
        t1 = test1 / (1 + anorm * xnorm / bnorm)
        rtol = btol + atol * anorm * xnorm / bnorm
        if itn >= iter_lim:
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
            prnt = False
            if n <= 40:
                prnt = True
            if itn <= 10:
                prnt = True
            if itn >= iter_lim - 10:
                prnt = True
            if test3 <= 2 * ctol:
                prnt = True
            if test2 <= 10 * atol:
                prnt = True
            if test1 <= 10 * rtol:
                prnt = True
            if istop != 0:
                prnt = True
            if prnt:
                str1 = f'{itn:6g} {x[0]:12.5e}'
                str2 = f' {r1norm:10.3e} {r2norm:10.3e}'
                str3 = f'  {test1:8.1e} {test2:8.1e}'
                str4 = f' {anorm:8.1e} {acond:8.1e}'
                print(str1, str2, str3, str4)
        if istop != 0:
            break
    if show:
        print(' ')
        print('LSQR finished')
        print(msg[istop])
        print(' ')
        str1 = f'istop ={istop:8g}   r1norm ={r1norm:8.1e}'
        str2 = f'anorm ={anorm:8.1e}   arnorm ={arnorm:8.1e}'
        str3 = f'itn   ={itn:8g}   r2norm ={r2norm:8.1e}'
        str4 = f'acond ={acond:8.1e}   xnorm  ={xnorm:8.1e}'
        print(str1 + '   ' + str2)
        print(str3 + '   ' + str4)
        print(' ')
    return (x, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm, var)