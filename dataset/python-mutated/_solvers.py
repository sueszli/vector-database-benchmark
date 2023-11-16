"""Matrix equation solver routines"""
import warnings
import numpy as np
from numpy.linalg import inv, LinAlgError, norm, cond, svd
from ._basic import solve, solve_triangular, matrix_balance
from .lapack import get_lapack_funcs
from ._decomp_schur import schur
from ._decomp_lu import lu
from ._decomp_qr import qr
from ._decomp_qz import ordqz
from ._decomp import _asarray_validated
from ._special_matrices import kron, block_diag
__all__ = ['solve_sylvester', 'solve_continuous_lyapunov', 'solve_discrete_lyapunov', 'solve_lyapunov', 'solve_continuous_are', 'solve_discrete_are']

def solve_sylvester(a, b, q):
    if False:
        i = 10
        return i + 15
    '\n    Computes a solution (X) to the Sylvester equation :math:`AX + XB = Q`.\n\n    Parameters\n    ----------\n    a : (M, M) array_like\n        Leading matrix of the Sylvester equation\n    b : (N, N) array_like\n        Trailing matrix of the Sylvester equation\n    q : (M, N) array_like\n        Right-hand side\n\n    Returns\n    -------\n    x : (M, N) ndarray\n        The solution to the Sylvester equation.\n\n    Raises\n    ------\n    LinAlgError\n        If solution was not found\n\n    Notes\n    -----\n    Computes a solution to the Sylvester matrix equation via the Bartels-\n    Stewart algorithm. The A and B matrices first undergo Schur\n    decompositions. The resulting matrices are used to construct an\n    alternative Sylvester equation (``RY + YS^T = F``) where the R and S\n    matrices are in quasi-triangular form (or, when R, S or F are complex,\n    triangular form). The simplified equation is then solved using\n    ``*TRSYL`` from LAPACK directly.\n\n    .. versionadded:: 0.11.0\n\n    Examples\n    --------\n    Given `a`, `b`, and `q` solve for `x`:\n\n    >>> import numpy as np\n    >>> from scipy import linalg\n    >>> a = np.array([[-3, -2, 0], [-1, -1, 3], [3, -5, -1]])\n    >>> b = np.array([[1]])\n    >>> q = np.array([[1],[2],[3]])\n    >>> x = linalg.solve_sylvester(a, b, q)\n    >>> x\n    array([[ 0.0625],\n           [-0.5625],\n           [ 0.6875]])\n    >>> np.allclose(a.dot(x) + x.dot(b), q)\n    True\n\n    '
    (r, u) = schur(a, output='real')
    (s, v) = schur(b.conj().transpose(), output='real')
    f = np.dot(np.dot(u.conj().transpose(), q), v)
    (trsyl,) = get_lapack_funcs(('trsyl',), (r, s, f))
    if trsyl is None:
        raise RuntimeError('LAPACK implementation does not contain a proper Sylvester equation solver (TRSYL)')
    (y, scale, info) = trsyl(r, s, f, tranb='C')
    y = scale * y
    if info < 0:
        raise LinAlgError('Illegal value encountered in the %d term' % (-info,))
    return np.dot(np.dot(u, y), v.conj().transpose())

def solve_continuous_lyapunov(a, q):
    if False:
        i = 10
        return i + 15
    '\n    Solves the continuous Lyapunov equation :math:`AX + XA^H = Q`.\n\n    Uses the Bartels-Stewart algorithm to find :math:`X`.\n\n    Parameters\n    ----------\n    a : array_like\n        A square matrix\n\n    q : array_like\n        Right-hand side square matrix\n\n    Returns\n    -------\n    x : ndarray\n        Solution to the continuous Lyapunov equation\n\n    See Also\n    --------\n    solve_discrete_lyapunov : computes the solution to the discrete-time\n        Lyapunov equation\n    solve_sylvester : computes the solution to the Sylvester equation\n\n    Notes\n    -----\n    The continuous Lyapunov equation is a special form of the Sylvester\n    equation, hence this solver relies on LAPACK routine ?TRSYL.\n\n    .. versionadded:: 0.11.0\n\n    Examples\n    --------\n    Given `a` and `q` solve for `x`:\n\n    >>> import numpy as np\n    >>> from scipy import linalg\n    >>> a = np.array([[-3, -2, 0], [-1, -1, 0], [0, -5, -1]])\n    >>> b = np.array([2, 4, -1])\n    >>> q = np.eye(3)\n    >>> x = linalg.solve_continuous_lyapunov(a, q)\n    >>> x\n    array([[ -0.75  ,   0.875 ,  -3.75  ],\n           [  0.875 ,  -1.375 ,   5.3125],\n           [ -3.75  ,   5.3125, -27.0625]])\n    >>> np.allclose(a.dot(x) + x.dot(a.T), q)\n    True\n    '
    a = np.atleast_2d(_asarray_validated(a, check_finite=True))
    q = np.atleast_2d(_asarray_validated(q, check_finite=True))
    r_or_c = float
    for (ind, _) in enumerate((a, q)):
        if np.iscomplexobj(_):
            r_or_c = complex
        if not np.equal(*_.shape):
            raise ValueError('Matrix {} should be square.'.format('aq'[ind]))
    if a.shape != q.shape:
        raise ValueError('Matrix a and q should have the same shape.')
    (r, u) = schur(a, output='real')
    f = u.conj().T.dot(q.dot(u))
    trsyl = get_lapack_funcs('trsyl', (r, f))
    dtype_string = 'T' if r_or_c == float else 'C'
    (y, scale, info) = trsyl(r, r, f, tranb=dtype_string)
    if info < 0:
        raise ValueError('?TRSYL exited with the internal error "illegal value in argument number {}.". See LAPACK documentation for the ?TRSYL error codes.'.format(-info))
    elif info == 1:
        warnings.warn('Input "a" has an eigenvalue pair whose sum is very close to or exactly zero. The solution is obtained via perturbing the coefficients.', RuntimeWarning)
    y *= scale
    return u.dot(y).dot(u.conj().T)
solve_lyapunov = solve_continuous_lyapunov

def _solve_discrete_lyapunov_direct(a, q):
    if False:
        for i in range(10):
            print('nop')
    '\n    Solves the discrete Lyapunov equation directly.\n\n    This function is called by the `solve_discrete_lyapunov` function with\n    `method=direct`. It is not supposed to be called directly.\n    '
    lhs = kron(a, a.conj())
    lhs = np.eye(lhs.shape[0]) - lhs
    x = solve(lhs, q.flatten())
    return np.reshape(x, q.shape)

def _solve_discrete_lyapunov_bilinear(a, q):
    if False:
        for i in range(10):
            print('nop')
    '\n    Solves the discrete Lyapunov equation using a bilinear transformation.\n\n    This function is called by the `solve_discrete_lyapunov` function with\n    `method=bilinear`. It is not supposed to be called directly.\n    '
    eye = np.eye(a.shape[0])
    aH = a.conj().transpose()
    aHI_inv = inv(aH + eye)
    b = np.dot(aH - eye, aHI_inv)
    c = 2 * np.dot(np.dot(inv(a + eye), q), aHI_inv)
    return solve_lyapunov(b.conj().transpose(), -c)

def solve_discrete_lyapunov(a, q, method=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Solves the discrete Lyapunov equation :math:`AXA^H - X + Q = 0`.\n\n    Parameters\n    ----------\n    a, q : (M, M) array_like\n        Square matrices corresponding to A and Q in the equation\n        above respectively. Must have the same shape.\n\n    method : {'direct', 'bilinear'}, optional\n        Type of solver.\n\n        If not given, chosen to be ``direct`` if ``M`` is less than 10 and\n        ``bilinear`` otherwise.\n\n    Returns\n    -------\n    x : ndarray\n        Solution to the discrete Lyapunov equation\n\n    See Also\n    --------\n    solve_continuous_lyapunov : computes the solution to the continuous-time\n        Lyapunov equation\n\n    Notes\n    -----\n    This section describes the available solvers that can be selected by the\n    'method' parameter. The default method is *direct* if ``M`` is less than 10\n    and ``bilinear`` otherwise.\n\n    Method *direct* uses a direct analytical solution to the discrete Lyapunov\n    equation. The algorithm is given in, for example, [1]_. However, it requires\n    the linear solution of a system with dimension :math:`M^2` so that\n    performance degrades rapidly for even moderately sized matrices.\n\n    Method *bilinear* uses a bilinear transformation to convert the discrete\n    Lyapunov equation to a continuous Lyapunov equation :math:`(BX+XB'=-C)`\n    where :math:`B=(A-I)(A+I)^{-1}` and\n    :math:`C=2(A' + I)^{-1} Q (A + I)^{-1}`. The continuous equation can be\n    efficiently solved since it is a special case of a Sylvester equation.\n    The transformation algorithm is from Popov (1964) as described in [2]_.\n\n    .. versionadded:: 0.11.0\n\n    References\n    ----------\n    .. [1] Hamilton, James D. Time Series Analysis, Princeton: Princeton\n       University Press, 1994.  265.  Print.\n       http://doc1.lbfl.li/aca/FLMF037168.pdf\n    .. [2] Gajic, Z., and M.T.J. Qureshi. 2008.\n       Lyapunov Matrix Equation in System Stability and Control.\n       Dover Books on Engineering Series. Dover Publications.\n\n    Examples\n    --------\n    Given `a` and `q` solve for `x`:\n\n    >>> import numpy as np\n    >>> from scipy import linalg\n    >>> a = np.array([[0.2, 0.5],[0.7, -0.9]])\n    >>> q = np.eye(2)\n    >>> x = linalg.solve_discrete_lyapunov(a, q)\n    >>> x\n    array([[ 0.70872893,  1.43518822],\n           [ 1.43518822, -2.4266315 ]])\n    >>> np.allclose(a.dot(x).dot(a.T)-x, -q)\n    True\n\n    "
    a = np.asarray(a)
    q = np.asarray(q)
    if method is None:
        if a.shape[0] >= 10:
            method = 'bilinear'
        else:
            method = 'direct'
    meth = method.lower()
    if meth == 'direct':
        x = _solve_discrete_lyapunov_direct(a, q)
    elif meth == 'bilinear':
        x = _solve_discrete_lyapunov_bilinear(a, q)
    else:
        raise ValueError('Unknown solver %s' % method)
    return x

def solve_continuous_are(a, b, q, r, e=None, s=None, balanced=True):
    if False:
        for i in range(10):
            print('nop')
    '\n    Solves the continuous-time algebraic Riccati equation (CARE).\n\n    The CARE is defined as\n\n    .. math::\n\n          X A + A^H X - X B R^{-1} B^H X + Q = 0\n\n    The limitations for a solution to exist are :\n\n        * All eigenvalues of :math:`A` on the right half plane, should be\n          controllable.\n\n        * The associated hamiltonian pencil (See Notes), should have\n          eigenvalues sufficiently away from the imaginary axis.\n\n    Moreover, if ``e`` or ``s`` is not precisely ``None``, then the\n    generalized version of CARE\n\n    .. math::\n\n          E^HXA + A^HXE - (E^HXB + S) R^{-1} (B^HXE + S^H) + Q = 0\n\n    is solved. When omitted, ``e`` is assumed to be the identity and ``s``\n    is assumed to be the zero matrix with sizes compatible with ``a`` and\n    ``b``, respectively.\n\n    Parameters\n    ----------\n    a : (M, M) array_like\n        Square matrix\n    b : (M, N) array_like\n        Input\n    q : (M, M) array_like\n        Input\n    r : (N, N) array_like\n        Nonsingular square matrix\n    e : (M, M) array_like, optional\n        Nonsingular square matrix\n    s : (M, N) array_like, optional\n        Input\n    balanced : bool, optional\n        The boolean that indicates whether a balancing step is performed\n        on the data. The default is set to True.\n\n    Returns\n    -------\n    x : (M, M) ndarray\n        Solution to the continuous-time algebraic Riccati equation.\n\n    Raises\n    ------\n    LinAlgError\n        For cases where the stable subspace of the pencil could not be\n        isolated. See Notes section and the references for details.\n\n    See Also\n    --------\n    solve_discrete_are : Solves the discrete-time algebraic Riccati equation\n\n    Notes\n    -----\n    The equation is solved by forming the extended hamiltonian matrix pencil,\n    as described in [1]_, :math:`H - \\lambda J` given by the block matrices ::\n\n        [ A    0    B ]             [ E   0    0 ]\n        [-Q  -A^H  -S ] - \\lambda * [ 0  E^H   0 ]\n        [ S^H B^H   R ]             [ 0   0    0 ]\n\n    and using a QZ decomposition method.\n\n    In this algorithm, the fail conditions are linked to the symmetry\n    of the product :math:`U_2 U_1^{-1}` and condition number of\n    :math:`U_1`. Here, :math:`U` is the 2m-by-m matrix that holds the\n    eigenvectors spanning the stable subspace with 2-m rows and partitioned\n    into two m-row matrices. See [1]_ and [2]_ for more details.\n\n    In order to improve the QZ decomposition accuracy, the pencil goes\n    through a balancing step where the sum of absolute values of\n    :math:`H` and :math:`J` entries (after removing the diagonal entries of\n    the sum) is balanced following the recipe given in [3]_.\n\n    .. versionadded:: 0.11.0\n\n    References\n    ----------\n    .. [1]  P. van Dooren , "A Generalized Eigenvalue Approach For Solving\n       Riccati Equations.", SIAM Journal on Scientific and Statistical\n       Computing, Vol.2(2), :doi:`10.1137/0902010`\n\n    .. [2] A.J. Laub, "A Schur Method for Solving Algebraic Riccati\n       Equations.", Massachusetts Institute of Technology. Laboratory for\n       Information and Decision Systems. LIDS-R ; 859. Available online :\n       http://hdl.handle.net/1721.1/1301\n\n    .. [3] P. Benner, "Symplectic Balancing of Hamiltonian Matrices", 2001,\n       SIAM J. Sci. Comput., 2001, Vol.22(5), :doi:`10.1137/S1064827500367993`\n\n    Examples\n    --------\n    Given `a`, `b`, `q`, and `r` solve for `x`:\n\n    >>> import numpy as np\n    >>> from scipy import linalg\n    >>> a = np.array([[4, 3], [-4.5, -3.5]])\n    >>> b = np.array([[1], [-1]])\n    >>> q = np.array([[9, 6], [6, 4.]])\n    >>> r = 1\n    >>> x = linalg.solve_continuous_are(a, b, q, r)\n    >>> x\n    array([[ 21.72792206,  14.48528137],\n           [ 14.48528137,   9.65685425]])\n    >>> np.allclose(a.T.dot(x) + x.dot(a)-x.dot(b).dot(b.T).dot(x), -q)\n    True\n\n    '
    (a, b, q, r, e, s, m, n, r_or_c, gen_are) = _are_validate_args(a, b, q, r, e, s, 'care')
    H = np.empty((2 * m + n, 2 * m + n), dtype=r_or_c)
    H[:m, :m] = a
    H[:m, m:2 * m] = 0.0
    H[:m, 2 * m:] = b
    H[m:2 * m, :m] = -q
    H[m:2 * m, m:2 * m] = -a.conj().T
    H[m:2 * m, 2 * m:] = 0.0 if s is None else -s
    H[2 * m:, :m] = 0.0 if s is None else s.conj().T
    H[2 * m:, m:2 * m] = b.conj().T
    H[2 * m:, 2 * m:] = r
    if gen_are and e is not None:
        J = block_diag(e, e.conj().T, np.zeros_like(r, dtype=r_or_c))
    else:
        J = block_diag(np.eye(2 * m), np.zeros_like(r, dtype=r_or_c))
    if balanced:
        M = np.abs(H) + np.abs(J)
        M[np.diag_indices_from(M)] = 0.0
        (_, (sca, _)) = matrix_balance(M, separate=1, permute=0)
        if not np.allclose(sca, np.ones_like(sca)):
            sca = np.log2(sca)
            s = np.round((sca[m:2 * m] - sca[:m]) / 2)
            sca = 2 ** np.r_[s, -s, sca[2 * m:]]
            elwisescale = sca[:, None] * np.reciprocal(sca)
            H *= elwisescale
            J *= elwisescale
    (q, r) = qr(H[:, -n:])
    H = q[:, n:].conj().T.dot(H[:, :2 * m])
    J = q[:2 * m, n:].conj().T.dot(J[:2 * m, :2 * m])
    out_str = 'real' if r_or_c == float else 'complex'
    (_, _, _, _, _, u) = ordqz(H, J, sort='lhp', overwrite_a=True, overwrite_b=True, check_finite=False, output=out_str)
    if e is not None:
        (u, _) = qr(np.vstack((e.dot(u[:m, :m]), u[m:, :m])))
    u00 = u[:m, :m]
    u10 = u[m:, :m]
    (up, ul, uu) = lu(u00)
    if 1 / cond(uu) < np.spacing(1.0):
        raise LinAlgError('Failed to find a finite solution.')
    x = solve_triangular(ul.conj().T, solve_triangular(uu.conj().T, u10.conj().T, lower=True), unit_diagonal=True).conj().T.dot(up.conj().T)
    if balanced:
        x *= sca[:m, None] * sca[:m]
    u_sym = u00.conj().T.dot(u10)
    n_u_sym = norm(u_sym, 1)
    u_sym = u_sym - u_sym.conj().T
    sym_threshold = np.max([np.spacing(1000.0), 0.1 * n_u_sym])
    if norm(u_sym, 1) > sym_threshold:
        raise LinAlgError('The associated Hamiltonian pencil has eigenvalues too close to the imaginary axis')
    return (x + x.conj().T) / 2

def solve_discrete_are(a, b, q, r, e=None, s=None, balanced=True):
    if False:
        return 10
    '\n    Solves the discrete-time algebraic Riccati equation (DARE).\n\n    The DARE is defined as\n\n    .. math::\n\n          A^HXA - X - (A^HXB) (R + B^HXB)^{-1} (B^HXA) + Q = 0\n\n    The limitations for a solution to exist are :\n\n        * All eigenvalues of :math:`A` outside the unit disc, should be\n          controllable.\n\n        * The associated symplectic pencil (See Notes), should have\n          eigenvalues sufficiently away from the unit circle.\n\n    Moreover, if ``e`` and ``s`` are not both precisely ``None``, then the\n    generalized version of DARE\n\n    .. math::\n\n          A^HXA - E^HXE - (A^HXB+S) (R+B^HXB)^{-1} (B^HXA+S^H) + Q = 0\n\n    is solved. When omitted, ``e`` is assumed to be the identity and ``s``\n    is assumed to be the zero matrix.\n\n    Parameters\n    ----------\n    a : (M, M) array_like\n        Square matrix\n    b : (M, N) array_like\n        Input\n    q : (M, M) array_like\n        Input\n    r : (N, N) array_like\n        Square matrix\n    e : (M, M) array_like, optional\n        Nonsingular square matrix\n    s : (M, N) array_like, optional\n        Input\n    balanced : bool\n        The boolean that indicates whether a balancing step is performed\n        on the data. The default is set to True.\n\n    Returns\n    -------\n    x : (M, M) ndarray\n        Solution to the discrete algebraic Riccati equation.\n\n    Raises\n    ------\n    LinAlgError\n        For cases where the stable subspace of the pencil could not be\n        isolated. See Notes section and the references for details.\n\n    See Also\n    --------\n    solve_continuous_are : Solves the continuous algebraic Riccati equation\n\n    Notes\n    -----\n    The equation is solved by forming the extended symplectic matrix pencil,\n    as described in [1]_, :math:`H - \\lambda J` given by the block matrices ::\n\n           [  A   0   B ]             [ E   0   B ]\n           [ -Q  E^H -S ] - \\lambda * [ 0  A^H  0 ]\n           [ S^H  0   R ]             [ 0 -B^H  0 ]\n\n    and using a QZ decomposition method.\n\n    In this algorithm, the fail conditions are linked to the symmetry\n    of the product :math:`U_2 U_1^{-1}` and condition number of\n    :math:`U_1`. Here, :math:`U` is the 2m-by-m matrix that holds the\n    eigenvectors spanning the stable subspace with 2-m rows and partitioned\n    into two m-row matrices. See [1]_ and [2]_ for more details.\n\n    In order to improve the QZ decomposition accuracy, the pencil goes\n    through a balancing step where the sum of absolute values of\n    :math:`H` and :math:`J` rows/cols (after removing the diagonal entries)\n    is balanced following the recipe given in [3]_. If the data has small\n    numerical noise, balancing may amplify their effects and some clean up\n    is required.\n\n    .. versionadded:: 0.11.0\n\n    References\n    ----------\n    .. [1]  P. van Dooren , "A Generalized Eigenvalue Approach For Solving\n       Riccati Equations.", SIAM Journal on Scientific and Statistical\n       Computing, Vol.2(2), :doi:`10.1137/0902010`\n\n    .. [2] A.J. Laub, "A Schur Method for Solving Algebraic Riccati\n       Equations.", Massachusetts Institute of Technology. Laboratory for\n       Information and Decision Systems. LIDS-R ; 859. Available online :\n       http://hdl.handle.net/1721.1/1301\n\n    .. [3] P. Benner, "Symplectic Balancing of Hamiltonian Matrices", 2001,\n       SIAM J. Sci. Comput., 2001, Vol.22(5), :doi:`10.1137/S1064827500367993`\n\n    Examples\n    --------\n    Given `a`, `b`, `q`, and `r` solve for `x`:\n\n    >>> import numpy as np\n    >>> from scipy import linalg as la\n    >>> a = np.array([[0, 1], [0, -1]])\n    >>> b = np.array([[1, 0], [2, 1]])\n    >>> q = np.array([[-4, -4], [-4, 7]])\n    >>> r = np.array([[9, 3], [3, 1]])\n    >>> x = la.solve_discrete_are(a, b, q, r)\n    >>> x\n    array([[-4., -4.],\n           [-4.,  7.]])\n    >>> R = la.solve(r + b.T.dot(x).dot(b), b.T.dot(x).dot(a))\n    >>> np.allclose(a.T.dot(x).dot(a) - x - a.T.dot(x).dot(b).dot(R), -q)\n    True\n\n    '
    (a, b, q, r, e, s, m, n, r_or_c, gen_are) = _are_validate_args(a, b, q, r, e, s, 'dare')
    H = np.zeros((2 * m + n, 2 * m + n), dtype=r_or_c)
    H[:m, :m] = a
    H[:m, 2 * m:] = b
    H[m:2 * m, :m] = -q
    H[m:2 * m, m:2 * m] = np.eye(m) if e is None else e.conj().T
    H[m:2 * m, 2 * m:] = 0.0 if s is None else -s
    H[2 * m:, :m] = 0.0 if s is None else s.conj().T
    H[2 * m:, 2 * m:] = r
    J = np.zeros_like(H, dtype=r_or_c)
    J[:m, :m] = np.eye(m) if e is None else e
    J[m:2 * m, m:2 * m] = a.conj().T
    J[2 * m:, m:2 * m] = -b.conj().T
    if balanced:
        M = np.abs(H) + np.abs(J)
        M[np.diag_indices_from(M)] = 0.0
        (_, (sca, _)) = matrix_balance(M, separate=1, permute=0)
        if not np.allclose(sca, np.ones_like(sca)):
            sca = np.log2(sca)
            s = np.round((sca[m:2 * m] - sca[:m]) / 2)
            sca = 2 ** np.r_[s, -s, sca[2 * m:]]
            elwisescale = sca[:, None] * np.reciprocal(sca)
            H *= elwisescale
            J *= elwisescale
    (q_of_qr, _) = qr(H[:, -n:])
    H = q_of_qr[:, n:].conj().T.dot(H[:, :2 * m])
    J = q_of_qr[:, n:].conj().T.dot(J[:, :2 * m])
    out_str = 'real' if r_or_c == float else 'complex'
    (_, _, _, _, _, u) = ordqz(H, J, sort='iuc', overwrite_a=True, overwrite_b=True, check_finite=False, output=out_str)
    if e is not None:
        (u, _) = qr(np.vstack((e.dot(u[:m, :m]), u[m:, :m])))
    u00 = u[:m, :m]
    u10 = u[m:, :m]
    (up, ul, uu) = lu(u00)
    if 1 / cond(uu) < np.spacing(1.0):
        raise LinAlgError('Failed to find a finite solution.')
    x = solve_triangular(ul.conj().T, solve_triangular(uu.conj().T, u10.conj().T, lower=True), unit_diagonal=True).conj().T.dot(up.conj().T)
    if balanced:
        x *= sca[:m, None] * sca[:m]
    u_sym = u00.conj().T.dot(u10)
    n_u_sym = norm(u_sym, 1)
    u_sym = u_sym - u_sym.conj().T
    sym_threshold = np.max([np.spacing(1000.0), 0.1 * n_u_sym])
    if norm(u_sym, 1) > sym_threshold:
        raise LinAlgError('The associated symplectic pencil has eigenvalues too close to the unit circle')
    return (x + x.conj().T) / 2

def _are_validate_args(a, b, q, r, e, s, eq_type='care'):
    if False:
        for i in range(10):
            print('nop')
    "\n    A helper function to validate the arguments supplied to the\n    Riccati equation solvers. Any discrepancy found in the input\n    matrices leads to a ``ValueError`` exception.\n\n    Essentially, it performs:\n\n        - a check whether the input is free of NaN and Infs\n        - a pass for the data through ``numpy.atleast_2d()``\n        - squareness check of the relevant arrays\n        - shape consistency check of the arrays\n        - singularity check of the relevant arrays\n        - symmetricity check of the relevant matrices\n        - a check whether the regular or the generalized version is asked.\n\n    This function is used by ``solve_continuous_are`` and\n    ``solve_discrete_are``.\n\n    Parameters\n    ----------\n    a, b, q, r, e, s : array_like\n        Input data\n    eq_type : str\n        Accepted arguments are 'care' and 'dare'.\n\n    Returns\n    -------\n    a, b, q, r, e, s : ndarray\n        Regularized input data\n    m, n : int\n        shape of the problem\n    r_or_c : type\n        Data type of the problem, returns float or complex\n    gen_or_not : bool\n        Type of the equation, True for generalized and False for regular ARE.\n\n    "
    if eq_type.lower() not in ('dare', 'care'):
        raise ValueError("Equation type unknown. Only 'care' and 'dare' is understood")
    a = np.atleast_2d(_asarray_validated(a, check_finite=True))
    b = np.atleast_2d(_asarray_validated(b, check_finite=True))
    q = np.atleast_2d(_asarray_validated(q, check_finite=True))
    r = np.atleast_2d(_asarray_validated(r, check_finite=True))
    r_or_c = complex if np.iscomplexobj(b) else float
    for (ind, mat) in enumerate((a, q, r)):
        if np.iscomplexobj(mat):
            r_or_c = complex
        if not np.equal(*mat.shape):
            raise ValueError('Matrix {} should be square.'.format('aqr'[ind]))
    (m, n) = b.shape
    if m != a.shape[0]:
        raise ValueError('Matrix a and b should have the same number of rows.')
    if m != q.shape[0]:
        raise ValueError('Matrix a and q should have the same shape.')
    if n != r.shape[0]:
        raise ValueError('Matrix b and r should have the same number of cols.')
    for (ind, mat) in enumerate((q, r)):
        if norm(mat - mat.conj().T, 1) > np.spacing(norm(mat, 1)) * 100:
            raise ValueError('Matrix {} should be symmetric/hermitian.'.format('qr'[ind]))
    if eq_type == 'care':
        min_sv = svd(r, compute_uv=False)[-1]
        if min_sv == 0.0 or min_sv < np.spacing(1.0) * norm(r, 1):
            raise ValueError('Matrix r is numerically singular.')
    generalized_case = e is not None or s is not None
    if generalized_case:
        if e is not None:
            e = np.atleast_2d(_asarray_validated(e, check_finite=True))
            if not np.equal(*e.shape):
                raise ValueError('Matrix e should be square.')
            if m != e.shape[0]:
                raise ValueError('Matrix a and e should have the same shape.')
            min_sv = svd(e, compute_uv=False)[-1]
            if min_sv == 0.0 or min_sv < np.spacing(1.0) * norm(e, 1):
                raise ValueError('Matrix e is numerically singular.')
            if np.iscomplexobj(e):
                r_or_c = complex
        if s is not None:
            s = np.atleast_2d(_asarray_validated(s, check_finite=True))
            if s.shape != b.shape:
                raise ValueError('Matrix b and s should have the same shape.')
            if np.iscomplexobj(s):
                r_or_c = complex
    return (a, b, q, r, e, s, m, n, r_or_c, generalized_case)