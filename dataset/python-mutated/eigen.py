from types import FunctionType
from collections import Counter
from mpmath import mp, workprec
from mpmath.libmp.libmpf import prec_to_dps
from sympy.core.sorting import default_sort_key
from sympy.core.evalf import DEFAULT_MAXPREC, PrecisionExhausted
from sympy.core.logic import fuzzy_and, fuzzy_or
from sympy.core.numbers import Float
from sympy.core.sympify import _sympify
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.polys import roots, CRootOf, ZZ, QQ, EX
from sympy.polys.matrices import DomainMatrix
from sympy.polys.matrices.eigen import dom_eigenvects, dom_eigenvects_to_sympy
from sympy.polys.polytools import gcd
from .common import MatrixError, NonSquareMatrixError
from .determinant import _find_reasonable_pivot
from .utilities import _iszero, _simplify

def _eigenvals_eigenvects_mpmath(M):
    if False:
        i = 10
        return i + 15
    norm2 = lambda v: mp.sqrt(sum((i ** 2 for i in v)))
    v1 = None
    prec = max([x._prec for x in M.atoms(Float)])
    eps = 2 ** (-prec)
    while prec < DEFAULT_MAXPREC:
        with workprec(prec):
            A = mp.matrix(M.evalf(n=prec_to_dps(prec)))
            (E, ER) = mp.eig(A)
            v2 = norm2([i for e in E for i in (mp.re(e), mp.im(e))])
            if v1 is not None and mp.fabs(v1 - v2) < eps:
                return (E, ER)
            v1 = v2
        prec *= 2
    raise PrecisionExhausted

def _eigenvals_mpmath(M, multiple=False):
    if False:
        i = 10
        return i + 15
    'Compute eigenvalues using mpmath'
    (E, _) = _eigenvals_eigenvects_mpmath(M)
    result = [_sympify(x) for x in E]
    if multiple:
        return result
    return dict(Counter(result))

def _eigenvects_mpmath(M):
    if False:
        i = 10
        return i + 15
    (E, ER) = _eigenvals_eigenvects_mpmath(M)
    result = []
    for i in range(M.rows):
        eigenval = _sympify(E[i])
        eigenvect = _sympify(ER[:, i])
        result.append((eigenval, 1, [eigenvect]))
    return result

def _eigenvals(M, error_when_incomplete=True, *, simplify=False, multiple=False, rational=False, **flags):
    if False:
        for i in range(10):
            print('nop')
    "Compute eigenvalues of the matrix.\n\n    Parameters\n    ==========\n\n    error_when_incomplete : bool, optional\n        If it is set to ``True``, it will raise an error if not all\n        eigenvalues are computed. This is caused by ``roots`` not returning\n        a full list of eigenvalues.\n\n    simplify : bool or function, optional\n        If it is set to ``True``, it attempts to return the most\n        simplified form of expressions returned by applying default\n        simplification method in every routine.\n\n        If it is set to ``False``, it will skip simplification in this\n        particular routine to save computation resources.\n\n        If a function is passed to, it will attempt to apply\n        the particular function as simplification method.\n\n    rational : bool, optional\n        If it is set to ``True``, every floating point numbers would be\n        replaced with rationals before computation. It can solve some\n        issues of ``roots`` routine not working well with floats.\n\n    multiple : bool, optional\n        If it is set to ``True``, the result will be in the form of a\n        list.\n\n        If it is set to ``False``, the result will be in the form of a\n        dictionary.\n\n    Returns\n    =======\n\n    eigs : list or dict\n        Eigenvalues of a matrix. The return format would be specified by\n        the key ``multiple``.\n\n    Raises\n    ======\n\n    MatrixError\n        If not enough roots had got computed.\n\n    NonSquareMatrixError\n        If attempted to compute eigenvalues from a non-square matrix.\n\n    Examples\n    ========\n\n    >>> from sympy import Matrix\n    >>> M = Matrix(3, 3, [0, 1, 1, 1, 0, 0, 1, 1, 1])\n    >>> M.eigenvals()\n    {-1: 1, 0: 1, 2: 1}\n\n    See Also\n    ========\n\n    MatrixDeterminant.charpoly\n    eigenvects\n\n    Notes\n    =====\n\n    Eigenvalues of a matrix $A$ can be computed by solving a matrix\n    equation $\\det(A - \\lambda I) = 0$\n\n    It's not always possible to return radical solutions for\n    eigenvalues for matrices larger than $4, 4$ shape due to\n    Abel-Ruffini theorem.\n\n    If there is no radical solution is found for the eigenvalue,\n    it may return eigenvalues in the form of\n    :class:`sympy.polys.rootoftools.ComplexRootOf`.\n    "
    if not M:
        if multiple:
            return []
        return {}
    if not M.is_square:
        raise NonSquareMatrixError('{} must be a square matrix.'.format(M))
    if M._rep.domain not in (ZZ, QQ):
        if all((x.is_number for x in M)) and M.has(Float):
            return _eigenvals_mpmath(M, multiple=multiple)
    if rational:
        from sympy.simplify import nsimplify
        M = M.applyfunc(lambda x: nsimplify(x, rational=True) if x.has(Float) else x)
    if multiple:
        return _eigenvals_list(M, error_when_incomplete=error_when_incomplete, simplify=simplify, **flags)
    return _eigenvals_dict(M, error_when_incomplete=error_when_incomplete, simplify=simplify, **flags)
eigenvals_error_message = 'It is not always possible to express the eigenvalues of a matrix ' + 'of size 5x5 or higher in radicals. ' + 'We have CRootOf, but domains other than the rationals are not ' + 'currently supported. ' + 'If there are no symbols in the matrix, ' + 'it should still be possible to compute numeric approximations ' + 'of the eigenvalues using ' + 'M.evalf().eigenvals() or M.charpoly().nroots().'

def _eigenvals_list(M, error_when_incomplete=True, simplify=False, **flags):
    if False:
        print('Hello World!')
    iblocks = M.strongly_connected_components()
    all_eigs = []
    is_dom = M._rep.domain in (ZZ, QQ)
    for b in iblocks:
        if is_dom and len(b) == 1:
            index = b[0]
            val = M[index, index]
            all_eigs.append(val)
            continue
        block = M[b, b]
        if isinstance(simplify, FunctionType):
            charpoly = block.charpoly(simplify=simplify)
        else:
            charpoly = block.charpoly()
        eigs = roots(charpoly, multiple=True, **flags)
        if len(eigs) != block.rows:
            try:
                eigs = charpoly.all_roots(multiple=True)
            except NotImplementedError:
                if error_when_incomplete:
                    raise MatrixError(eigenvals_error_message)
                else:
                    eigs = []
        all_eigs += eigs
    if not simplify:
        return all_eigs
    if not isinstance(simplify, FunctionType):
        simplify = _simplify
    return [simplify(value) for value in all_eigs]

def _eigenvals_dict(M, error_when_incomplete=True, simplify=False, **flags):
    if False:
        while True:
            i = 10
    iblocks = M.strongly_connected_components()
    all_eigs = {}
    is_dom = M._rep.domain in (ZZ, QQ)
    for b in iblocks:
        if is_dom and len(b) == 1:
            index = b[0]
            val = M[index, index]
            all_eigs[val] = all_eigs.get(val, 0) + 1
            continue
        block = M[b, b]
        if isinstance(simplify, FunctionType):
            charpoly = block.charpoly(simplify=simplify)
        else:
            charpoly = block.charpoly()
        eigs = roots(charpoly, multiple=False, **flags)
        if sum(eigs.values()) != block.rows:
            try:
                eigs = dict(charpoly.all_roots(multiple=False))
            except NotImplementedError:
                if error_when_incomplete:
                    raise MatrixError(eigenvals_error_message)
                else:
                    eigs = {}
        for (k, v) in eigs.items():
            if k in all_eigs:
                all_eigs[k] += v
            else:
                all_eigs[k] = v
    if not simplify:
        return all_eigs
    if not isinstance(simplify, FunctionType):
        simplify = _simplify
    return {simplify(key): value for (key, value) in all_eigs.items()}

def _eigenspace(M, eigenval, iszerofunc=_iszero, simplify=False):
    if False:
        for i in range(10):
            print('nop')
    'Get a basis for the eigenspace for a particular eigenvalue'
    m = M - M.eye(M.rows) * eigenval
    ret = m.nullspace(iszerofunc=iszerofunc)
    if len(ret) == 0 and simplify:
        ret = m.nullspace(iszerofunc=iszerofunc, simplify=True)
    if len(ret) == 0:
        raise NotImplementedError("Can't evaluate eigenvector for eigenvalue {}".format(eigenval))
    return ret

def _eigenvects_DOM(M, **kwargs):
    if False:
        print('Hello World!')
    DOM = DomainMatrix.from_Matrix(M, field=True, extension=True)
    DOM = DOM.to_dense()
    if DOM.domain != EX:
        (rational, algebraic) = dom_eigenvects(DOM)
        eigenvects = dom_eigenvects_to_sympy(rational, algebraic, M.__class__, **kwargs)
        eigenvects = sorted(eigenvects, key=lambda x: default_sort_key(x[0]))
        return eigenvects
    return None

def _eigenvects_sympy(M, iszerofunc, simplify=True, **flags):
    if False:
        return 10
    eigenvals = M.eigenvals(rational=False, **flags)
    for x in eigenvals:
        if x.has(CRootOf):
            raise MatrixError('Eigenvector computation is not implemented if the matrix have eigenvalues in CRootOf form')
    eigenvals = sorted(eigenvals.items(), key=default_sort_key)
    ret = []
    for (val, mult) in eigenvals:
        vects = _eigenspace(M, val, iszerofunc=iszerofunc, simplify=simplify)
        ret.append((val, mult, vects))
    return ret

def _eigenvects(M, error_when_incomplete=True, iszerofunc=_iszero, *, chop=False, **flags):
    if False:
        print('Hello World!')
    "Compute eigenvectors of the matrix.\n\n    Parameters\n    ==========\n\n    error_when_incomplete : bool, optional\n        Raise an error when not all eigenvalues are computed. This is\n        caused by ``roots`` not returning a full list of eigenvalues.\n\n    iszerofunc : function, optional\n        Specifies a zero testing function to be used in ``rref``.\n\n        Default value is ``_iszero``, which uses SymPy's naive and fast\n        default assumption handler.\n\n        It can also accept any user-specified zero testing function, if it\n        is formatted as a function which accepts a single symbolic argument\n        and returns ``True`` if it is tested as zero and ``False`` if it\n        is tested as non-zero, and ``None`` if it is undecidable.\n\n    simplify : bool or function, optional\n        If ``True``, ``as_content_primitive()`` will be used to tidy up\n        normalization artifacts.\n\n        It will also be used by the ``nullspace`` routine.\n\n    chop : bool or positive number, optional\n        If the matrix contains any Floats, they will be changed to Rationals\n        for computation purposes, but the answers will be returned after\n        being evaluated with evalf. The ``chop`` flag is passed to ``evalf``.\n        When ``chop=True`` a default precision will be used; a number will\n        be interpreted as the desired level of precision.\n\n    Returns\n    =======\n\n    ret : [(eigenval, multiplicity, eigenspace), ...]\n        A ragged list containing tuples of data obtained by ``eigenvals``\n        and ``nullspace``.\n\n        ``eigenspace`` is a list containing the ``eigenvector`` for each\n        eigenvalue.\n\n        ``eigenvector`` is a vector in the form of a ``Matrix``. e.g.\n        a vector of length 3 is returned as ``Matrix([a_1, a_2, a_3])``.\n\n    Raises\n    ======\n\n    NotImplementedError\n        If failed to compute nullspace.\n\n    Examples\n    ========\n\n    >>> from sympy import Matrix\n    >>> M = Matrix(3, 3, [0, 1, 1, 1, 0, 0, 1, 1, 1])\n    >>> M.eigenvects()\n    [(-1, 1, [Matrix([\n    [-1],\n    [ 1],\n    [ 0]])]), (0, 1, [Matrix([\n    [ 0],\n    [-1],\n    [ 1]])]), (2, 1, [Matrix([\n    [2/3],\n    [1/3],\n    [  1]])])]\n\n    See Also\n    ========\n\n    eigenvals\n    MatrixSubspaces.nullspace\n    "
    simplify = flags.get('simplify', True)
    primitive = flags.get('simplify', False)
    flags.pop('simplify', None)
    flags.pop('multiple', None)
    if not isinstance(simplify, FunctionType):
        simpfunc = _simplify if simplify else lambda x: x
    has_floats = M.has(Float)
    if has_floats:
        if all((x.is_number for x in M)):
            return _eigenvects_mpmath(M)
        from sympy.simplify import nsimplify
        M = M.applyfunc(lambda x: nsimplify(x, rational=True))
    ret = _eigenvects_DOM(M)
    if ret is None:
        ret = _eigenvects_sympy(M, iszerofunc, simplify=simplify, **flags)
    if primitive:

        def denom_clean(l):
            if False:
                i = 10
                return i + 15
            return [(v / gcd(list(v))).applyfunc(simpfunc) for v in l]
        ret = [(val, mult, denom_clean(es)) for (val, mult, es) in ret]
    if has_floats:
        ret = [(val.evalf(chop=chop), mult, [v.evalf(chop=chop) for v in es]) for (val, mult, es) in ret]
    return ret

def _is_diagonalizable_with_eigen(M, reals_only=False):
    if False:
        i = 10
        return i + 15
    'See _is_diagonalizable. This function returns the bool along with the\n    eigenvectors to avoid calculating them again in functions like\n    ``diagonalize``.'
    if not M.is_square:
        return (False, [])
    eigenvecs = M.eigenvects(simplify=True)
    for (val, mult, basis) in eigenvecs:
        if reals_only and (not val.is_real):
            return (False, eigenvecs)
        if mult != len(basis):
            return (False, eigenvecs)
    return (True, eigenvecs)

def _is_diagonalizable(M, reals_only=False, **kwargs):
    if False:
        return 10
    'Returns ``True`` if a matrix is diagonalizable.\n\n    Parameters\n    ==========\n\n    reals_only : bool, optional\n        If ``True``, it tests whether the matrix can be diagonalized\n        to contain only real numbers on the diagonal.\n\n\n        If ``False``, it tests whether the matrix can be diagonalized\n        at all, even with numbers that may not be real.\n\n    Examples\n    ========\n\n    Example of a diagonalizable matrix:\n\n    >>> from sympy import Matrix\n    >>> M = Matrix([[1, 2, 0], [0, 3, 0], [2, -4, 2]])\n    >>> M.is_diagonalizable()\n    True\n\n    Example of a non-diagonalizable matrix:\n\n    >>> M = Matrix([[0, 1], [0, 0]])\n    >>> M.is_diagonalizable()\n    False\n\n    Example of a matrix that is diagonalized in terms of non-real entries:\n\n    >>> M = Matrix([[0, 1], [-1, 0]])\n    >>> M.is_diagonalizable(reals_only=False)\n    True\n    >>> M.is_diagonalizable(reals_only=True)\n    False\n\n    See Also\n    ========\n\n    sympy.matrices.common.MatrixCommon.is_diagonal\n    diagonalize\n    '
    if not M.is_square:
        return False
    if all((e.is_real for e in M)) and M.is_symmetric():
        return True
    if all((e.is_complex for e in M)) and M.is_hermitian:
        return True
    return _is_diagonalizable_with_eigen(M, reals_only=reals_only)[0]

def _householder_vector(x):
    if False:
        for i in range(10):
            print('nop')
    if not x.cols == 1:
        raise ValueError('Input must be a column matrix')
    v = x.copy()
    v_plus = x.copy()
    v_minus = x.copy()
    q = x[0, 0] / abs(x[0, 0])
    norm_x = x.norm()
    v_plus[0, 0] = x[0, 0] + q * norm_x
    v_minus[0, 0] = x[0, 0] - q * norm_x
    if x[1:, 0].norm() == 0:
        bet = 0
        v[0, 0] = 1
    else:
        if v_plus.norm() <= v_minus.norm():
            v = v_plus
        else:
            v = v_minus
        v = v / v[0]
        bet = 2 / v.norm() ** 2
    return (v, bet)

def _bidiagonal_decmp_hholder(M):
    if False:
        for i in range(10):
            print('nop')
    m = M.rows
    n = M.cols
    A = M.as_mutable()
    (U, V) = (A.eye(m), A.eye(n))
    for i in range(min(m, n)):
        (v, bet) = _householder_vector(A[i:, i])
        hh_mat = A.eye(m - i) - bet * v * v.H
        A[i:, i:] = hh_mat * A[i:, i:]
        temp = A.eye(m)
        temp[i:, i:] = hh_mat
        U = U * temp
        if i + 1 <= n - 2:
            (v, bet) = _householder_vector(A[i, i + 1:].T)
            hh_mat = A.eye(n - i - 1) - bet * v * v.H
            A[i:, i + 1:] = A[i:, i + 1:] * hh_mat
            temp = A.eye(n)
            temp[i + 1:, i + 1:] = hh_mat
            V = temp * V
    return (U, A, V)

def _eval_bidiag_hholder(M):
    if False:
        return 10
    m = M.rows
    n = M.cols
    A = M.as_mutable()
    for i in range(min(m, n)):
        (v, bet) = _householder_vector(A[i:, i])
        hh_mat = A.eye(m - i) - bet * v * v.H
        A[i:, i:] = hh_mat * A[i:, i:]
        if i + 1 <= n - 2:
            (v, bet) = _householder_vector(A[i, i + 1:].T)
            hh_mat = A.eye(n - i - 1) - bet * v * v.H
            A[i:, i + 1:] = A[i:, i + 1:] * hh_mat
    return A

def _bidiagonal_decomposition(M, upper=True):
    if False:
        i = 10
        return i + 15
    '\n    Returns $(U,B,V.H)$ for\n\n    $$A = UBV^{H}$$\n\n    where $A$ is the input matrix, and $B$ is its Bidiagonalized form\n\n    Note: Bidiagonal Computation can hang for symbolic matrices.\n\n    Parameters\n    ==========\n\n    upper : bool. Whether to do upper bidiagnalization or lower.\n                True for upper and False for lower.\n\n    References\n    ==========\n\n    .. [1] Algorithm 5.4.2, Matrix computations by Golub and Van Loan, 4th edition\n    .. [2] Complex Matrix Bidiagonalization, https://github.com/vslobody/Householder-Bidiagonalization\n\n    '
    if not isinstance(upper, bool):
        raise ValueError('upper must be a boolean')
    if upper:
        return _bidiagonal_decmp_hholder(M)
    X = _bidiagonal_decmp_hholder(M.H)
    return (X[2].H, X[1].H, X[0].H)

def _bidiagonalize(M, upper=True):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns $B$, the Bidiagonalized form of the input matrix.\n\n    Note: Bidiagonal Computation can hang for symbolic matrices.\n\n    Parameters\n    ==========\n\n    upper : bool. Whether to do upper bidiagnalization or lower.\n                True for upper and False for lower.\n\n    References\n    ==========\n\n    .. [1] Algorithm 5.4.2, Matrix computations by Golub and Van Loan, 4th edition\n    .. [2] Complex Matrix Bidiagonalization : https://github.com/vslobody/Householder-Bidiagonalization\n\n    '
    if not isinstance(upper, bool):
        raise ValueError('upper must be a boolean')
    if upper:
        return _eval_bidiag_hholder(M)
    return _eval_bidiag_hholder(M.H).H

def _diagonalize(M, reals_only=False, sort=False, normalize=False):
    if False:
        return 10
    '\n    Return (P, D), where D is diagonal and\n\n        D = P^-1 * M * P\n\n    where M is current matrix.\n\n    Parameters\n    ==========\n\n    reals_only : bool. Whether to throw an error if complex numbers are need\n                    to diagonalize. (Default: False)\n\n    sort : bool. Sort the eigenvalues along the diagonal. (Default: False)\n\n    normalize : bool. If True, normalize the columns of P. (Default: False)\n\n    Examples\n    ========\n\n    >>> from sympy import Matrix\n    >>> M = Matrix(3, 3, [1, 2, 0, 0, 3, 0, 2, -4, 2])\n    >>> M\n    Matrix([\n    [1,  2, 0],\n    [0,  3, 0],\n    [2, -4, 2]])\n    >>> (P, D) = M.diagonalize()\n    >>> D\n    Matrix([\n    [1, 0, 0],\n    [0, 2, 0],\n    [0, 0, 3]])\n    >>> P\n    Matrix([\n    [-1, 0, -1],\n    [ 0, 0, -1],\n    [ 2, 1,  2]])\n    >>> P.inv() * M * P\n    Matrix([\n    [1, 0, 0],\n    [0, 2, 0],\n    [0, 0, 3]])\n\n    See Also\n    ========\n\n    sympy.matrices.common.MatrixCommon.is_diagonal\n    is_diagonalizable\n    '
    if not M.is_square:
        raise NonSquareMatrixError()
    (is_diagonalizable, eigenvecs) = _is_diagonalizable_with_eigen(M, reals_only=reals_only)
    if not is_diagonalizable:
        raise MatrixError('Matrix is not diagonalizable')
    if sort:
        eigenvecs = sorted(eigenvecs, key=default_sort_key)
    (p_cols, diag) = ([], [])
    for (val, mult, basis) in eigenvecs:
        diag += [val] * mult
        p_cols += basis
    if normalize:
        p_cols = [v / v.norm() for v in p_cols]
    return (M.hstack(*p_cols), M.diag(*diag))

def _fuzzy_positive_definite(M):
    if False:
        return 10
    positive_diagonals = M._has_positive_diagonals()
    if positive_diagonals is False:
        return False
    if positive_diagonals and M.is_strongly_diagonally_dominant:
        return True
    return None

def _fuzzy_positive_semidefinite(M):
    if False:
        print('Hello World!')
    nonnegative_diagonals = M._has_nonnegative_diagonals()
    if nonnegative_diagonals is False:
        return False
    if nonnegative_diagonals and M.is_weakly_diagonally_dominant:
        return True
    return None

def _is_positive_definite(M):
    if False:
        while True:
            i = 10
    if not M.is_hermitian:
        if not M.is_square:
            return False
        M = M + M.H
    fuzzy = _fuzzy_positive_definite(M)
    if fuzzy is not None:
        return fuzzy
    return _is_positive_definite_GE(M)

def _is_positive_semidefinite(M):
    if False:
        while True:
            i = 10
    if not M.is_hermitian:
        if not M.is_square:
            return False
        M = M + M.H
    fuzzy = _fuzzy_positive_semidefinite(M)
    if fuzzy is not None:
        return fuzzy
    return _is_positive_semidefinite_cholesky(M)

def _is_negative_definite(M):
    if False:
        return 10
    return _is_positive_definite(-M)

def _is_negative_semidefinite(M):
    if False:
        i = 10
        return i + 15
    return _is_positive_semidefinite(-M)

def _is_indefinite(M):
    if False:
        for i in range(10):
            print('nop')
    if M.is_hermitian:
        eigen = M.eigenvals()
        args1 = [x.is_positive for x in eigen.keys()]
        any_positive = fuzzy_or(args1)
        args2 = [x.is_negative for x in eigen.keys()]
        any_negative = fuzzy_or(args2)
        return fuzzy_and([any_positive, any_negative])
    elif M.is_square:
        return (M + M.H).is_indefinite
    return False

def _is_positive_definite_GE(M):
    if False:
        for i in range(10):
            print('nop')
    'A division-free gaussian elimination method for testing\n    positive-definiteness.'
    M = M.as_mutable()
    size = M.rows
    for i in range(size):
        is_positive = M[i, i].is_positive
        if is_positive is not True:
            return is_positive
        for j in range(i + 1, size):
            M[j, i + 1:] = M[i, i] * M[j, i + 1:] - M[j, i] * M[i, i + 1:]
    return True

def _is_positive_semidefinite_cholesky(M):
    if False:
        for i in range(10):
            print('nop')
    'Uses Cholesky factorization with complete pivoting\n\n    References\n    ==========\n\n    .. [1] http://eprints.ma.man.ac.uk/1199/1/covered/MIMS_ep2008_116.pdf\n\n    .. [2] https://www.value-at-risk.net/cholesky-factorization/\n    '
    M = M.as_mutable()
    for k in range(M.rows):
        diags = [M[i, i] for i in range(k, M.rows)]
        (pivot, pivot_val, nonzero, _) = _find_reasonable_pivot(diags)
        if nonzero:
            return None
        if pivot is None:
            for i in range(k + 1, M.rows):
                for j in range(k, M.cols):
                    iszero = M[i, j].is_zero
                    if iszero is None:
                        return None
                    elif iszero is False:
                        return False
            return True
        if M[k, k].is_negative or pivot_val.is_negative:
            return False
        elif not (M[k, k].is_nonnegative and pivot_val.is_nonnegative):
            return None
        if pivot > 0:
            M.col_swap(k, k + pivot)
            M.row_swap(k, k + pivot)
        M[k, k] = sqrt(M[k, k])
        M[k, k + 1:] /= M[k, k]
        M[k + 1:, k + 1:] -= M[k, k + 1:].H * M[k, k + 1:]
    return M[-1, -1].is_nonnegative
_doc_positive_definite = 'Finds out the definiteness of a matrix.\n\n    Explanation\n    ===========\n\n    A square real matrix $A$ is:\n\n    - A positive definite matrix if $x^T A x > 0$\n      for all non-zero real vectors $x$.\n    - A positive semidefinite matrix if $x^T A x \\geq 0$\n      for all non-zero real vectors $x$.\n    - A negative definite matrix if $x^T A x < 0$\n      for all non-zero real vectors $x$.\n    - A negative semidefinite matrix if $x^T A x \\leq 0$\n      for all non-zero real vectors $x$.\n    - An indefinite matrix if there exists non-zero real vectors\n      $x, y$ with $x^T A x > 0 > y^T A y$.\n\n    A square complex matrix $A$ is:\n\n    - A positive definite matrix if $\\text{re}(x^H A x) > 0$\n      for all non-zero complex vectors $x$.\n    - A positive semidefinite matrix if $\\text{re}(x^H A x) \\geq 0$\n      for all non-zero complex vectors $x$.\n    - A negative definite matrix if $\\text{re}(x^H A x) < 0$\n      for all non-zero complex vectors $x$.\n    - A negative semidefinite matrix if $\\text{re}(x^H A x) \\leq 0$\n      for all non-zero complex vectors $x$.\n    - An indefinite matrix if there exists non-zero complex vectors\n      $x, y$ with $\\text{re}(x^H A x) > 0 > \\text{re}(y^H A y)$.\n\n    A matrix need not be symmetric or hermitian to be positive definite.\n\n    - A real non-symmetric matrix is positive definite if and only if\n      $\\frac{A + A^T}{2}$ is positive definite.\n    - A complex non-hermitian matrix is positive definite if and only if\n      $\\frac{A + A^H}{2}$ is positive definite.\n\n    And this extension can apply for all the definitions above.\n\n    However, for complex cases, you can restrict the definition of\n    $\\text{re}(x^H A x) > 0$ to $x^H A x > 0$ and require the matrix\n    to be hermitian.\n    But we do not present this restriction for computation because you\n    can check ``M.is_hermitian`` independently with this and use\n    the same procedure.\n\n    Examples\n    ========\n\n    An example of symmetric positive definite matrix:\n\n    .. plot::\n        :context: reset\n        :format: doctest\n        :include-source: True\n\n        >>> from sympy import Matrix, symbols\n        >>> from sympy.plotting import plot3d\n        >>> a, b = symbols(\'a b\')\n        >>> x = Matrix([a, b])\n\n        >>> A = Matrix([[1, 0], [0, 1]])\n        >>> A.is_positive_definite\n        True\n        >>> A.is_positive_semidefinite\n        True\n\n        >>> p = plot3d((x.T*A*x)[0, 0], (a, -1, 1), (b, -1, 1))\n\n    An example of symmetric positive semidefinite matrix:\n\n    .. plot::\n        :context: close-figs\n        :format: doctest\n        :include-source: True\n\n        >>> A = Matrix([[1, -1], [-1, 1]])\n        >>> A.is_positive_definite\n        False\n        >>> A.is_positive_semidefinite\n        True\n\n        >>> p = plot3d((x.T*A*x)[0, 0], (a, -1, 1), (b, -1, 1))\n\n    An example of symmetric negative definite matrix:\n\n    .. plot::\n        :context: close-figs\n        :format: doctest\n        :include-source: True\n\n        >>> A = Matrix([[-1, 0], [0, -1]])\n        >>> A.is_negative_definite\n        True\n        >>> A.is_negative_semidefinite\n        True\n        >>> A.is_indefinite\n        False\n\n        >>> p = plot3d((x.T*A*x)[0, 0], (a, -1, 1), (b, -1, 1))\n\n    An example of symmetric indefinite matrix:\n\n    .. plot::\n        :context: close-figs\n        :format: doctest\n        :include-source: True\n\n        >>> A = Matrix([[1, 2], [2, -1]])\n        >>> A.is_indefinite\n        True\n\n        >>> p = plot3d((x.T*A*x)[0, 0], (a, -1, 1), (b, -1, 1))\n\n    An example of non-symmetric positive definite matrix.\n\n    .. plot::\n        :context: close-figs\n        :format: doctest\n        :include-source: True\n\n        >>> A = Matrix([[1, 2], [-2, 1]])\n        >>> A.is_positive_definite\n        True\n        >>> A.is_positive_semidefinite\n        True\n\n        >>> p = plot3d((x.T*A*x)[0, 0], (a, -1, 1), (b, -1, 1))\n\n    Notes\n    =====\n\n    Although some people trivialize the definition of positive definite\n    matrices only for symmetric or hermitian matrices, this restriction\n    is not correct because it does not classify all instances of\n    positive definite matrices from the definition $x^T A x > 0$ or\n    $\\text{re}(x^H A x) > 0$.\n\n    For instance, ``Matrix([[1, 2], [-2, 1]])`` presented in\n    the example above is an example of real positive definite matrix\n    that is not symmetric.\n\n    However, since the following formula holds true;\n\n    .. math::\n        \\text{re}(x^H A x) > 0 \\iff\n        \\text{re}(x^H \\frac{A + A^H}{2} x) > 0\n\n    We can classify all positive definite matrices that may or may not\n    be symmetric or hermitian by transforming the matrix to\n    $\\frac{A + A^T}{2}$ or $\\frac{A + A^H}{2}$\n    (which is guaranteed to be always real symmetric or complex\n    hermitian) and we can defer most of the studies to symmetric or\n    hermitian positive definite matrices.\n\n    But it is a different problem for the existance of Cholesky\n    decomposition. Because even though a non symmetric or a non\n    hermitian matrix can be positive definite, Cholesky or LDL\n    decomposition does not exist because the decompositions require the\n    matrix to be symmetric or hermitian.\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Definiteness_of_a_matrix#Eigenvalues\n\n    .. [2] https://mathworld.wolfram.com/PositiveDefiniteMatrix.html\n\n    .. [3] Johnson, C. R. "Positive Definite Matrices." Amer.\n        Math. Monthly 77, 259-264 1970.\n    '
_is_positive_definite.__doc__ = _doc_positive_definite
_is_positive_semidefinite.__doc__ = _doc_positive_definite
_is_negative_definite.__doc__ = _doc_positive_definite
_is_negative_semidefinite.__doc__ = _doc_positive_definite
_is_indefinite.__doc__ = _doc_positive_definite

def _jordan_form(M, calc_transform=True, *, chop=False):
    if False:
        return 10
    'Return $(P, J)$ where $J$ is a Jordan block\n    matrix and $P$ is a matrix such that $M = P J P^{-1}$\n\n    Parameters\n    ==========\n\n    calc_transform : bool\n        If ``False``, then only $J$ is returned.\n\n    chop : bool\n        All matrices are converted to exact types when computing\n        eigenvalues and eigenvectors.  As a result, there may be\n        approximation errors.  If ``chop==True``, these errors\n        will be truncated.\n\n    Examples\n    ========\n\n    >>> from sympy import Matrix\n    >>> M = Matrix([[ 6,  5, -2, -3], [-3, -1,  3,  3], [ 2,  1, -2, -3], [-1,  1,  5,  5]])\n    >>> P, J = M.jordan_form()\n    >>> J\n    Matrix([\n    [2, 1, 0, 0],\n    [0, 2, 0, 0],\n    [0, 0, 2, 1],\n    [0, 0, 0, 2]])\n\n    See Also\n    ========\n\n    jordan_block\n    '
    if not M.is_square:
        raise NonSquareMatrixError('Only square matrices have Jordan forms')
    mat = M
    has_floats = M.has(Float)
    if has_floats:
        try:
            max_prec = max((term._prec for term in M.values() if isinstance(term, Float)))
        except ValueError:
            max_prec = 53
        max_dps = max(prec_to_dps(max_prec), 15)

    def restore_floats(*args):
        if False:
            for i in range(10):
                print('nop')
        'If ``has_floats`` is `True`, cast all ``args`` as\n        matrices of floats.'
        if has_floats:
            args = [m.evalf(n=max_dps, chop=chop) for m in args]
        if len(args) == 1:
            return args[0]
        return args
    mat_cache = {}

    def eig_mat(val, pow):
        if False:
            print('Hello World!')
        'Cache computations of ``(M - val*I)**pow`` for quick\n        retrieval'
        if (val, pow) in mat_cache:
            return mat_cache[val, pow]
        if (val, pow - 1) in mat_cache:
            mat_cache[val, pow] = mat_cache[val, pow - 1].multiply(mat_cache[val, 1], dotprodsimp=None)
        else:
            mat_cache[val, pow] = (mat - val * M.eye(M.rows)).pow(pow)
        return mat_cache[val, pow]

    def nullity_chain(val, algebraic_multiplicity):
        if False:
            for i in range(10):
                print('nop')
        'Calculate the sequence  [0, nullity(E), nullity(E**2), ...]\n        until it is constant where ``E = M - val*I``'
        cols = M.cols
        ret = [0]
        nullity = cols - eig_mat(val, 1).rank()
        i = 2
        while nullity != ret[-1]:
            ret.append(nullity)
            if nullity == algebraic_multiplicity:
                break
            nullity = cols - eig_mat(val, i).rank()
            i += 1
            if nullity < ret[-1] or nullity > algebraic_multiplicity:
                raise MatrixError('SymPy had encountered an inconsistent result while computing Jordan block: {}'.format(M))
        return ret

    def blocks_from_nullity_chain(d):
        if False:
            for i in range(10):
                print('nop')
        'Return a list of the size of each Jordan block.\n        If d_n is the nullity of E**n, then the number\n        of Jordan blocks of size n is\n\n            2*d_n - d_(n-1) - d_(n+1)'
        mid = [2 * d[n] - d[n - 1] - d[n + 1] for n in range(1, len(d) - 1)]
        end = [d[-1] - d[-2]] if len(d) > 1 else [d[0]]
        return mid + end

    def pick_vec(small_basis, big_basis):
        if False:
            while True:
                i = 10
        "Picks a vector from big_basis that isn't in\n        the subspace spanned by small_basis"
        if len(small_basis) == 0:
            return big_basis[0]
        for v in big_basis:
            (_, pivots) = M.hstack(*small_basis + [v]).echelon_form(with_pivots=True)
            if pivots[-1] == len(small_basis):
                return v
    if has_floats:
        from sympy.simplify import nsimplify
        mat = mat.applyfunc(lambda x: nsimplify(x, rational=True))
    eigs = mat.eigenvals()
    for x in eigs:
        if x.has(CRootOf):
            raise MatrixError('Jordan normal form is not implemented if the matrix have eigenvalues in CRootOf form')
    if len(eigs.keys()) == mat.cols:
        blocks = sorted(eigs.keys(), key=default_sort_key)
        jordan_mat = mat.diag(*blocks)
        if not calc_transform:
            return restore_floats(jordan_mat)
        jordan_basis = [eig_mat(eig, 1).nullspace()[0] for eig in blocks]
        basis_mat = mat.hstack(*jordan_basis)
        return restore_floats(basis_mat, jordan_mat)
    block_structure = []
    for eig in sorted(eigs.keys(), key=default_sort_key):
        algebraic_multiplicity = eigs[eig]
        chain = nullity_chain(eig, algebraic_multiplicity)
        block_sizes = blocks_from_nullity_chain(chain)
        size_nums = [(i + 1, num) for (i, num) in enumerate(block_sizes)]
        size_nums.reverse()
        block_structure.extend([(eig, size) for (size, num) in size_nums for _ in range(num)])
    jordan_form_size = sum((size for (eig, size) in block_structure))
    if jordan_form_size != M.rows:
        raise MatrixError('SymPy had encountered an inconsistent result while computing Jordan block. : {}'.format(M))
    blocks = (mat.jordan_block(size=size, eigenvalue=eig) for (eig, size) in block_structure)
    jordan_mat = mat.diag(*blocks)
    if not calc_transform:
        return restore_floats(jordan_mat)
    jordan_basis = []
    for eig in sorted(eigs.keys(), key=default_sort_key):
        eig_basis = []
        for (block_eig, size) in block_structure:
            if block_eig != eig:
                continue
            null_big = eig_mat(eig, size).nullspace()
            null_small = eig_mat(eig, size - 1).nullspace()
            vec = pick_vec(null_small + eig_basis, null_big)
            new_vecs = [eig_mat(eig, i).multiply(vec, dotprodsimp=None) for i in range(size)]
            eig_basis.extend(new_vecs)
            jordan_basis.extend(reversed(new_vecs))
    basis_mat = mat.hstack(*jordan_basis)
    return restore_floats(basis_mat, jordan_mat)

def _left_eigenvects(M, **flags):
    if False:
        print('Hello World!')
    'Returns left eigenvectors and eigenvalues.\n\n    This function returns the list of triples (eigenval, multiplicity,\n    basis) for the left eigenvectors. Options are the same as for\n    eigenvects(), i.e. the ``**flags`` arguments gets passed directly to\n    eigenvects().\n\n    Examples\n    ========\n\n    >>> from sympy import Matrix\n    >>> M = Matrix([[0, 1, 1], [1, 0, 0], [1, 1, 1]])\n    >>> M.eigenvects()\n    [(-1, 1, [Matrix([\n    [-1],\n    [ 1],\n    [ 0]])]), (0, 1, [Matrix([\n    [ 0],\n    [-1],\n    [ 1]])]), (2, 1, [Matrix([\n    [2/3],\n    [1/3],\n    [  1]])])]\n    >>> M.left_eigenvects()\n    [(-1, 1, [Matrix([[-2, 1, 1]])]), (0, 1, [Matrix([[-1, -1, 1]])]), (2,\n    1, [Matrix([[1, 1, 1]])])]\n\n    '
    eigs = M.transpose().eigenvects(**flags)
    return [(val, mult, [l.transpose() for l in basis]) for (val, mult, basis) in eigs]

def _singular_values(M):
    if False:
        for i in range(10):
            print('nop')
    "Compute the singular values of a Matrix\n\n    Examples\n    ========\n\n    >>> from sympy import Matrix, Symbol\n    >>> x = Symbol('x', real=True)\n    >>> M = Matrix([[0, 1, 0], [0, x, 0], [-1, 0, 0]])\n    >>> M.singular_values()\n    [sqrt(x**2 + 1), 1, 0]\n\n    See Also\n    ========\n\n    condition_number\n    "
    if M.rows >= M.cols:
        valmultpairs = M.H.multiply(M).eigenvals()
    else:
        valmultpairs = M.multiply(M.H).eigenvals()
    vals = []
    for (k, v) in valmultpairs.items():
        vals += [sqrt(k)] * v
    if len(vals) < M.cols:
        vals += [M.zero] * (M.cols - len(vals))
    vals.sort(reverse=True, key=default_sort_key)
    return vals