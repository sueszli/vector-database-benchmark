from types import FunctionType
from sympy.core.cache import cacheit
from sympy.core.numbers import Float, Integer
from sympy.core.singleton import S
from sympy.core.symbol import uniquely_named_symbol
from sympy.core.mul import Mul
from sympy.polys import PurePoly, cancel
from sympy.functions.combinatorial.numbers import nC
from sympy.polys.matrices.domainmatrix import DomainMatrix
from sympy.polys.matrices.ddm import DDM
from .common import NonSquareMatrixError
from .utilities import _get_intermediate_simp, _get_intermediate_simp_bool, _iszero, _is_zero_after_expand_mul, _dotprodsimp, _simplify

def _find_reasonable_pivot(col, iszerofunc=_iszero, simpfunc=_simplify):
    if False:
        return 10
    ' Find the lowest index of an item in ``col`` that is\n    suitable for a pivot.  If ``col`` consists only of\n    Floats, the pivot with the largest norm is returned.\n    Otherwise, the first element where ``iszerofunc`` returns\n    False is used.  If ``iszerofunc`` does not return false,\n    items are simplified and retested until a suitable\n    pivot is found.\n\n    Returns a 4-tuple\n        (pivot_offset, pivot_val, assumed_nonzero, newly_determined)\n    where pivot_offset is the index of the pivot, pivot_val is\n    the (possibly simplified) value of the pivot, assumed_nonzero\n    is True if an assumption that the pivot was non-zero\n    was made without being proved, and newly_determined are\n    elements that were simplified during the process of pivot\n    finding.'
    newly_determined = []
    col = list(col)
    if all((isinstance(x, (Float, Integer)) for x in col)) and any((isinstance(x, Float) for x in col)):
        col_abs = [abs(x) for x in col]
        max_value = max(col_abs)
        if iszerofunc(max_value):
            if max_value != 0:
                newly_determined = [(i, 0) for (i, x) in enumerate(col) if x != 0]
            return (None, None, False, newly_determined)
        index = col_abs.index(max_value)
        return (index, col[index], False, newly_determined)
    possible_zeros = []
    for (i, x) in enumerate(col):
        is_zero = iszerofunc(x)
        if is_zero == False:
            return (i, x, False, newly_determined)
        possible_zeros.append(is_zero)
    if all(possible_zeros):
        return (None, None, False, newly_determined)
    for (i, x) in enumerate(col):
        if possible_zeros[i] is not None:
            continue
        simped = simpfunc(x)
        is_zero = iszerofunc(simped)
        if is_zero in (True, False):
            newly_determined.append((i, simped))
        if is_zero == False:
            return (i, simped, False, newly_determined)
        possible_zeros[i] = is_zero
    if all(possible_zeros):
        return (None, None, False, newly_determined)
    for (i, x) in enumerate(col):
        if possible_zeros[i] is not None:
            continue
        if x.equals(S.Zero):
            possible_zeros[i] = True
            newly_determined.append((i, S.Zero))
    if all(possible_zeros):
        return (None, None, False, newly_determined)
    i = possible_zeros.index(None)
    return (i, col[i], True, newly_determined)

def _find_reasonable_pivot_naive(col, iszerofunc=_iszero, simpfunc=None):
    if False:
        while True:
            i = 10
    '\n    Helper that computes the pivot value and location from a\n    sequence of contiguous matrix column elements. As a side effect\n    of the pivot search, this function may simplify some of the elements\n    of the input column. A list of these simplified entries and their\n    indices are also returned.\n    This function mimics the behavior of _find_reasonable_pivot(),\n    but does less work trying to determine if an indeterminate candidate\n    pivot simplifies to zero. This more naive approach can be much faster,\n    with the trade-off that it may erroneously return a pivot that is zero.\n\n    ``col`` is a sequence of contiguous column entries to be searched for\n    a suitable pivot.\n    ``iszerofunc`` is a callable that returns a Boolean that indicates\n    if its input is zero, or None if no such determination can be made.\n    ``simpfunc`` is a callable that simplifies its input. It must return\n    its input if it does not simplify its input. Passing in\n    ``simpfunc=None`` indicates that the pivot search should not attempt\n    to simplify any candidate pivots.\n\n    Returns a 4-tuple:\n    (pivot_offset, pivot_val, assumed_nonzero, newly_determined)\n    ``pivot_offset`` is the sequence index of the pivot.\n    ``pivot_val`` is the value of the pivot.\n    pivot_val and col[pivot_index] are equivalent, but will be different\n    when col[pivot_index] was simplified during the pivot search.\n    ``assumed_nonzero`` is a boolean indicating if the pivot cannot be\n    guaranteed to be zero. If assumed_nonzero is true, then the pivot\n    may or may not be non-zero. If assumed_nonzero is false, then\n    the pivot is non-zero.\n    ``newly_determined`` is a list of index-value pairs of pivot candidates\n    that were simplified during the pivot search.\n    '
    indeterminates = []
    for (i, col_val) in enumerate(col):
        col_val_is_zero = iszerofunc(col_val)
        if col_val_is_zero == False:
            return (i, col_val, False, [])
        elif col_val_is_zero is None:
            indeterminates.append((i, col_val))
    if len(indeterminates) == 0:
        return (None, None, False, [])
    if simpfunc is None:
        return (indeterminates[0][0], indeterminates[0][1], True, [])
    newly_determined = []
    for (i, col_val) in indeterminates:
        tmp_col_val = simpfunc(col_val)
        if id(col_val) != id(tmp_col_val):
            newly_determined.append((i, tmp_col_val))
            if iszerofunc(tmp_col_val) == False:
                return (i, tmp_col_val, False, newly_determined)
    return (indeterminates[0][0], indeterminates[0][1], True, newly_determined)

def _berkowitz_toeplitz_matrix(M):
    if False:
        while True:
            i = 10
    'Return (A,T) where T the Toeplitz matrix used in the Berkowitz algorithm\n    corresponding to ``M`` and A is the first principal submatrix.\n    '
    if M.rows == 0 and M.cols == 0:
        return M._new(1, 1, [M.one])
    (a, R) = (M[0, 0], M[0, 1:])
    (C, A) = (M[1:, 0], M[1:, 1:])
    diags = [C]
    for i in range(M.rows - 2):
        diags.append(A.multiply(diags[i], dotprodsimp=None))
    diags = [(-R).multiply(d, dotprodsimp=None)[0, 0] for d in diags]
    diags = [M.one, -a] + diags

    def entry(i, j):
        if False:
            print('Hello World!')
        if j > i:
            return M.zero
        return diags[i - j]
    toeplitz = M._new(M.cols + 1, M.rows, entry)
    return (A, toeplitz)

def _berkowitz_vector(M):
    if False:
        for i in range(10):
            print('nop')
    ' Run the Berkowitz algorithm and return a vector whose entries\n        are the coefficients of the characteristic polynomial of ``M``.\n\n        Given N x N matrix, efficiently compute\n        coefficients of characteristic polynomials of ``M``\n        without division in the ground domain.\n\n        This method is particularly useful for computing determinant,\n        principal minors and characteristic polynomial when ``M``\n        has complicated coefficients e.g. polynomials. Semi-direct\n        usage of this algorithm is also important in computing\n        efficiently sub-resultant PRS.\n\n        Assuming that M is a square matrix of dimension N x N and\n        I is N x N identity matrix, then the Berkowitz vector is\n        an N x 1 vector whose entries are coefficients of the\n        polynomial\n\n                        charpoly(M) = det(t*I - M)\n\n        As a consequence, all polynomials generated by Berkowitz\n        algorithm are monic.\n\n        For more information on the implemented algorithm refer to:\n\n        [1] S.J. Berkowitz, On computing the determinant in small\n            parallel time using a small number of processors, ACM,\n            Information Processing Letters 18, 1984, pp. 147-150\n\n        [2] M. Keber, Division-Free computation of sub-resultants\n            using Bezout matrices, Tech. Report MPI-I-2006-1-006,\n            Saarbrucken, 2006\n    '
    if M.rows == 0 and M.cols == 0:
        return M._new(1, 1, [M.one])
    elif M.rows == 1 and M.cols == 1:
        return M._new(2, 1, [M.one, -M[0, 0]])
    (submat, toeplitz) = _berkowitz_toeplitz_matrix(M)
    return toeplitz.multiply(_berkowitz_vector(submat), dotprodsimp=None)

def _adjugate(M, method='berkowitz'):
    if False:
        while True:
            i = 10
    'Returns the adjugate, or classical adjoint, of\n    a matrix.  That is, the transpose of the matrix of cofactors.\n\n    https://en.wikipedia.org/wiki/Adjugate\n\n    Parameters\n    ==========\n\n    method : string, optional\n        Method to use to find the cofactors, can be "bareiss", "berkowitz",\n        "bird", "laplace" or "lu".\n\n    Examples\n    ========\n\n    >>> from sympy import Matrix\n    >>> M = Matrix([[1, 2], [3, 4]])\n    >>> M.adjugate()\n    Matrix([\n    [ 4, -2],\n    [-3,  1]])\n\n    See Also\n    ========\n\n    cofactor_matrix\n    sympy.matrices.common.MatrixCommon.transpose\n    '
    return M.cofactor_matrix(method=method).transpose()

def _charpoly(M, x='lambda', simplify=_simplify):
    if False:
        print('Hello World!')
    'Computes characteristic polynomial det(x*I - M) where I is\n    the identity matrix.\n\n    A PurePoly is returned, so using different variables for ``x`` does\n    not affect the comparison or the polynomials:\n\n    Parameters\n    ==========\n\n    x : string, optional\n        Name for the "lambda" variable, defaults to "lambda".\n\n    simplify : function, optional\n        Simplification function to use on the characteristic polynomial\n        calculated. Defaults to ``simplify``.\n\n    Examples\n    ========\n\n    >>> from sympy import Matrix\n    >>> from sympy.abc import x, y\n    >>> M = Matrix([[1, 3], [2, 0]])\n    >>> M.charpoly()\n    PurePoly(lambda**2 - lambda - 6, lambda, domain=\'ZZ\')\n    >>> M.charpoly(x) == M.charpoly(y)\n    True\n    >>> M.charpoly(x) == M.charpoly(y)\n    True\n\n    Specifying ``x`` is optional; a symbol named ``lambda`` is used by\n    default (which looks good when pretty-printed in unicode):\n\n    >>> M.charpoly().as_expr()\n    lambda**2 - lambda - 6\n\n    And if ``x`` clashes with an existing symbol, underscores will\n    be prepended to the name to make it unique:\n\n    >>> M = Matrix([[1, 2], [x, 0]])\n    >>> M.charpoly(x).as_expr()\n    _x**2 - _x - 2*x\n\n    Whether you pass a symbol or not, the generator can be obtained\n    with the gen attribute since it may not be the same as the symbol\n    that was passed:\n\n    >>> M.charpoly(x).gen\n    _x\n    >>> M.charpoly(x).gen == x\n    False\n\n    Notes\n    =====\n\n    The Samuelson-Berkowitz algorithm is used to compute\n    the characteristic polynomial efficiently and without any\n    division operations.  Thus the characteristic polynomial over any\n    commutative ring without zero divisors can be computed.\n\n    If the determinant det(x*I - M) can be found out easily as\n    in the case of an upper or a lower triangular matrix, then\n    instead of Samuelson-Berkowitz algorithm, eigenvalues are computed\n    and the characteristic polynomial with their help.\n\n    See Also\n    ========\n\n    det\n    '
    if not M.is_square:
        raise NonSquareMatrixError()
    dM = M.to_DM()
    K = dM.domain
    cp = dM.charpoly()
    x = uniquely_named_symbol(x, M, modify=lambda s: '_' + s)
    if K.is_EXRAW or simplify is not _simplify:
        berk_vector = [K.to_sympy(c) for c in cp]
        berk_vector = [simplify(a) for a in berk_vector]
        p = PurePoly(berk_vector, x)
    else:
        p = PurePoly(cp, x, domain=K)
    return p

def _cofactor(M, i, j, method='berkowitz'):
    if False:
        while True:
            i = 10
    'Calculate the cofactor of an element.\n\n    Parameters\n    ==========\n\n    method : string, optional\n        Method to use to find the cofactors, can be "bareiss", "berkowitz",\n        "bird", "laplace" or "lu".\n\n    Examples\n    ========\n\n    >>> from sympy import Matrix\n    >>> M = Matrix([[1, 2], [3, 4]])\n    >>> M.cofactor(0, 1)\n    -3\n\n    See Also\n    ========\n\n    cofactor_matrix\n    minor\n    minor_submatrix\n    '
    if not M.is_square or M.rows < 1:
        raise NonSquareMatrixError()
    return S.NegativeOne ** ((i + j) % 2) * M.minor(i, j, method)

def _cofactor_matrix(M, method='berkowitz'):
    if False:
        print('Hello World!')
    'Return a matrix containing the cofactor of each element.\n\n    Parameters\n    ==========\n\n    method : string, optional\n        Method to use to find the cofactors, can be "bareiss", "berkowitz",\n        "bird", "laplace" or "lu".\n\n    Examples\n    ========\n\n    >>> from sympy import Matrix\n    >>> M = Matrix([[1, 2], [3, 4]])\n    >>> M.cofactor_matrix()\n    Matrix([\n    [ 4, -3],\n    [-2,  1]])\n\n    See Also\n    ========\n\n    cofactor\n    minor\n    minor_submatrix\n    '
    if not M.is_square:
        raise NonSquareMatrixError()
    return M._new(M.rows, M.cols, lambda i, j: M.cofactor(i, j, method))

def _per(M):
    if False:
        i = 10
        return i + 15
    "Returns the permanent of a matrix. Unlike determinant,\n    permanent is defined for both square and non-square matrices.\n\n    For an m x n matrix, with m less than or equal to n,\n    it is given as the sum over the permutations s of size\n    less than or equal to m on [1, 2, . . . n] of the product\n    from i = 1 to m of M[i, s[i]]. Taking the transpose will\n    not affect the value of the permanent.\n\n    In the case of a square matrix, this is the same as the permutation\n    definition of the determinant, but it does not take the sign of the\n    permutation into account. Computing the permanent with this definition\n    is quite inefficient, so here the Ryser formula is used.\n\n    Examples\n    ========\n\n    >>> from sympy import Matrix\n    >>> M = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])\n    >>> M.per()\n    450\n    >>> M = Matrix([1, 5, 7])\n    >>> M.per()\n    13\n\n    References\n    ==========\n\n    .. [1] Prof. Frank Ben's notes: https://math.berkeley.edu/~bernd/ban275.pdf\n    .. [2] Wikipedia article on Permanent: https://en.wikipedia.org/wiki/Permanent_%28mathematics%29\n    .. [3] https://reference.wolfram.com/language/ref/Permanent.html\n    .. [4] Permanent of a rectangular matrix : https://arxiv.org/pdf/0904.3251.pdf\n    "
    import itertools
    (m, n) = M.shape
    if m > n:
        M = M.T
        (m, n) = (n, m)
    s = list(range(n))
    subsets = []
    for i in range(1, m + 1):
        subsets += list(map(list, itertools.combinations(s, i)))
    perm = 0
    for subset in subsets:
        prod = 1
        sub_len = len(subset)
        for i in range(m):
            prod *= sum([M[i, j] for j in subset])
        perm += prod * S.NegativeOne ** sub_len * nC(n - sub_len, m - sub_len)
    perm *= S.NegativeOne ** m
    return perm.simplify()

def _det_DOM(M):
    if False:
        for i in range(10):
            print('nop')
    DOM = DomainMatrix.from_Matrix(M, field=True, extension=True)
    K = DOM.domain
    return K.to_sympy(DOM.det())

def _det(M, method='bareiss', iszerofunc=None):
    if False:
        return 10
    'Computes the determinant of a matrix if ``M`` is a concrete matrix object\n    otherwise return an expressions ``Determinant(M)`` if ``M`` is a\n    ``MatrixSymbol`` or other expression.\n\n    Parameters\n    ==========\n\n    method : string, optional\n        Specifies the algorithm used for computing the matrix determinant.\n\n        If the matrix is at most 3x3, a hard-coded formula is used and the\n        specified method is ignored. Otherwise, it defaults to\n        ``\'bareiss\'``.\n\n        Also, if the matrix is an upper or a lower triangular matrix, determinant\n        is computed by simple multiplication of diagonal elements, and the\n        specified method is ignored.\n\n        If it is set to ``\'domain-ge\'``, then Gaussian elimination method will\n        be used via using DomainMatrix.\n\n        If it is set to ``\'bareiss\'``, Bareiss\' fraction-free algorithm will\n        be used.\n\n        If it is set to ``\'berkowitz\'``, Berkowitz\' algorithm will be used.\n\n        If it is set to ``\'bird\'``, Bird\'s algorithm will be used [1]_.\n\n        If it is set to ``\'laplace\'``, Laplace\'s algorithm will be used.\n\n        Otherwise, if it is set to ``\'lu\'``, LU decomposition will be used.\n\n        .. note::\n            For backward compatibility, legacy keys like "bareis" and\n            "det_lu" can still be used to indicate the corresponding\n            methods.\n            And the keys are also case-insensitive for now. However, it is\n            suggested to use the precise keys for specifying the method.\n\n    iszerofunc : FunctionType or None, optional\n        If it is set to ``None``, it will be defaulted to ``_iszero`` if the\n        method is set to ``\'bareiss\'``, and ``_is_zero_after_expand_mul`` if\n        the method is set to ``\'lu\'``.\n\n        It can also accept any user-specified zero testing function, if it\n        is formatted as a function which accepts a single symbolic argument\n        and returns ``True`` if it is tested as zero and ``False`` if it\n        tested as non-zero, and also ``None`` if it is undecidable.\n\n    Returns\n    =======\n\n    det : Basic\n        Result of determinant.\n\n    Raises\n    ======\n\n    ValueError\n        If unrecognized keys are given for ``method`` or ``iszerofunc``.\n\n    NonSquareMatrixError\n        If attempted to calculate determinant from a non-square matrix.\n\n    Examples\n    ========\n\n    >>> from sympy import Matrix, eye, det\n    >>> I3 = eye(3)\n    >>> det(I3)\n    1\n    >>> M = Matrix([[1, 2], [3, 4]])\n    >>> det(M)\n    -2\n    >>> det(M) == M.det()\n    True\n    >>> M.det(method="domain-ge")\n    -2\n\n    References\n    ==========\n\n    .. [1] Bird, R. S. (2011). A simple division-free algorithm for computing\n           determinants. Inf. Process. Lett., 111(21), 1072-1074. doi:\n           10.1016/j.ipl.2011.08.006\n    '
    method = method.lower()
    if method == 'bareis':
        method = 'bareiss'
    elif method == 'det_lu':
        method = 'lu'
    if method not in ('bareiss', 'berkowitz', 'lu', 'domain-ge', 'bird', 'laplace'):
        raise ValueError("Determinant method '%s' unrecognized" % method)
    if iszerofunc is None:
        if method == 'bareiss':
            iszerofunc = _is_zero_after_expand_mul
        elif method == 'lu':
            iszerofunc = _iszero
    elif not isinstance(iszerofunc, FunctionType):
        raise ValueError("Zero testing method '%s' unrecognized" % iszerofunc)
    n = M.rows
    if n == M.cols:
        if n == 0:
            return M.one
        elif n == 1:
            return M[0, 0]
        elif n == 2:
            m = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
            return _get_intermediate_simp(_dotprodsimp)(m)
        elif n == 3:
            m = M[0, 0] * M[1, 1] * M[2, 2] + M[0, 1] * M[1, 2] * M[2, 0] + M[0, 2] * M[1, 0] * M[2, 1] - M[0, 2] * M[1, 1] * M[2, 0] - M[0, 0] * M[1, 2] * M[2, 1] - M[0, 1] * M[1, 0] * M[2, 2]
            return _get_intermediate_simp(_dotprodsimp)(m)
    dets = []
    for b in M.strongly_connected_components():
        if method == 'domain-ge':
            det = _det_DOM(M[b, b])
        elif method == 'bareiss':
            det = M[b, b]._eval_det_bareiss(iszerofunc=iszerofunc)
        elif method == 'berkowitz':
            det = M[b, b]._eval_det_berkowitz()
        elif method == 'lu':
            det = M[b, b]._eval_det_lu(iszerofunc=iszerofunc)
        elif method == 'bird':
            det = M[b, b]._eval_det_bird()
        elif method == 'laplace':
            det = M[b, b]._eval_det_laplace()
        dets.append(det)
    return Mul(*dets)

def _det_bareiss(M, iszerofunc=_is_zero_after_expand_mul):
    if False:
        for i in range(10):
            print('nop')
    "Compute matrix determinant using Bareiss' fraction-free\n    algorithm which is an extension of the well known Gaussian\n    elimination method. This approach is best suited for dense\n    symbolic matrices and will result in a determinant with\n    minimal number of fractions. It means that less term\n    rewriting is needed on resulting formulae.\n\n    Parameters\n    ==========\n\n    iszerofunc : function, optional\n        The function to use to determine zeros when doing an LU decomposition.\n        Defaults to ``lambda x: x.is_zero``.\n\n    TODO: Implement algorithm for sparse matrices (SFF),\n    http://www.eecis.udel.edu/~saunders/papers/sffge/it5.ps.\n    "

    def bareiss(mat, cumm=1):
        if False:
            for i in range(10):
                print('nop')
        if mat.rows == 0:
            return mat.one
        elif mat.rows == 1:
            return mat[0, 0]
        (pivot_pos, pivot_val, _, _) = _find_reasonable_pivot(mat[:, 0], iszerofunc=iszerofunc)
        if pivot_pos is None:
            return mat.zero
        sign = (-1) ** (pivot_pos % 2)
        rows = [i for i in range(mat.rows) if i != pivot_pos]
        cols = list(range(mat.cols))
        tmp_mat = mat.extract(rows, cols)

        def entry(i, j):
            if False:
                print('Hello World!')
            ret = (pivot_val * tmp_mat[i, j + 1] - mat[pivot_pos, j + 1] * tmp_mat[i, 0]) / cumm
            if _get_intermediate_simp_bool(True):
                return _dotprodsimp(ret)
            elif not ret.is_Atom:
                return cancel(ret)
            return ret
        return sign * bareiss(M._new(mat.rows - 1, mat.cols - 1, entry), pivot_val)
    if not M.is_square:
        raise NonSquareMatrixError()
    if M.rows == 0:
        return M.one
    return bareiss(M)

def _det_berkowitz(M):
    if False:
        for i in range(10):
            print('nop')
    ' Use the Berkowitz algorithm to compute the determinant.'
    if not M.is_square:
        raise NonSquareMatrixError()
    if M.rows == 0:
        return M.one
    berk_vector = _berkowitz_vector(M)
    return (-1) ** (len(berk_vector) - 1) * berk_vector[-1]

def _det_LU(M, iszerofunc=_iszero, simpfunc=None):
    if False:
        i = 10
        return i + 15
    ' Computes the determinant of a matrix from its LU decomposition.\n    This function uses the LU decomposition computed by\n    LUDecomposition_Simple().\n\n    The keyword arguments iszerofunc and simpfunc are passed to\n    LUDecomposition_Simple().\n    iszerofunc is a callable that returns a boolean indicating if its\n    input is zero, or None if it cannot make the determination.\n    simpfunc is a callable that simplifies its input.\n    The default is simpfunc=None, which indicate that the pivot search\n    algorithm should not attempt to simplify any candidate pivots.\n    If simpfunc fails to simplify its input, then it must return its input\n    instead of a copy.\n\n    Parameters\n    ==========\n\n    iszerofunc : function, optional\n        The function to use to determine zeros when doing an LU decomposition.\n        Defaults to ``lambda x: x.is_zero``.\n\n    simpfunc : function, optional\n        The simplification function to use when looking for zeros for pivots.\n    '
    if not M.is_square:
        raise NonSquareMatrixError()
    if M.rows == 0:
        return M.one
    (lu, row_swaps) = M.LUdecomposition_Simple(iszerofunc=iszerofunc, simpfunc=simpfunc)
    if iszerofunc(lu[lu.rows - 1, lu.rows - 1]):
        return M.zero
    det = -M.one if len(row_swaps) % 2 else M.one
    for k in range(lu.rows):
        det *= lu[k, k]
    return det

@cacheit
def __det_laplace(M):
    if False:
        while True:
            i = 10
    'Compute the determinant of a matrix using Laplace expansion.\n\n    This is a recursive function, and it should not be called directly.\n    Use _det_laplace() instead. The reason for splitting this function\n    into two is to allow caching of determinants of submatrices. While\n    one could also define this function inside _det_laplace(), that\n    would remove the advantage of using caching in Cramer Solve.\n    '
    n = M.shape[0]
    if n == 1:
        return M[0]
    elif n == 2:
        return M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    else:
        return sum(((-1) ** i * M[0, i] * __det_laplace(M.minor_submatrix(0, i)) for i in range(n)))

def _det_laplace(M):
    if False:
        for i in range(10):
            print('nop')
    'Compute the determinant of a matrix using Laplace expansion.\n\n    While Laplace expansion is not the most efficient method of computing\n    a determinant, it is a simple one, and it has the advantage of\n    being division free. To improve efficiency, this function uses\n    caching to avoid recomputing determinants of submatrices.\n    '
    if not M.is_square:
        raise NonSquareMatrixError()
    if M.shape[0] == 0:
        return M.one
    return __det_laplace(M.as_immutable())

def _det_bird(M):
    if False:
        print('Hello World!')
    "Compute the determinant of a matrix using Bird's algorithm.\n\n    Bird's algorithm is a simple division-free algorithm for computing, which\n    is of lower order than the Laplace's algorithm. It is described in [1]_.\n\n    References\n    ==========\n\n    .. [1] Bird, R. S. (2011). A simple division-free algorithm for computing\n           determinants. Inf. Process. Lett., 111(21), 1072-1074. doi:\n           10.1016/j.ipl.2011.08.006\n    "

    def mu(X):
        if False:
            return 10
        n = X.shape[0]
        zero = X.domain.zero
        total = zero
        diag_sums = [zero]
        for i in reversed(range(1, n)):
            total -= X[i][i]
            diag_sums.append(total)
        diag_sums = diag_sums[::-1]
        elems = [[zero] * i + [diag_sums[i]] + X_i[i + 1:] for (i, X_i) in enumerate(X)]
        return DDM(elems, X.shape, X.domain)
    Mddm = M._rep.to_ddm()
    n = M.shape[0]
    if n == 0:
        return M.one
    Fn1 = Mddm
    for _ in range(n - 1):
        Fn1 = mu(Fn1).matmul(Mddm)
    detA = Fn1[0][0]
    if n % 2 == 0:
        detA = -detA
    return Mddm.domain.to_sympy(detA)

def _minor(M, i, j, method='berkowitz'):
    if False:
        while True:
            i = 10
    'Return the (i,j) minor of ``M``.  That is,\n    return the determinant of the matrix obtained by deleting\n    the `i`th row and `j`th column from ``M``.\n\n    Parameters\n    ==========\n\n    i, j : int\n        The row and column to exclude to obtain the submatrix.\n\n    method : string, optional\n        Method to use to find the determinant of the submatrix, can be\n        "bareiss", "berkowitz", "bird", "laplace" or "lu".\n\n    Examples\n    ========\n\n    >>> from sympy import Matrix\n    >>> M = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])\n    >>> M.minor(1, 1)\n    -12\n\n    See Also\n    ========\n\n    minor_submatrix\n    cofactor\n    det\n    '
    if not M.is_square:
        raise NonSquareMatrixError()
    return M.minor_submatrix(i, j).det(method=method)

def _minor_submatrix(M, i, j):
    if False:
        for i in range(10):
            print('nop')
    'Return the submatrix obtained by removing the `i`th row\n    and `j`th column from ``M`` (works with Pythonic negative indices).\n\n    Parameters\n    ==========\n\n    i, j : int\n        The row and column to exclude to obtain the submatrix.\n\n    Examples\n    ========\n\n    >>> from sympy import Matrix\n    >>> M = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])\n    >>> M.minor_submatrix(1, 1)\n    Matrix([\n    [1, 3],\n    [7, 9]])\n\n    See Also\n    ========\n\n    minor\n    cofactor\n    '
    if i < 0:
        i += M.rows
    if j < 0:
        j += M.cols
    if not 0 <= i < M.rows or not 0 <= j < M.cols:
        raise ValueError('`i` and `j` must satisfy 0 <= i < ``M.rows`` (%d)' % M.rows + 'and 0 <= j < ``M.cols`` (%d).' % M.cols)
    rows = [a for a in range(M.rows) if a != i]
    cols = [a for a in range(M.cols) if a != j]
    return M.extract(rows, cols)