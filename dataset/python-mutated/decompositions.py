import copy
from sympy.core import S
from sympy.core.function import expand_mul
from sympy.functions.elementary.miscellaneous import Min, sqrt
from sympy.functions.elementary.complexes import sign
from .common import NonSquareMatrixError, NonPositiveDefiniteMatrixError
from .utilities import _get_intermediate_simp, _iszero
from .determinant import _find_reasonable_pivot_naive

def _rank_decomposition(M, iszerofunc=_iszero, simplify=False):
    if False:
        i = 10
        return i + 15
    'Returns a pair of matrices (`C`, `F`) with matching rank\n    such that `A = C F`.\n\n    Parameters\n    ==========\n\n    iszerofunc : Function, optional\n        A function used for detecting whether an element can\n        act as a pivot.  ``lambda x: x.is_zero`` is used by default.\n\n    simplify : Bool or Function, optional\n        A function used to simplify elements when looking for a\n        pivot. By default SymPy\'s ``simplify`` is used.\n\n    Returns\n    =======\n\n    (C, F) : Matrices\n        `C` and `F` are full-rank matrices with rank as same as `A`,\n        whose product gives `A`.\n\n        See Notes for additional mathematical details.\n\n    Examples\n    ========\n\n    >>> from sympy import Matrix\n    >>> A = Matrix([\n    ...     [1, 3, 1, 4],\n    ...     [2, 7, 3, 9],\n    ...     [1, 5, 3, 1],\n    ...     [1, 2, 0, 8]\n    ... ])\n    >>> C, F = A.rank_decomposition()\n    >>> C\n    Matrix([\n    [1, 3, 4],\n    [2, 7, 9],\n    [1, 5, 1],\n    [1, 2, 8]])\n    >>> F\n    Matrix([\n    [1, 0, -2, 0],\n    [0, 1,  1, 0],\n    [0, 0,  0, 1]])\n    >>> C * F == A\n    True\n\n    Notes\n    =====\n\n    Obtaining `F`, an RREF of `A`, is equivalent to creating a\n    product\n\n    .. math::\n        E_n E_{n-1} ... E_1 A = F\n\n    where `E_n, E_{n-1}, \\dots, E_1` are the elimination matrices or\n    permutation matrices equivalent to each row-reduction step.\n\n    The inverse of the same product of elimination matrices gives\n    `C`:\n\n    .. math::\n        C = \\left(E_n E_{n-1} \\dots E_1\\right)^{-1}\n\n    It is not necessary, however, to actually compute the inverse:\n    the columns of `C` are those from the original matrix with the\n    same column indices as the indices of the pivot columns of `F`.\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Rank_factorization\n\n    .. [2] Piziak, R.; Odell, P. L. (1 June 1999).\n        "Full Rank Factorization of Matrices".\n        Mathematics Magazine. 72 (3): 193. doi:10.2307/2690882\n\n    See Also\n    ========\n\n    sympy.matrices.matrices.MatrixReductions.rref\n    '
    (F, pivot_cols) = M.rref(simplify=simplify, iszerofunc=iszerofunc, pivots=True)
    rank = len(pivot_cols)
    C = M.extract(range(M.rows), pivot_cols)
    F = F[:rank, :]
    return (C, F)

def _liupc(M):
    if False:
        while True:
            i = 10
    "Liu's algorithm, for pre-determination of the Elimination Tree of\n    the given matrix, used in row-based symbolic Cholesky factorization.\n\n    Examples\n    ========\n\n    >>> from sympy import SparseMatrix\n    >>> S = SparseMatrix([\n    ... [1, 0, 3, 2],\n    ... [0, 0, 1, 0],\n    ... [4, 0, 0, 5],\n    ... [0, 6, 7, 0]])\n    >>> S.liupc()\n    ([[0], [], [0], [1, 2]], [4, 3, 4, 4])\n\n    References\n    ==========\n\n    .. [1] Symbolic Sparse Cholesky Factorization using Elimination Trees,\n           Jeroen Van Grondelle (1999)\n           https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.39.7582\n    "
    R = [[] for r in range(M.rows)]
    for (r, c, _) in M.row_list():
        if c <= r:
            R[r].append(c)
    inf = len(R)
    parent = [inf] * M.rows
    virtual = [inf] * M.rows
    for r in range(M.rows):
        for c in R[r][:-1]:
            while virtual[c] < r:
                t = virtual[c]
                virtual[c] = r
                c = t
            if virtual[c] == inf:
                parent[c] = virtual[c] = r
    return (R, parent)

def _row_structure_symbolic_cholesky(M):
    if False:
        for i in range(10):
            print('nop')
    'Symbolic cholesky factorization, for pre-determination of the\n    non-zero structure of the Cholesky factororization.\n\n    Examples\n    ========\n\n    >>> from sympy import SparseMatrix\n    >>> S = SparseMatrix([\n    ... [1, 0, 3, 2],\n    ... [0, 0, 1, 0],\n    ... [4, 0, 0, 5],\n    ... [0, 6, 7, 0]])\n    >>> S.row_structure_symbolic_cholesky()\n    [[0], [], [0], [1, 2]]\n\n    References\n    ==========\n\n    .. [1] Symbolic Sparse Cholesky Factorization using Elimination Trees,\n           Jeroen Van Grondelle (1999)\n           https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.39.7582\n    '
    (R, parent) = M.liupc()
    inf = len(R)
    Lrow = copy.deepcopy(R)
    for k in range(M.rows):
        for j in R[k]:
            while j != inf and j != k:
                Lrow[k].append(j)
                j = parent[j]
        Lrow[k] = sorted(set(Lrow[k]))
    return Lrow

def _cholesky(M, hermitian=True):
    if False:
        for i in range(10):
            print('nop')
    'Returns the Cholesky-type decomposition L of a matrix A\n    such that L * L.H == A if hermitian flag is True,\n    or L * L.T == A if hermitian is False.\n\n    A must be a Hermitian positive-definite matrix if hermitian is True,\n    or a symmetric matrix if it is False.\n\n    Examples\n    ========\n\n    >>> from sympy import Matrix\n    >>> A = Matrix(((25, 15, -5), (15, 18, 0), (-5, 0, 11)))\n    >>> A.cholesky()\n    Matrix([\n    [ 5, 0, 0],\n    [ 3, 3, 0],\n    [-1, 1, 3]])\n    >>> A.cholesky() * A.cholesky().T\n    Matrix([\n    [25, 15, -5],\n    [15, 18,  0],\n    [-5,  0, 11]])\n\n    The matrix can have complex entries:\n\n    >>> from sympy import I\n    >>> A = Matrix(((9, 3*I), (-3*I, 5)))\n    >>> A.cholesky()\n    Matrix([\n    [ 3, 0],\n    [-I, 2]])\n    >>> A.cholesky() * A.cholesky().H\n    Matrix([\n    [   9, 3*I],\n    [-3*I,   5]])\n\n    Non-hermitian Cholesky-type decomposition may be useful when the\n    matrix is not positive-definite.\n\n    >>> A = Matrix([[1, 2], [2, 1]])\n    >>> L = A.cholesky(hermitian=False)\n    >>> L\n    Matrix([\n    [1,         0],\n    [2, sqrt(3)*I]])\n    >>> L*L.T == A\n    True\n\n    See Also\n    ========\n\n    sympy.matrices.dense.DenseMatrix.LDLdecomposition\n    sympy.matrices.matrices.MatrixBase.LUdecomposition\n    QRdecomposition\n    '
    from .dense import MutableDenseMatrix
    if not M.is_square:
        raise NonSquareMatrixError('Matrix must be square.')
    if hermitian and (not M.is_hermitian):
        raise ValueError('Matrix must be Hermitian.')
    if not hermitian and (not M.is_symmetric()):
        raise ValueError('Matrix must be symmetric.')
    L = MutableDenseMatrix.zeros(M.rows, M.rows)
    if hermitian:
        for i in range(M.rows):
            for j in range(i):
                L[i, j] = 1 / L[j, j] * (M[i, j] - sum((L[i, k] * L[j, k].conjugate() for k in range(j))))
            Lii2 = M[i, i] - sum((L[i, k] * L[i, k].conjugate() for k in range(i)))
            if Lii2.is_positive is False:
                raise NonPositiveDefiniteMatrixError('Matrix must be positive-definite')
            L[i, i] = sqrt(Lii2)
    else:
        for i in range(M.rows):
            for j in range(i):
                L[i, j] = 1 / L[j, j] * (M[i, j] - sum((L[i, k] * L[j, k] for k in range(j))))
            L[i, i] = sqrt(M[i, i] - sum((L[i, k] ** 2 for k in range(i))))
    return M._new(L)

def _cholesky_sparse(M, hermitian=True):
    if False:
        i = 10
        return i + 15
    '\n    Returns the Cholesky decomposition L of a matrix A\n    such that L * L.T = A\n\n    A must be a square, symmetric, positive-definite\n    and non-singular matrix\n\n    Examples\n    ========\n\n    >>> from sympy import SparseMatrix\n    >>> A = SparseMatrix(((25,15,-5),(15,18,0),(-5,0,11)))\n    >>> A.cholesky()\n    Matrix([\n    [ 5, 0, 0],\n    [ 3, 3, 0],\n    [-1, 1, 3]])\n    >>> A.cholesky() * A.cholesky().T == A\n    True\n\n    The matrix can have complex entries:\n\n    >>> from sympy import I\n    >>> A = SparseMatrix(((9, 3*I), (-3*I, 5)))\n    >>> A.cholesky()\n    Matrix([\n    [ 3, 0],\n    [-I, 2]])\n    >>> A.cholesky() * A.cholesky().H\n    Matrix([\n    [   9, 3*I],\n    [-3*I,   5]])\n\n    Non-hermitian Cholesky-type decomposition may be useful when the\n    matrix is not positive-definite.\n\n    >>> A = SparseMatrix([[1, 2], [2, 1]])\n    >>> L = A.cholesky(hermitian=False)\n    >>> L\n    Matrix([\n    [1,         0],\n    [2, sqrt(3)*I]])\n    >>> L*L.T == A\n    True\n\n    See Also\n    ========\n\n    sympy.matrices.sparse.SparseMatrix.LDLdecomposition\n    sympy.matrices.matrices.MatrixBase.LUdecomposition\n    QRdecomposition\n    '
    from .dense import MutableDenseMatrix
    if not M.is_square:
        raise NonSquareMatrixError('Matrix must be square.')
    if hermitian and (not M.is_hermitian):
        raise ValueError('Matrix must be Hermitian.')
    if not hermitian and (not M.is_symmetric()):
        raise ValueError('Matrix must be symmetric.')
    dps = _get_intermediate_simp(expand_mul, expand_mul)
    Crowstruc = M.row_structure_symbolic_cholesky()
    C = MutableDenseMatrix.zeros(M.rows)
    for i in range(len(Crowstruc)):
        for j in Crowstruc[i]:
            if i != j:
                C[i, j] = M[i, j]
                summ = 0
                for p1 in Crowstruc[i]:
                    if p1 < j:
                        for p2 in Crowstruc[j]:
                            if p2 < j:
                                if p1 == p2:
                                    if hermitian:
                                        summ += C[i, p1] * C[j, p1].conjugate()
                                    else:
                                        summ += C[i, p1] * C[j, p1]
                            else:
                                break
                        else:
                            break
                C[i, j] = dps((C[i, j] - summ) / C[j, j])
            else:
                C[j, j] = M[j, j]
                summ = 0
                for k in Crowstruc[j]:
                    if k < j:
                        if hermitian:
                            summ += C[j, k] * C[j, k].conjugate()
                        else:
                            summ += C[j, k] ** 2
                    else:
                        break
                Cjj2 = dps(C[j, j] - summ)
                if hermitian and Cjj2.is_positive is False:
                    raise NonPositiveDefiniteMatrixError('Matrix must be positive-definite')
                C[j, j] = sqrt(Cjj2)
    return M._new(C)

def _LDLdecomposition(M, hermitian=True):
    if False:
        while True:
            i = 10
    'Returns the LDL Decomposition (L, D) of matrix A,\n    such that L * D * L.H == A if hermitian flag is True, or\n    L * D * L.T == A if hermitian is False.\n    This method eliminates the use of square root.\n    Further this ensures that all the diagonal entries of L are 1.\n    A must be a Hermitian positive-definite matrix if hermitian is True,\n    or a symmetric matrix otherwise.\n\n    Examples\n    ========\n\n    >>> from sympy import Matrix, eye\n    >>> A = Matrix(((25, 15, -5), (15, 18, 0), (-5, 0, 11)))\n    >>> L, D = A.LDLdecomposition()\n    >>> L\n    Matrix([\n    [   1,   0, 0],\n    [ 3/5,   1, 0],\n    [-1/5, 1/3, 1]])\n    >>> D\n    Matrix([\n    [25, 0, 0],\n    [ 0, 9, 0],\n    [ 0, 0, 9]])\n    >>> L * D * L.T * A.inv() == eye(A.rows)\n    True\n\n    The matrix can have complex entries:\n\n    >>> from sympy import I\n    >>> A = Matrix(((9, 3*I), (-3*I, 5)))\n    >>> L, D = A.LDLdecomposition()\n    >>> L\n    Matrix([\n    [   1, 0],\n    [-I/3, 1]])\n    >>> D\n    Matrix([\n    [9, 0],\n    [0, 4]])\n    >>> L*D*L.H == A\n    True\n\n    See Also\n    ========\n\n    sympy.matrices.dense.DenseMatrix.cholesky\n    sympy.matrices.matrices.MatrixBase.LUdecomposition\n    QRdecomposition\n    '
    from .dense import MutableDenseMatrix
    if not M.is_square:
        raise NonSquareMatrixError('Matrix must be square.')
    if hermitian and (not M.is_hermitian):
        raise ValueError('Matrix must be Hermitian.')
    if not hermitian and (not M.is_symmetric()):
        raise ValueError('Matrix must be symmetric.')
    D = MutableDenseMatrix.zeros(M.rows, M.rows)
    L = MutableDenseMatrix.eye(M.rows)
    if hermitian:
        for i in range(M.rows):
            for j in range(i):
                L[i, j] = 1 / D[j, j] * (M[i, j] - sum((L[i, k] * L[j, k].conjugate() * D[k, k] for k in range(j))))
            D[i, i] = M[i, i] - sum((L[i, k] * L[i, k].conjugate() * D[k, k] for k in range(i)))
            if D[i, i].is_positive is False:
                raise NonPositiveDefiniteMatrixError('Matrix must be positive-definite')
    else:
        for i in range(M.rows):
            for j in range(i):
                L[i, j] = 1 / D[j, j] * (M[i, j] - sum((L[i, k] * L[j, k] * D[k, k] for k in range(j))))
            D[i, i] = M[i, i] - sum((L[i, k] ** 2 * D[k, k] for k in range(i)))
    return (M._new(L), M._new(D))

def _LDLdecomposition_sparse(M, hermitian=True):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns the LDL Decomposition (matrices ``L`` and ``D``) of matrix\n    ``A``, such that ``L * D * L.T == A``. ``A`` must be a square,\n    symmetric, positive-definite and non-singular.\n\n    This method eliminates the use of square root and ensures that all\n    the diagonal entries of L are 1.\n\n    Examples\n    ========\n\n    >>> from sympy import SparseMatrix\n    >>> A = SparseMatrix(((25, 15, -5), (15, 18, 0), (-5, 0, 11)))\n    >>> L, D = A.LDLdecomposition()\n    >>> L\n    Matrix([\n    [   1,   0, 0],\n    [ 3/5,   1, 0],\n    [-1/5, 1/3, 1]])\n    >>> D\n    Matrix([\n    [25, 0, 0],\n    [ 0, 9, 0],\n    [ 0, 0, 9]])\n    >>> L * D * L.T == A\n    True\n\n    '
    from .dense import MutableDenseMatrix
    if not M.is_square:
        raise NonSquareMatrixError('Matrix must be square.')
    if hermitian and (not M.is_hermitian):
        raise ValueError('Matrix must be Hermitian.')
    if not hermitian and (not M.is_symmetric()):
        raise ValueError('Matrix must be symmetric.')
    dps = _get_intermediate_simp(expand_mul, expand_mul)
    Lrowstruc = M.row_structure_symbolic_cholesky()
    L = MutableDenseMatrix.eye(M.rows)
    D = MutableDenseMatrix.zeros(M.rows, M.cols)
    for i in range(len(Lrowstruc)):
        for j in Lrowstruc[i]:
            if i != j:
                L[i, j] = M[i, j]
                summ = 0
                for p1 in Lrowstruc[i]:
                    if p1 < j:
                        for p2 in Lrowstruc[j]:
                            if p2 < j:
                                if p1 == p2:
                                    if hermitian:
                                        summ += L[i, p1] * L[j, p1].conjugate() * D[p1, p1]
                                    else:
                                        summ += L[i, p1] * L[j, p1] * D[p1, p1]
                            else:
                                break
                    else:
                        break
                L[i, j] = dps((L[i, j] - summ) / D[j, j])
            else:
                D[i, i] = M[i, i]
                summ = 0
                for k in Lrowstruc[i]:
                    if k < i:
                        if hermitian:
                            summ += L[i, k] * L[i, k].conjugate() * D[k, k]
                        else:
                            summ += L[i, k] ** 2 * D[k, k]
                    else:
                        break
                D[i, i] = dps(D[i, i] - summ)
                if hermitian and D[i, i].is_positive is False:
                    raise NonPositiveDefiniteMatrixError('Matrix must be positive-definite')
    return (M._new(L), M._new(D))

def _LUdecomposition(M, iszerofunc=_iszero, simpfunc=None, rankcheck=False):
    if False:
        print('Hello World!')
    'Returns (L, U, perm) where L is a lower triangular matrix with unit\n    diagonal, U is an upper triangular matrix, and perm is a list of row\n    swap index pairs. If A is the original matrix, then\n    ``A = (L*U).permuteBkwd(perm)``, and the row permutation matrix P such\n    that $P A = L U$ can be computed by ``P = eye(A.rows).permuteFwd(perm)``.\n\n    See documentation for LUCombined for details about the keyword argument\n    rankcheck, iszerofunc, and simpfunc.\n\n    Parameters\n    ==========\n\n    rankcheck : bool, optional\n        Determines if this function should detect the rank\n        deficiency of the matrixis and should raise a\n        ``ValueError``.\n\n    iszerofunc : function, optional\n        A function which determines if a given expression is zero.\n\n        The function should be a callable that takes a single\n        SymPy expression and returns a 3-valued boolean value\n        ``True``, ``False``, or ``None``.\n\n        It is internally used by the pivot searching algorithm.\n        See the notes section for a more information about the\n        pivot searching algorithm.\n\n    simpfunc : function or None, optional\n        A function that simplifies the input.\n\n        If this is specified as a function, this function should be\n        a callable that takes a single SymPy expression and returns\n        an another SymPy expression that is algebraically\n        equivalent.\n\n        If ``None``, it indicates that the pivot search algorithm\n        should not attempt to simplify any candidate pivots.\n\n        It is internally used by the pivot searching algorithm.\n        See the notes section for a more information about the\n        pivot searching algorithm.\n\n    Examples\n    ========\n\n    >>> from sympy import Matrix\n    >>> a = Matrix([[4, 3], [6, 3]])\n    >>> L, U, _ = a.LUdecomposition()\n    >>> L\n    Matrix([\n    [  1, 0],\n    [3/2, 1]])\n    >>> U\n    Matrix([\n    [4,    3],\n    [0, -3/2]])\n\n    See Also\n    ========\n\n    sympy.matrices.dense.DenseMatrix.cholesky\n    sympy.matrices.dense.DenseMatrix.LDLdecomposition\n    QRdecomposition\n    LUdecomposition_Simple\n    LUdecompositionFF\n    LUsolve\n    '
    (combined, p) = M.LUdecomposition_Simple(iszerofunc=iszerofunc, simpfunc=simpfunc, rankcheck=rankcheck)

    def entry_L(i, j):
        if False:
            print('Hello World!')
        if i < j:
            return M.zero
        elif i == j:
            return M.one
        elif j < combined.cols:
            return combined[i, j]
        return M.zero

    def entry_U(i, j):
        if False:
            return 10
        return M.zero if i > j else combined[i, j]
    L = M._new(combined.rows, combined.rows, entry_L)
    U = M._new(combined.rows, combined.cols, entry_U)
    return (L, U, p)

def _LUdecomposition_Simple(M, iszerofunc=_iszero, simpfunc=None, rankcheck=False):
    if False:
        i = 10
        return i + 15
    'Compute the PLU decomposition of the matrix.\n\n    Parameters\n    ==========\n\n    rankcheck : bool, optional\n        Determines if this function should detect the rank\n        deficiency of the matrixis and should raise a\n        ``ValueError``.\n\n    iszerofunc : function, optional\n        A function which determines if a given expression is zero.\n\n        The function should be a callable that takes a single\n        SymPy expression and returns a 3-valued boolean value\n        ``True``, ``False``, or ``None``.\n\n        It is internally used by the pivot searching algorithm.\n        See the notes section for a more information about the\n        pivot searching algorithm.\n\n    simpfunc : function or None, optional\n        A function that simplifies the input.\n\n        If this is specified as a function, this function should be\n        a callable that takes a single SymPy expression and returns\n        an another SymPy expression that is algebraically\n        equivalent.\n\n        If ``None``, it indicates that the pivot search algorithm\n        should not attempt to simplify any candidate pivots.\n\n        It is internally used by the pivot searching algorithm.\n        See the notes section for a more information about the\n        pivot searching algorithm.\n\n    Returns\n    =======\n\n    (lu, row_swaps) : (Matrix, list)\n        If the original matrix is a $m, n$ matrix:\n\n        *lu* is a $m, n$ matrix, which contains result of the\n        decomposition in a compressed form. See the notes section\n        to see how the matrix is compressed.\n\n        *row_swaps* is a $m$-element list where each element is a\n        pair of row exchange indices.\n\n        ``A = (L*U).permute_backward(perm)``, and the row\n        permutation matrix $P$ from the formula $P A = L U$ can be\n        computed by ``P=eye(A.row).permute_forward(perm)``.\n\n    Raises\n    ======\n\n    ValueError\n        Raised if ``rankcheck=True`` and the matrix is found to\n        be rank deficient during the computation.\n\n    Notes\n    =====\n\n    About the PLU decomposition:\n\n    PLU decomposition is a generalization of a LU decomposition\n    which can be extended for rank-deficient matrices.\n\n    It can further be generalized for non-square matrices, and this\n    is the notation that SymPy is using.\n\n    PLU decomposition is a decomposition of a $m, n$ matrix $A$ in\n    the form of $P A = L U$ where\n\n    * $L$ is a $m, m$ lower triangular matrix with unit diagonal\n        entries.\n    * $U$ is a $m, n$ upper triangular matrix.\n    * $P$ is a $m, m$ permutation matrix.\n\n    So, for a square matrix, the decomposition would look like:\n\n    .. math::\n        L = \\begin{bmatrix}\n        1 & 0 & 0 & \\cdots & 0 \\\\\n        L_{1, 0} & 1 & 0 & \\cdots & 0 \\\\\n        L_{2, 0} & L_{2, 1} & 1 & \\cdots & 0 \\\\\n        \\vdots & \\vdots & \\vdots & \\ddots & \\vdots \\\\\n        L_{n-1, 0} & L_{n-1, 1} & L_{n-1, 2} & \\cdots & 1\n        \\end{bmatrix}\n\n    .. math::\n        U = \\begin{bmatrix}\n        U_{0, 0} & U_{0, 1} & U_{0, 2} & \\cdots & U_{0, n-1} \\\\\n        0 & U_{1, 1} & U_{1, 2} & \\cdots & U_{1, n-1} \\\\\n        0 & 0 & U_{2, 2} & \\cdots & U_{2, n-1} \\\\\n        \\vdots & \\vdots & \\vdots & \\ddots & \\vdots \\\\\n        0 & 0 & 0 & \\cdots & U_{n-1, n-1}\n        \\end{bmatrix}\n\n    And for a matrix with more rows than the columns,\n    the decomposition would look like:\n\n    .. math::\n        L = \\begin{bmatrix}\n        1 & 0 & 0 & \\cdots & 0 & 0 & \\cdots & 0 \\\\\n        L_{1, 0} & 1 & 0 & \\cdots & 0 & 0 & \\cdots & 0 \\\\\n        L_{2, 0} & L_{2, 1} & 1 & \\cdots & 0 & 0 & \\cdots & 0 \\\\\n        \\vdots & \\vdots & \\vdots & \\ddots & \\vdots & \\vdots & \\ddots\n        & \\vdots \\\\\n        L_{n-1, 0} & L_{n-1, 1} & L_{n-1, 2} & \\cdots & 1 & 0\n        & \\cdots & 0 \\\\\n        L_{n, 0} & L_{n, 1} & L_{n, 2} & \\cdots & L_{n, n-1} & 1\n        & \\cdots & 0 \\\\\n        \\vdots & \\vdots & \\vdots & \\ddots & \\vdots & \\vdots\n        & \\ddots & \\vdots \\\\\n        L_{m-1, 0} & L_{m-1, 1} & L_{m-1, 2} & \\cdots & L_{m-1, n-1}\n        & 0 & \\cdots & 1 \\\\\n        \\end{bmatrix}\n\n    .. math::\n        U = \\begin{bmatrix}\n        U_{0, 0} & U_{0, 1} & U_{0, 2} & \\cdots & U_{0, n-1} \\\\\n        0 & U_{1, 1} & U_{1, 2} & \\cdots & U_{1, n-1} \\\\\n        0 & 0 & U_{2, 2} & \\cdots & U_{2, n-1} \\\\\n        \\vdots & \\vdots & \\vdots & \\ddots & \\vdots \\\\\n        0 & 0 & 0 & \\cdots & U_{n-1, n-1} \\\\\n        0 & 0 & 0 & \\cdots & 0 \\\\\n        \\vdots & \\vdots & \\vdots & \\ddots & \\vdots \\\\\n        0 & 0 & 0 & \\cdots & 0\n        \\end{bmatrix}\n\n    Finally, for a matrix with more columns than the rows, the\n    decomposition would look like:\n\n    .. math::\n        L = \\begin{bmatrix}\n        1 & 0 & 0 & \\cdots & 0 \\\\\n        L_{1, 0} & 1 & 0 & \\cdots & 0 \\\\\n        L_{2, 0} & L_{2, 1} & 1 & \\cdots & 0 \\\\\n        \\vdots & \\vdots & \\vdots & \\ddots & \\vdots \\\\\n        L_{m-1, 0} & L_{m-1, 1} & L_{m-1, 2} & \\cdots & 1\n        \\end{bmatrix}\n\n    .. math::\n        U = \\begin{bmatrix}\n        U_{0, 0} & U_{0, 1} & U_{0, 2} & \\cdots & U_{0, m-1}\n        & \\cdots & U_{0, n-1} \\\\\n        0 & U_{1, 1} & U_{1, 2} & \\cdots & U_{1, m-1}\n        & \\cdots & U_{1, n-1} \\\\\n        0 & 0 & U_{2, 2} & \\cdots & U_{2, m-1}\n        & \\cdots & U_{2, n-1} \\\\\n        \\vdots & \\vdots & \\vdots & \\ddots & \\vdots\n        & \\cdots & \\vdots \\\\\n        0 & 0 & 0 & \\cdots & U_{m-1, m-1}\n        & \\cdots & U_{m-1, n-1} \\\\\n        \\end{bmatrix}\n\n    About the compressed LU storage:\n\n    The results of the decomposition are often stored in compressed\n    forms rather than returning $L$ and $U$ matrices individually.\n\n    It may be less intiuitive, but it is commonly used for a lot of\n    numeric libraries because of the efficiency.\n\n    The storage matrix is defined as following for this specific\n    method:\n\n    * The subdiagonal elements of $L$ are stored in the subdiagonal\n        portion of $LU$, that is $LU_{i, j} = L_{i, j}$ whenever\n        $i > j$.\n    * The elements on the diagonal of $L$ are all 1, and are not\n        explicitly stored.\n    * $U$ is stored in the upper triangular portion of $LU$, that is\n        $LU_{i, j} = U_{i, j}$ whenever $i <= j$.\n    * For a case of $m > n$, the right side of the $L$ matrix is\n        trivial to store.\n    * For a case of $m < n$, the below side of the $U$ matrix is\n        trivial to store.\n\n    So, for a square matrix, the compressed output matrix would be:\n\n    .. math::\n        LU = \\begin{bmatrix}\n        U_{0, 0} & U_{0, 1} & U_{0, 2} & \\cdots & U_{0, n-1} \\\\\n        L_{1, 0} & U_{1, 1} & U_{1, 2} & \\cdots & U_{1, n-1} \\\\\n        L_{2, 0} & L_{2, 1} & U_{2, 2} & \\cdots & U_{2, n-1} \\\\\n        \\vdots & \\vdots & \\vdots & \\ddots & \\vdots \\\\\n        L_{n-1, 0} & L_{n-1, 1} & L_{n-1, 2} & \\cdots & U_{n-1, n-1}\n        \\end{bmatrix}\n\n    For a matrix with more rows than the columns, the compressed\n    output matrix would be:\n\n    .. math::\n        LU = \\begin{bmatrix}\n        U_{0, 0} & U_{0, 1} & U_{0, 2} & \\cdots & U_{0, n-1} \\\\\n        L_{1, 0} & U_{1, 1} & U_{1, 2} & \\cdots & U_{1, n-1} \\\\\n        L_{2, 0} & L_{2, 1} & U_{2, 2} & \\cdots & U_{2, n-1} \\\\\n        \\vdots & \\vdots & \\vdots & \\ddots & \\vdots \\\\\n        L_{n-1, 0} & L_{n-1, 1} & L_{n-1, 2} & \\cdots\n        & U_{n-1, n-1} \\\\\n        \\vdots & \\vdots & \\vdots & \\ddots & \\vdots \\\\\n        L_{m-1, 0} & L_{m-1, 1} & L_{m-1, 2} & \\cdots\n        & L_{m-1, n-1} \\\\\n        \\end{bmatrix}\n\n    For a matrix with more columns than the rows, the compressed\n    output matrix would be:\n\n    .. math::\n        LU = \\begin{bmatrix}\n        U_{0, 0} & U_{0, 1} & U_{0, 2} & \\cdots & U_{0, m-1}\n        & \\cdots & U_{0, n-1} \\\\\n        L_{1, 0} & U_{1, 1} & U_{1, 2} & \\cdots & U_{1, m-1}\n        & \\cdots & U_{1, n-1} \\\\\n        L_{2, 0} & L_{2, 1} & U_{2, 2} & \\cdots & U_{2, m-1}\n        & \\cdots & U_{2, n-1} \\\\\n        \\vdots & \\vdots & \\vdots & \\ddots & \\vdots\n        & \\cdots & \\vdots \\\\\n        L_{m-1, 0} & L_{m-1, 1} & L_{m-1, 2} & \\cdots & U_{m-1, m-1}\n        & \\cdots & U_{m-1, n-1} \\\\\n        \\end{bmatrix}\n\n    About the pivot searching algorithm:\n\n    When a matrix contains symbolic entries, the pivot search algorithm\n    differs from the case where every entry can be categorized as zero or\n    nonzero.\n    The algorithm searches column by column through the submatrix whose\n    top left entry coincides with the pivot position.\n    If it exists, the pivot is the first entry in the current search\n    column that iszerofunc guarantees is nonzero.\n    If no such candidate exists, then each candidate pivot is simplified\n    if simpfunc is not None.\n    The search is repeated, with the difference that a candidate may be\n    the pivot if ``iszerofunc()`` cannot guarantee that it is nonzero.\n    In the second search the pivot is the first candidate that\n    iszerofunc can guarantee is nonzero.\n    If no such candidate exists, then the pivot is the first candidate\n    for which iszerofunc returns None.\n    If no such candidate exists, then the search is repeated in the next\n    column to the right.\n    The pivot search algorithm differs from the one in ``rref()``, which\n    relies on ``_find_reasonable_pivot()``.\n    Future versions of ``LUdecomposition_simple()`` may use\n    ``_find_reasonable_pivot()``.\n\n    See Also\n    ========\n\n    sympy.matrices.matrices.MatrixBase.LUdecomposition\n    LUdecompositionFF\n    LUsolve\n    '
    if rankcheck:
        pass
    if S.Zero in M.shape:
        return (M.zeros(M.rows, M.cols), [])
    dps = _get_intermediate_simp()
    lu = M.as_mutable()
    row_swaps = []
    pivot_col = 0
    for pivot_row in range(0, lu.rows - 1):
        iszeropivot = True
        while pivot_col != M.cols and iszeropivot:
            sub_col = (lu[r, pivot_col] for r in range(pivot_row, M.rows))
            (pivot_row_offset, pivot_value, is_assumed_non_zero, ind_simplified_pairs) = _find_reasonable_pivot_naive(sub_col, iszerofunc, simpfunc)
            iszeropivot = pivot_value is None
            if iszeropivot:
                pivot_col += 1
        if rankcheck and pivot_col != pivot_row:
            raise ValueError('Rank of matrix is strictly less than number of rows or columns. Pass keyword argument rankcheck=False to compute the LU decomposition of this matrix.')
        candidate_pivot_row = None if pivot_row_offset is None else pivot_row + pivot_row_offset
        if candidate_pivot_row is None and iszeropivot:
            return (lu, row_swaps)
        for (offset, val) in ind_simplified_pairs:
            lu[pivot_row + offset, pivot_col] = val
        if pivot_row != candidate_pivot_row:
            row_swaps.append([pivot_row, candidate_pivot_row])
            (lu[pivot_row, 0:pivot_row], lu[candidate_pivot_row, 0:pivot_row]) = (lu[candidate_pivot_row, 0:pivot_row], lu[pivot_row, 0:pivot_row])
            (lu[pivot_row, pivot_col:lu.cols], lu[candidate_pivot_row, pivot_col:lu.cols]) = (lu[candidate_pivot_row, pivot_col:lu.cols], lu[pivot_row, pivot_col:lu.cols])
        start_col = pivot_col + 1
        for row in range(pivot_row + 1, lu.rows):
            lu[row, pivot_row] = dps(lu[row, pivot_col] / lu[pivot_row, pivot_col])
            for c in range(start_col, lu.cols):
                lu[row, c] = dps(lu[row, c] - lu[row, pivot_row] * lu[pivot_row, c])
        if pivot_row != pivot_col:
            for row in range(pivot_row + 1, lu.rows):
                lu[row, pivot_col] = M.zero
        pivot_col += 1
        if pivot_col == lu.cols:
            return (lu, row_swaps)
    if rankcheck:
        if iszerofunc(lu[Min(lu.rows, lu.cols) - 1, Min(lu.rows, lu.cols) - 1]):
            raise ValueError('Rank of matrix is strictly less than number of rows or columns. Pass keyword argument rankcheck=False to compute the LU decomposition of this matrix.')
    return (lu, row_swaps)

def _LUdecompositionFF(M):
    if False:
        for i in range(10):
            print('nop')
    'Compute a fraction-free LU decomposition.\n\n    Returns 4 matrices P, L, D, U such that PA = L D**-1 U.\n    If the elements of the matrix belong to some integral domain I, then all\n    elements of L, D and U are guaranteed to belong to I.\n\n    See Also\n    ========\n\n    sympy.matrices.matrices.MatrixBase.LUdecomposition\n    LUdecomposition_Simple\n    LUsolve\n\n    References\n    ==========\n\n    .. [1] W. Zhou & D.J. Jeffrey, "Fraction-free matrix factors: new forms\n        for LU and QR factors". Frontiers in Computer Science in China,\n        Vol 2, no. 1, pp. 67-80, 2008.\n    '
    from sympy.matrices import SparseMatrix
    zeros = SparseMatrix.zeros
    eye = SparseMatrix.eye
    (n, m) = (M.rows, M.cols)
    (U, L, P) = (M.as_mutable(), eye(n), eye(n))
    DD = zeros(n, n)
    oldpivot = 1
    for k in range(n - 1):
        if U[k, k] == 0:
            for kpivot in range(k + 1, n):
                if U[kpivot, k]:
                    break
            else:
                raise ValueError('Matrix is not full rank')
            (U[k, k:], U[kpivot, k:]) = (U[kpivot, k:], U[k, k:])
            (L[k, :k], L[kpivot, :k]) = (L[kpivot, :k], L[k, :k])
            (P[k, :], P[kpivot, :]) = (P[kpivot, :], P[k, :])
        L[k, k] = Ukk = U[k, k]
        DD[k, k] = oldpivot * Ukk
        for i in range(k + 1, n):
            L[i, k] = Uik = U[i, k]
            for j in range(k + 1, m):
                U[i, j] = (Ukk * U[i, j] - U[k, j] * Uik) / oldpivot
            U[i, k] = 0
        oldpivot = Ukk
    DD[n - 1, n - 1] = oldpivot
    return (P, L, DD, U)

def _singular_value_decomposition(A):
    if False:
        return 10
    'Returns a Condensed Singular Value decomposition.\n\n    Explanation\n    ===========\n\n    A Singular Value decomposition is a decomposition in the form $A = U \\Sigma V^H$\n    where\n\n    - $U, V$ are column orthogonal matrix.\n    - $\\Sigma$ is a diagonal matrix, where the main diagonal contains singular\n      values of matrix A.\n\n    A column orthogonal matrix satisfies\n    $\\mathbb{I} = U^H U$ while a full orthogonal matrix satisfies\n    relation $\\mathbb{I} = U U^H = U^H U$ where $\\mathbb{I}$ is an identity\n    matrix with matching dimensions.\n\n    For matrices which are not square or are rank-deficient, it is\n    sufficient to return a column orthogonal matrix because augmenting\n    them may introduce redundant computations.\n    In condensed Singular Value Decomposition we only return column orthogonal\n    matrices because of this reason\n\n    If you want to augment the results to return a full orthogonal\n    decomposition, you should use the following procedures.\n\n    - Augment the $U, V$ matrices with columns that are orthogonal to every\n      other columns and make it square.\n    - Augment the $\\Sigma$ matrix with zero rows to make it have the same\n      shape as the original matrix.\n\n    The procedure will be illustrated in the examples section.\n\n    Examples\n    ========\n\n    we take a full rank matrix first:\n\n    >>> from sympy import Matrix\n    >>> A = Matrix([[1, 2],[2,1]])\n    >>> U, S, V = A.singular_value_decomposition()\n    >>> U\n    Matrix([\n    [ sqrt(2)/2, sqrt(2)/2],\n    [-sqrt(2)/2, sqrt(2)/2]])\n    >>> S\n    Matrix([\n    [1, 0],\n    [0, 3]])\n    >>> V\n    Matrix([\n    [-sqrt(2)/2, sqrt(2)/2],\n    [ sqrt(2)/2, sqrt(2)/2]])\n\n    If a matrix if square and full rank both U, V\n    are orthogonal in both directions\n\n    >>> U * U.H\n    Matrix([\n    [1, 0],\n    [0, 1]])\n    >>> U.H * U\n    Matrix([\n    [1, 0],\n    [0, 1]])\n\n    >>> V * V.H\n    Matrix([\n    [1, 0],\n    [0, 1]])\n    >>> V.H * V\n    Matrix([\n    [1, 0],\n    [0, 1]])\n    >>> A == U * S * V.H\n    True\n\n    >>> C = Matrix([\n    ...         [1, 0, 0, 0, 2],\n    ...         [0, 0, 3, 0, 0],\n    ...         [0, 0, 0, 0, 0],\n    ...         [0, 2, 0, 0, 0],\n    ...     ])\n    >>> U, S, V = C.singular_value_decomposition()\n\n    >>> V.H * V\n    Matrix([\n    [1, 0, 0],\n    [0, 1, 0],\n    [0, 0, 1]])\n    >>> V * V.H\n    Matrix([\n    [1/5, 0, 0, 0, 2/5],\n    [  0, 1, 0, 0,   0],\n    [  0, 0, 1, 0,   0],\n    [  0, 0, 0, 0,   0],\n    [2/5, 0, 0, 0, 4/5]])\n\n    If you want to augment the results to be a full orthogonal\n    decomposition, you should augment $V$ with an another orthogonal\n    column.\n\n    You are able to append an arbitrary standard basis that are linearly\n    independent to every other columns and you can run the Gram-Schmidt\n    process to make them augmented as orthogonal basis.\n\n    >>> V_aug = V.row_join(Matrix([[0,0,0,0,1],\n    ... [0,0,0,1,0]]).H)\n    >>> V_aug = V_aug.QRdecomposition()[0]\n    >>> V_aug\n    Matrix([\n    [0,   sqrt(5)/5, 0, -2*sqrt(5)/5, 0],\n    [1,           0, 0,            0, 0],\n    [0,           0, 1,            0, 0],\n    [0,           0, 0,            0, 1],\n    [0, 2*sqrt(5)/5, 0,    sqrt(5)/5, 0]])\n    >>> V_aug.H * V_aug\n    Matrix([\n    [1, 0, 0, 0, 0],\n    [0, 1, 0, 0, 0],\n    [0, 0, 1, 0, 0],\n    [0, 0, 0, 1, 0],\n    [0, 0, 0, 0, 1]])\n    >>> V_aug * V_aug.H\n    Matrix([\n    [1, 0, 0, 0, 0],\n    [0, 1, 0, 0, 0],\n    [0, 0, 1, 0, 0],\n    [0, 0, 0, 1, 0],\n    [0, 0, 0, 0, 1]])\n\n    Similarly we augment U\n\n    >>> U_aug = U.row_join(Matrix([0,0,1,0]))\n    >>> U_aug = U_aug.QRdecomposition()[0]\n    >>> U_aug\n    Matrix([\n    [0, 1, 0, 0],\n    [0, 0, 1, 0],\n    [0, 0, 0, 1],\n    [1, 0, 0, 0]])\n\n    >>> U_aug.H * U_aug\n    Matrix([\n    [1, 0, 0, 0],\n    [0, 1, 0, 0],\n    [0, 0, 1, 0],\n    [0, 0, 0, 1]])\n    >>> U_aug * U_aug.H\n    Matrix([\n    [1, 0, 0, 0],\n    [0, 1, 0, 0],\n    [0, 0, 1, 0],\n    [0, 0, 0, 1]])\n\n    We add 2 zero columns and one row to S\n\n    >>> S_aug = S.col_join(Matrix([[0,0,0]]))\n    >>> S_aug = S_aug.row_join(Matrix([[0,0,0,0],\n    ... [0,0,0,0]]).H)\n    >>> S_aug\n    Matrix([\n    [2,       0, 0, 0, 0],\n    [0, sqrt(5), 0, 0, 0],\n    [0,       0, 3, 0, 0],\n    [0,       0, 0, 0, 0]])\n\n\n\n    >>> U_aug * S_aug * V_aug.H == C\n    True\n\n    '
    AH = A.H
    (m, n) = A.shape
    if m >= n:
        (V, S) = (AH * A).diagonalize()
        ranked = []
        for (i, x) in enumerate(S.diagonal()):
            if not x.is_zero:
                ranked.append(i)
        V = V[:, ranked]
        Singular_vals = [sqrt(S[i, i]) for i in range(S.rows) if i in ranked]
        S = S.diag(*Singular_vals)
        (V, _) = V.QRdecomposition()
        U = A * V * S.inv()
    else:
        (U, S) = (A * AH).diagonalize()
        ranked = []
        for (i, x) in enumerate(S.diagonal()):
            if not x.is_zero:
                ranked.append(i)
        U = U[:, ranked]
        Singular_vals = [sqrt(S[i, i]) for i in range(S.rows) if i in ranked]
        S = S.diag(*Singular_vals)
        (U, _) = U.QRdecomposition()
        V = AH * U * S.inv()
    return (U, S, V)

def _QRdecomposition_optional(M, normalize=True):
    if False:
        print('Hello World!')

    def dot(u, v):
        if False:
            i = 10
            return i + 15
        return u.dot(v, hermitian=True)
    dps = _get_intermediate_simp(expand_mul, expand_mul)
    A = M.as_mutable()
    ranked = []
    Q = A
    R = A.zeros(A.cols)
    for j in range(A.cols):
        for i in range(j):
            if Q[:, i].is_zero_matrix:
                continue
            R[i, j] = dot(Q[:, i], Q[:, j]) / dot(Q[:, i], Q[:, i])
            R[i, j] = dps(R[i, j])
            Q[:, j] -= Q[:, i] * R[i, j]
        Q[:, j] = dps(Q[:, j])
        if Q[:, j].is_zero_matrix is not True:
            ranked.append(j)
            R[j, j] = M.one
    Q = Q.extract(range(Q.rows), ranked)
    R = R.extract(ranked, range(R.cols))
    if normalize:
        for i in range(Q.cols):
            norm = Q[:, i].norm()
            Q[:, i] /= norm
            R[i, :] *= norm
    return (M.__class__(Q), M.__class__(R))

def _QRdecomposition(M):
    if False:
        i = 10
        return i + 15
    'Returns a QR decomposition.\n\n    Explanation\n    ===========\n\n    A QR decomposition is a decomposition in the form $A = Q R$\n    where\n\n    - $Q$ is a column orthogonal matrix.\n    - $R$ is a upper triangular (trapezoidal) matrix.\n\n    A column orthogonal matrix satisfies\n    $\\mathbb{I} = Q^H Q$ while a full orthogonal matrix satisfies\n    relation $\\mathbb{I} = Q Q^H = Q^H Q$ where $I$ is an identity\n    matrix with matching dimensions.\n\n    For matrices which are not square or are rank-deficient, it is\n    sufficient to return a column orthogonal matrix because augmenting\n    them may introduce redundant computations.\n    And an another advantage of this is that you can easily inspect the\n    matrix rank by counting the number of columns of $Q$.\n\n    If you want to augment the results to return a full orthogonal\n    decomposition, you should use the following procedures.\n\n    - Augment the $Q$ matrix with columns that are orthogonal to every\n      other columns and make it square.\n    - Augment the $R$ matrix with zero rows to make it have the same\n      shape as the original matrix.\n\n    The procedure will be illustrated in the examples section.\n\n    Examples\n    ========\n\n    A full rank matrix example:\n\n    >>> from sympy import Matrix\n    >>> A = Matrix([[12, -51, 4], [6, 167, -68], [-4, 24, -41]])\n    >>> Q, R = A.QRdecomposition()\n    >>> Q\n    Matrix([\n    [ 6/7, -69/175, -58/175],\n    [ 3/7, 158/175,   6/175],\n    [-2/7,    6/35,  -33/35]])\n    >>> R\n    Matrix([\n    [14,  21, -14],\n    [ 0, 175, -70],\n    [ 0,   0,  35]])\n\n    If the matrix is square and full rank, the $Q$ matrix becomes\n    orthogonal in both directions, and needs no augmentation.\n\n    >>> Q * Q.H\n    Matrix([\n    [1, 0, 0],\n    [0, 1, 0],\n    [0, 0, 1]])\n    >>> Q.H * Q\n    Matrix([\n    [1, 0, 0],\n    [0, 1, 0],\n    [0, 0, 1]])\n\n    >>> A == Q*R\n    True\n\n    A rank deficient matrix example:\n\n    >>> A = Matrix([[12, -51, 0], [6, 167, 0], [-4, 24, 0]])\n    >>> Q, R = A.QRdecomposition()\n    >>> Q\n    Matrix([\n    [ 6/7, -69/175],\n    [ 3/7, 158/175],\n    [-2/7,    6/35]])\n    >>> R\n    Matrix([\n    [14,  21, 0],\n    [ 0, 175, 0]])\n\n    QRdecomposition might return a matrix Q that is rectangular.\n    In this case the orthogonality condition might be satisfied as\n    $\\mathbb{I} = Q.H*Q$ but not in the reversed product\n    $\\mathbb{I} = Q * Q.H$.\n\n    >>> Q.H * Q\n    Matrix([\n    [1, 0],\n    [0, 1]])\n    >>> Q * Q.H\n    Matrix([\n    [27261/30625,   348/30625, -1914/6125],\n    [  348/30625, 30589/30625,   198/6125],\n    [ -1914/6125,    198/6125,   136/1225]])\n\n    If you want to augment the results to be a full orthogonal\n    decomposition, you should augment $Q$ with an another orthogonal\n    column.\n\n    You are able to append an identity matrix,\n    and you can run the Gram-Schmidt\n    process to make them augmented as orthogonal basis.\n\n    >>> Q_aug = Q.row_join(Matrix.eye(3))\n    >>> Q_aug = Q_aug.QRdecomposition()[0]\n    >>> Q_aug\n    Matrix([\n    [ 6/7, -69/175, 58/175],\n    [ 3/7, 158/175, -6/175],\n    [-2/7,    6/35,  33/35]])\n    >>> Q_aug.H * Q_aug\n    Matrix([\n    [1, 0, 0],\n    [0, 1, 0],\n    [0, 0, 1]])\n    >>> Q_aug * Q_aug.H\n    Matrix([\n    [1, 0, 0],\n    [0, 1, 0],\n    [0, 0, 1]])\n\n    Augmenting the $R$ matrix with zero row is straightforward.\n\n    >>> R_aug = R.col_join(Matrix([[0, 0, 0]]))\n    >>> R_aug\n    Matrix([\n    [14,  21, 0],\n    [ 0, 175, 0],\n    [ 0,   0, 0]])\n    >>> Q_aug * R_aug == A\n    True\n\n    A zero matrix example:\n\n    >>> from sympy import Matrix\n    >>> A = Matrix.zeros(3, 4)\n    >>> Q, R = A.QRdecomposition()\n\n    They may return matrices with zero rows and columns.\n\n    >>> Q\n    Matrix(3, 0, [])\n    >>> R\n    Matrix(0, 4, [])\n    >>> Q*R\n    Matrix([\n    [0, 0, 0, 0],\n    [0, 0, 0, 0],\n    [0, 0, 0, 0]])\n\n    As the same augmentation rule described above, $Q$ can be augmented\n    with columns of an identity matrix and $R$ can be augmented with\n    rows of a zero matrix.\n\n    >>> Q_aug = Q.row_join(Matrix.eye(3))\n    >>> R_aug = R.col_join(Matrix.zeros(3, 4))\n    >>> Q_aug * Q_aug.T\n    Matrix([\n    [1, 0, 0],\n    [0, 1, 0],\n    [0, 0, 1]])\n    >>> R_aug\n    Matrix([\n    [0, 0, 0, 0],\n    [0, 0, 0, 0],\n    [0, 0, 0, 0]])\n    >>> Q_aug * R_aug == A\n    True\n\n    See Also\n    ========\n\n    sympy.matrices.dense.DenseMatrix.cholesky\n    sympy.matrices.dense.DenseMatrix.LDLdecomposition\n    sympy.matrices.matrices.MatrixBase.LUdecomposition\n    QRsolve\n    '
    return _QRdecomposition_optional(M, normalize=True)

def _upper_hessenberg_decomposition(A):
    if False:
        for i in range(10):
            print('nop')
    'Converts a matrix into Hessenberg matrix H.\n\n    Returns 2 matrices H, P s.t.\n    $P H P^{T} = A$, where H is an upper hessenberg matrix\n    and P is an orthogonal matrix\n\n    Examples\n    ========\n\n    >>> from sympy import Matrix\n    >>> A = Matrix([\n    ...     [1,2,3],\n    ...     [-3,5,6],\n    ...     [4,-8,9],\n    ... ])\n    >>> H, P = A.upper_hessenberg_decomposition()\n    >>> H\n    Matrix([\n    [1,    6/5,    17/5],\n    [5, 213/25, -134/25],\n    [0, 216/25,  137/25]])\n    >>> P\n    Matrix([\n    [1,    0,   0],\n    [0, -3/5, 4/5],\n    [0,  4/5, 3/5]])\n    >>> P * H * P.H == A\n    True\n\n\n    References\n    ==========\n\n    .. [#] https://mathworld.wolfram.com/HessenbergDecomposition.html\n    '
    M = A.as_mutable()
    if not M.is_square:
        raise NonSquareMatrixError('Matrix must be square.')
    n = M.cols
    P = M.eye(n)
    H = M
    for j in range(n - 2):
        u = H[j + 1:, j]
        if u[1:, :].is_zero_matrix:
            continue
        if sign(u[0]) != 0:
            u[0] = u[0] + sign(u[0]) * u.norm()
        else:
            u[0] = u[0] + u.norm()
        v = u / u.norm()
        H[j + 1:, :] = H[j + 1:, :] - 2 * v * (v.H * H[j + 1:, :])
        H[:, j + 1:] = H[:, j + 1:] - H[:, j + 1:] * (2 * v) * v.H
        P[:, j + 1:] = P[:, j + 1:] - P[:, j + 1:] * (2 * v) * v.H
    return (H, P)