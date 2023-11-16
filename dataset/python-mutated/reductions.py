from types import FunctionType
from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.domains import ZZ, QQ
from .utilities import _get_intermediate_simp, _iszero, _dotprodsimp, _simplify
from .determinant import _find_reasonable_pivot

def _row_reduce_list(mat, rows, cols, one, iszerofunc, simpfunc, normalize_last=True, normalize=True, zero_above=True):
    if False:
        for i in range(10):
            print('nop')
    'Row reduce a flat list representation of a matrix and return a tuple\n    (rref_matrix, pivot_cols, swaps) where ``rref_matrix`` is a flat list,\n    ``pivot_cols`` are the pivot columns and ``swaps`` are any row swaps that\n    were used in the process of row reduction.\n\n    Parameters\n    ==========\n\n    mat : list\n        list of matrix elements, must be ``rows`` * ``cols`` in length\n\n    rows, cols : integer\n        number of rows and columns in flat list representation\n\n    one : SymPy object\n        represents the value one, from ``Matrix.one``\n\n    iszerofunc : determines if an entry can be used as a pivot\n\n    simpfunc : used to simplify elements and test if they are\n        zero if ``iszerofunc`` returns `None`\n\n    normalize_last : indicates where all row reduction should\n        happen in a fraction-free manner and then the rows are\n        normalized (so that the pivots are 1), or whether\n        rows should be normalized along the way (like the naive\n        row reduction algorithm)\n\n    normalize : whether pivot rows should be normalized so that\n        the pivot value is 1\n\n    zero_above : whether entries above the pivot should be zeroed.\n        If ``zero_above=False``, an echelon matrix will be returned.\n    '

    def get_col(i):
        if False:
            while True:
                i = 10
        return mat[i::cols]

    def row_swap(i, j):
        if False:
            for i in range(10):
                print('nop')
        (mat[i * cols:(i + 1) * cols], mat[j * cols:(j + 1) * cols]) = (mat[j * cols:(j + 1) * cols], mat[i * cols:(i + 1) * cols])

    def cross_cancel(a, i, b, j):
        if False:
            i = 10
            return i + 15
        'Does the row op row[i] = a*row[i] - b*row[j]'
        q = (j - i) * cols
        for p in range(i * cols, (i + 1) * cols):
            mat[p] = isimp(a * mat[p] - b * mat[p + q])
    isimp = _get_intermediate_simp(_dotprodsimp)
    (piv_row, piv_col) = (0, 0)
    pivot_cols = []
    swaps = []
    while piv_col < cols and piv_row < rows:
        (pivot_offset, pivot_val, assumed_nonzero, newly_determined) = _find_reasonable_pivot(get_col(piv_col)[piv_row:], iszerofunc, simpfunc)
        for (offset, val) in newly_determined:
            offset += piv_row
            mat[offset * cols + piv_col] = val
        if pivot_offset is None:
            piv_col += 1
            continue
        pivot_cols.append(piv_col)
        if pivot_offset != 0:
            row_swap(piv_row, pivot_offset + piv_row)
            swaps.append((piv_row, pivot_offset + piv_row))
        if normalize_last is False:
            (i, j) = (piv_row, piv_col)
            mat[i * cols + j] = one
            for p in range(i * cols + j + 1, (i + 1) * cols):
                mat[p] = isimp(mat[p] / pivot_val)
            pivot_val = one
        for row in range(rows):
            if row == piv_row:
                continue
            if zero_above is False and row < piv_row:
                continue
            val = mat[row * cols + piv_col]
            if iszerofunc(val):
                continue
            cross_cancel(pivot_val, row, val, piv_row)
        piv_row += 1
    if normalize_last is True and normalize is True:
        for (piv_i, piv_j) in enumerate(pivot_cols):
            pivot_val = mat[piv_i * cols + piv_j]
            mat[piv_i * cols + piv_j] = one
            for p in range(piv_i * cols + piv_j + 1, (piv_i + 1) * cols):
                mat[p] = isimp(mat[p] / pivot_val)
    return (mat, tuple(pivot_cols), tuple(swaps))

def _row_reduce(M, iszerofunc, simpfunc, normalize_last=True, normalize=True, zero_above=True):
    if False:
        i = 10
        return i + 15
    (mat, pivot_cols, swaps) = _row_reduce_list(list(M), M.rows, M.cols, M.one, iszerofunc, simpfunc, normalize_last=normalize_last, normalize=normalize, zero_above=zero_above)
    return (M._new(M.rows, M.cols, mat), pivot_cols, swaps)

def _is_echelon(M, iszerofunc=_iszero):
    if False:
        for i in range(10):
            print('nop')
    'Returns `True` if the matrix is in echelon form. That is, all rows of\n    zeros are at the bottom, and below each leading non-zero in a row are\n    exclusively zeros.'
    if M.rows <= 0 or M.cols <= 0:
        return True
    zeros_below = all((iszerofunc(t) for t in M[1:, 0]))
    if iszerofunc(M[0, 0]):
        return zeros_below and _is_echelon(M[:, 1:], iszerofunc)
    return zeros_below and _is_echelon(M[1:, 1:], iszerofunc)

def _echelon_form(M, iszerofunc=_iszero, simplify=False, with_pivots=False):
    if False:
        print('Hello World!')
    'Returns a matrix row-equivalent to ``M`` that is in echelon form. Note\n    that echelon form of a matrix is *not* unique, however, properties like the\n    row space and the null space are preserved.\n\n    Examples\n    ========\n\n    >>> from sympy import Matrix\n    >>> M = Matrix([[1, 2], [3, 4]])\n    >>> M.echelon_form()\n    Matrix([\n    [1,  2],\n    [0, -2]])\n    '
    simpfunc = simplify if isinstance(simplify, FunctionType) else _simplify
    (mat, pivots, _) = _row_reduce(M, iszerofunc, simpfunc, normalize_last=True, normalize=False, zero_above=False)
    if with_pivots:
        return (mat, pivots)
    return mat

def _rank(M, iszerofunc=_iszero, simplify=False):
    if False:
        while True:
            i = 10
    'Returns the rank of a matrix.\n\n    Examples\n    ========\n\n    >>> from sympy import Matrix\n    >>> from sympy.abc import x\n    >>> m = Matrix([[1, 2], [x, 1 - 1/x]])\n    >>> m.rank()\n    2\n    >>> n = Matrix(3, 3, range(1, 10))\n    >>> n.rank()\n    2\n    '

    def _permute_complexity_right(M, iszerofunc):
        if False:
            while True:
                i = 10
        'Permute columns with complicated elements as\n        far right as they can go.  Since the ``sympy`` row reduction\n        algorithms start on the left, having complexity right-shifted\n        speeds things up.\n\n        Returns a tuple (mat, perm) where perm is a permutation\n        of the columns to perform to shift the complex columns right, and mat\n        is the permuted matrix.'

        def complexity(i):
            if False:
                i = 10
                return i + 15
            return sum((1 if iszerofunc(e) is None else 0 for e in M[:, i]))
        complex = [(complexity(i), i) for i in range(M.cols)]
        perm = [j for (i, j) in sorted(complex)]
        return (M.permute(perm, orientation='cols'), perm)
    simpfunc = simplify if isinstance(simplify, FunctionType) else _simplify
    if M.rows <= 0 or M.cols <= 0:
        return 0
    if M.rows <= 1 or M.cols <= 1:
        zeros = [iszerofunc(x) for x in M]
        if False in zeros:
            return 1
    if M.rows == 2 and M.cols == 2:
        zeros = [iszerofunc(x) for x in M]
        if False not in zeros and None not in zeros:
            return 0
        d = M.det()
        if iszerofunc(d) and False in zeros:
            return 1
        if iszerofunc(d) is False:
            return 2
    (mat, _) = _permute_complexity_right(M, iszerofunc=iszerofunc)
    (_, pivots, _) = _row_reduce(mat, iszerofunc, simpfunc, normalize_last=True, normalize=False, zero_above=False)
    return len(pivots)

def _to_DM_ZZ_QQ(M):
    if False:
        return 10
    if not hasattr(M, '_rep'):
        return None
    rep = M._rep
    K = rep.domain
    if K.is_ZZ:
        return rep
    elif K.is_QQ:
        try:
            return rep.convert_to(ZZ)
        except CoercionFailed:
            return rep
    else:
        if not all((e.is_Rational for e in M)):
            return None
        try:
            return rep.convert_to(ZZ)
        except CoercionFailed:
            return rep.convert_to(QQ)

def _rref_dm(dM):
    if False:
        for i in range(10):
            print('nop')
    'Compute the reduced row echelon form of a DomainMatrix.'
    K = dM.domain
    if K.is_ZZ:
        (dM_rref, den, pivots) = dM.rref_den(keep_domain=False)
        dM_rref = dM_rref.to_field() / den
    elif K.is_QQ:
        (dM_rref, pivots) = dM.rref()
    else:
        assert False
    M_rref = dM_rref.to_Matrix()
    return (M_rref, pivots)

def _rref(M, iszerofunc=_iszero, simplify=False, pivots=True, normalize_last=True):
    if False:
        for i in range(10):
            print('nop')
    "Return reduced row-echelon form of matrix and indices\n    of pivot vars.\n\n    Parameters\n    ==========\n\n    iszerofunc : Function\n        A function used for detecting whether an element can\n        act as a pivot.  ``lambda x: x.is_zero`` is used by default.\n\n    simplify : Function\n        A function used to simplify elements when looking for a pivot.\n        By default SymPy's ``simplify`` is used.\n\n    pivots : True or False\n        If ``True``, a tuple containing the row-reduced matrix and a tuple\n        of pivot columns is returned.  If ``False`` just the row-reduced\n        matrix is returned.\n\n    normalize_last : True or False\n        If ``True``, no pivots are normalized to `1` until after all\n        entries above and below each pivot are zeroed.  This means the row\n        reduction algorithm is fraction free until the very last step.\n        If ``False``, the naive row reduction procedure is used where\n        each pivot is normalized to be `1` before row operations are\n        used to zero above and below the pivot.\n\n    Examples\n    ========\n\n    >>> from sympy import Matrix\n    >>> from sympy.abc import x\n    >>> m = Matrix([[1, 2], [x, 1 - 1/x]])\n    >>> m.rref()\n    (Matrix([\n    [1, 0],\n    [0, 1]]), (0, 1))\n    >>> rref_matrix, rref_pivots = m.rref()\n    >>> rref_matrix\n    Matrix([\n    [1, 0],\n    [0, 1]])\n    >>> rref_pivots\n    (0, 1)\n\n    ``iszerofunc`` can correct rounding errors in matrices with float\n    values. In the following example, calling ``rref()`` leads to\n    floating point errors, incorrectly row reducing the matrix.\n    ``iszerofunc= lambda x: abs(x) < 1e-9`` sets sufficiently small numbers\n    to zero, avoiding this error.\n\n    >>> m = Matrix([[0.9, -0.1, -0.2, 0], [-0.8, 0.9, -0.4, 0], [-0.1, -0.8, 0.6, 0]])\n    >>> m.rref()\n    (Matrix([\n    [1, 0, 0, 0],\n    [0, 1, 0, 0],\n    [0, 0, 1, 0]]), (0, 1, 2))\n    >>> m.rref(iszerofunc=lambda x:abs(x)<1e-9)\n    (Matrix([\n    [1, 0, -0.301369863013699, 0],\n    [0, 1, -0.712328767123288, 0],\n    [0, 0,         0,          0]]), (0, 1))\n\n    Notes\n    =====\n\n    The default value of ``normalize_last=True`` can provide significant\n    speedup to row reduction, especially on matrices with symbols.  However,\n    if you depend on the form row reduction algorithm leaves entries\n    of the matrix, set ``normalize_last=False``\n    "
    dM = _to_DM_ZZ_QQ(M)
    if dM is not None:
        (mat, pivot_cols) = _rref_dm(dM)
    else:
        if isinstance(simplify, FunctionType):
            simpfunc = simplify
        else:
            simpfunc = _simplify
        (mat, pivot_cols, _) = _row_reduce(M, iszerofunc, simpfunc, normalize_last, normalize=True, zero_above=True)
    if pivots:
        return (mat, pivot_cols)
    else:
        return mat