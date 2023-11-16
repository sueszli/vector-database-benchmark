from .utilities import _iszero

def _columnspace(M, simplify=False):
    if False:
        i = 10
        return i + 15
    'Returns a list of vectors (Matrix objects) that span columnspace of ``M``\n\n    Examples\n    ========\n\n    >>> from sympy import Matrix\n    >>> M = Matrix(3, 3, [1, 3, 0, -2, -6, 0, 3, 9, 6])\n    >>> M\n    Matrix([\n    [ 1,  3, 0],\n    [-2, -6, 0],\n    [ 3,  9, 6]])\n    >>> M.columnspace()\n    [Matrix([\n    [ 1],\n    [-2],\n    [ 3]]), Matrix([\n    [0],\n    [0],\n    [6]])]\n\n    See Also\n    ========\n\n    nullspace\n    rowspace\n    '
    (reduced, pivots) = M.echelon_form(simplify=simplify, with_pivots=True)
    return [M.col(i) for i in pivots]

def _nullspace(M, simplify=False, iszerofunc=_iszero):
    if False:
        print('Hello World!')
    'Returns list of vectors (Matrix objects) that span nullspace of ``M``\n\n    Examples\n    ========\n\n    >>> from sympy import Matrix\n    >>> M = Matrix(3, 3, [1, 3, 0, -2, -6, 0, 3, 9, 6])\n    >>> M\n    Matrix([\n    [ 1,  3, 0],\n    [-2, -6, 0],\n    [ 3,  9, 6]])\n    >>> M.nullspace()\n    [Matrix([\n    [-3],\n    [ 1],\n    [ 0]])]\n\n    See Also\n    ========\n\n    columnspace\n    rowspace\n    '
    (reduced, pivots) = M.rref(iszerofunc=iszerofunc, simplify=simplify)
    free_vars = [i for i in range(M.cols) if i not in pivots]
    basis = []
    for free_var in free_vars:
        vec = [M.zero] * M.cols
        vec[free_var] = M.one
        for (piv_row, piv_col) in enumerate(pivots):
            vec[piv_col] -= reduced[piv_row, free_var]
        basis.append(vec)
    return [M._new(M.cols, 1, b) for b in basis]

def _rowspace(M, simplify=False):
    if False:
        return 10
    'Returns a list of vectors that span the row space of ``M``.\n\n    Examples\n    ========\n\n    >>> from sympy import Matrix\n    >>> M = Matrix(3, 3, [1, 3, 0, -2, -6, 0, 3, 9, 6])\n    >>> M\n    Matrix([\n    [ 1,  3, 0],\n    [-2, -6, 0],\n    [ 3,  9, 6]])\n    >>> M.rowspace()\n    [Matrix([[1, 3, 0]]), Matrix([[0, 0, 6]])]\n    '
    (reduced, pivots) = M.echelon_form(simplify=simplify, with_pivots=True)
    return [reduced.row(i) for i in range(len(pivots))]

def _orthogonalize(cls, *vecs, normalize=False, rankcheck=False):
    if False:
        print('Hello World!')
    'Apply the Gram-Schmidt orthogonalization procedure\n    to vectors supplied in ``vecs``.\n\n    Parameters\n    ==========\n\n    vecs\n        vectors to be made orthogonal\n\n    normalize : bool\n        If ``True``, return an orthonormal basis.\n\n    rankcheck : bool\n        If ``True``, the computation does not stop when encountering\n        linearly dependent vectors.\n\n        If ``False``, it will raise ``ValueError`` when any zero\n        or linearly dependent vectors are found.\n\n    Returns\n    =======\n\n    list\n        List of orthogonal (or orthonormal) basis vectors.\n\n    Examples\n    ========\n\n    >>> from sympy import I, Matrix\n    >>> v = [Matrix([1, I]), Matrix([1, -I])]\n    >>> Matrix.orthogonalize(*v)\n    [Matrix([\n    [1],\n    [I]]), Matrix([\n    [ 1],\n    [-I]])]\n\n    See Also\n    ========\n\n    MatrixBase.QRdecomposition\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process\n    '
    from .decompositions import _QRdecomposition_optional
    if not vecs:
        return []
    all_row_vecs = vecs[0].rows == 1
    vecs = [x.vec() for x in vecs]
    M = cls.hstack(*vecs)
    (Q, R) = _QRdecomposition_optional(M, normalize=normalize)
    if rankcheck and Q.cols < len(vecs):
        raise ValueError('GramSchmidt: vector set not linearly independent')
    ret = []
    for i in range(Q.cols):
        if all_row_vecs:
            col = cls(Q[:, i].T)
        else:
            col = cls(Q[:, i])
        ret.append(col)
    return ret