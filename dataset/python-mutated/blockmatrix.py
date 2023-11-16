from sympy.assumptions.ask import Q, ask
from sympy.core import Basic, Add, Mul, S
from sympy.core.sympify import _sympify
from sympy.functions import adjoint
from sympy.functions.elementary.complexes import re, im
from sympy.strategies import typed, exhaust, condition, do_one, unpack
from sympy.strategies.traverse import bottom_up
from sympy.utilities.iterables import is_sequence, sift
from sympy.utilities.misc import filldedent
from sympy.matrices import Matrix, ShapeError
from sympy.matrices.common import NonInvertibleMatrixError
from sympy.matrices.expressions.determinant import det, Determinant
from sympy.matrices.expressions.inverse import Inverse
from sympy.matrices.expressions.matadd import MatAdd
from sympy.matrices.expressions.matexpr import MatrixExpr, MatrixElement
from sympy.matrices.expressions.matmul import MatMul
from sympy.matrices.expressions.matpow import MatPow
from sympy.matrices.expressions.slice import MatrixSlice
from sympy.matrices.expressions.special import ZeroMatrix, Identity
from sympy.matrices.expressions.trace import trace
from sympy.matrices.expressions.transpose import Transpose, transpose

class BlockMatrix(MatrixExpr):
    """A BlockMatrix is a Matrix comprised of other matrices.

    The submatrices are stored in a SymPy Matrix object but accessed as part of
    a Matrix Expression

    >>> from sympy import (MatrixSymbol, BlockMatrix, symbols,
    ...     Identity, ZeroMatrix, block_collapse)
    >>> n,m,l = symbols('n m l')
    >>> X = MatrixSymbol('X', n, n)
    >>> Y = MatrixSymbol('Y', m, m)
    >>> Z = MatrixSymbol('Z', n, m)
    >>> B = BlockMatrix([[X, Z], [ZeroMatrix(m,n), Y]])
    >>> print(B)
    Matrix([
    [X, Z],
    [0, Y]])

    >>> C = BlockMatrix([[Identity(n), Z]])
    >>> print(C)
    Matrix([[I, Z]])

    >>> print(block_collapse(C*B))
    Matrix([[X, Z + Z*Y]])

    Some matrices might be comprised of rows of blocks with
    the matrices in each row having the same height and the
    rows all having the same total number of columns but
    not having the same number of columns for each matrix
    in each row. In this case, the matrix is not a block
    matrix and should be instantiated by Matrix.

    >>> from sympy import ones, Matrix
    >>> dat = [
    ... [ones(3,2), ones(3,3)*2],
    ... [ones(2,3)*3, ones(2,2)*4]]
    ...
    >>> BlockMatrix(dat)
    Traceback (most recent call last):
    ...
    ValueError:
    Although this matrix is comprised of blocks, the blocks do not fill
    the matrix in a size-symmetric fashion. To create a full matrix from
    these arguments, pass them directly to Matrix.
    >>> Matrix(dat)
    Matrix([
    [1, 1, 2, 2, 2],
    [1, 1, 2, 2, 2],
    [1, 1, 2, 2, 2],
    [3, 3, 3, 4, 4],
    [3, 3, 3, 4, 4]])

    See Also
    ========
    sympy.matrices.matrices.MatrixBase.irregular
    """

    def __new__(cls, *args, **kwargs):
        if False:
            return 10
        from sympy.matrices.immutable import ImmutableDenseMatrix
        isMat = lambda i: getattr(i, 'is_Matrix', False)
        if len(args) != 1 or not is_sequence(args[0]) or len({isMat(r) for r in args[0]}) != 1:
            raise ValueError(filldedent('\n                expecting a sequence of 1 or more rows\n                containing Matrices.'))
        rows = args[0] if args else []
        if not isMat(rows):
            if rows and isMat(rows[0]):
                rows = [rows]
            blocky = ok = len({len(r) for r in rows}) == 1
            if ok:
                for r in rows:
                    ok = len({i.rows for i in r}) == 1
                    if not ok:
                        break
                blocky = ok
                if ok:
                    for c in range(len(rows[0])):
                        ok = len({rows[i][c].cols for i in range(len(rows))}) == 1
                        if not ok:
                            break
            if not ok:
                ok = len({sum([i.cols for i in r]) for r in rows}) == 1
                if blocky and ok:
                    raise ValueError(filldedent('\n                        Although this matrix is comprised of blocks,\n                        the blocks do not fill the matrix in a\n                        size-symmetric fashion. To create a full matrix\n                        from these arguments, pass them directly to\n                        Matrix.'))
                raise ValueError(filldedent("\n                    When there are not the same number of rows in each\n                    row's matrices or there are not the same number of\n                    total columns in each row, the matrix is not a\n                    block matrix. If this matrix is known to consist of\n                    blocks fully filling a 2-D space then see\n                    Matrix.irregular."))
        mat = ImmutableDenseMatrix(rows, evaluate=False)
        obj = Basic.__new__(cls, mat)
        return obj

    @property
    def shape(self):
        if False:
            i = 10
            return i + 15
        numrows = numcols = 0
        M = self.blocks
        for i in range(M.shape[0]):
            numrows += M[i, 0].shape[0]
        for i in range(M.shape[1]):
            numcols += M[0, i].shape[1]
        return (numrows, numcols)

    @property
    def blockshape(self):
        if False:
            return 10
        return self.blocks.shape

    @property
    def blocks(self):
        if False:
            while True:
                i = 10
        return self.args[0]

    @property
    def rowblocksizes(self):
        if False:
            print('Hello World!')
        return [self.blocks[i, 0].rows for i in range(self.blockshape[0])]

    @property
    def colblocksizes(self):
        if False:
            print('Hello World!')
        return [self.blocks[0, i].cols for i in range(self.blockshape[1])]

    def structurally_equal(self, other):
        if False:
            return 10
        return isinstance(other, BlockMatrix) and self.shape == other.shape and (self.blockshape == other.blockshape) and (self.rowblocksizes == other.rowblocksizes) and (self.colblocksizes == other.colblocksizes)

    def _blockmul(self, other):
        if False:
            print('Hello World!')
        if isinstance(other, BlockMatrix) and self.colblocksizes == other.rowblocksizes:
            return BlockMatrix(self.blocks * other.blocks)
        return self * other

    def _blockadd(self, other):
        if False:
            print('Hello World!')
        if isinstance(other, BlockMatrix) and self.structurally_equal(other):
            return BlockMatrix(self.blocks + other.blocks)
        return self + other

    def _eval_transpose(self):
        if False:
            i = 10
            return i + 15
        matrices = [transpose(matrix) for matrix in self.blocks]
        M = Matrix(self.blockshape[0], self.blockshape[1], matrices)
        M = M.transpose()
        return BlockMatrix(M)

    def _eval_adjoint(self):
        if False:
            i = 10
            return i + 15
        matrices = [adjoint(matrix) for matrix in self.blocks]
        M = Matrix(self.blockshape[0], self.blockshape[1], matrices)
        M = M.transpose()
        return BlockMatrix(M)

    def _eval_trace(self):
        if False:
            i = 10
            return i + 15
        if self.rowblocksizes == self.colblocksizes:
            blocks = [self.blocks[i, i] for i in range(self.blockshape[0])]
            return Add(*[trace(block) for block in blocks])

    def _eval_determinant(self):
        if False:
            while True:
                i = 10
        if self.blockshape == (1, 1):
            return det(self.blocks[0, 0])
        if self.blockshape == (2, 2):
            [[A, B], [C, D]] = self.blocks.tolist()
            if ask(Q.invertible(A)):
                return det(A) * det(D - C * A.I * B)
            elif ask(Q.invertible(D)):
                return det(D) * det(A - B * D.I * C)
        return Determinant(self)

    def _eval_as_real_imag(self):
        if False:
            print('Hello World!')
        real_matrices = [re(matrix) for matrix in self.blocks]
        real_matrices = Matrix(self.blockshape[0], self.blockshape[1], real_matrices)
        im_matrices = [im(matrix) for matrix in self.blocks]
        im_matrices = Matrix(self.blockshape[0], self.blockshape[1], im_matrices)
        return (BlockMatrix(real_matrices), BlockMatrix(im_matrices))

    def _eval_derivative(self, x):
        if False:
            while True:
                i = 10
        return BlockMatrix(self.blocks.diff(x))

    def transpose(self):
        if False:
            return 10
        "Return transpose of matrix.\n\n        Examples\n        ========\n\n        >>> from sympy import MatrixSymbol, BlockMatrix, ZeroMatrix\n        >>> from sympy.abc import m, n\n        >>> X = MatrixSymbol('X', n, n)\n        >>> Y = MatrixSymbol('Y', m, m)\n        >>> Z = MatrixSymbol('Z', n, m)\n        >>> B = BlockMatrix([[X, Z], [ZeroMatrix(m,n), Y]])\n        >>> B.transpose()\n        Matrix([\n        [X.T,  0],\n        [Z.T, Y.T]])\n        >>> _.transpose()\n        Matrix([\n        [X, Z],\n        [0, Y]])\n        "
        return self._eval_transpose()

    def schur(self, mat='A', generalized=False):
        if False:
            while True:
                i = 10
        'Return the Schur Complement of the 2x2 BlockMatrix\n\n        Parameters\n        ==========\n\n        mat : String, optional\n            The matrix with respect to which the\n            Schur Complement is calculated. \'A\' is\n            used by default\n\n        generalized : bool, optional\n            If True, returns the generalized Schur\n            Component which uses Moore-Penrose Inverse\n\n        Examples\n        ========\n\n        >>> from sympy import symbols, MatrixSymbol, BlockMatrix\n        >>> m, n = symbols(\'m n\')\n        >>> A = MatrixSymbol(\'A\', n, n)\n        >>> B = MatrixSymbol(\'B\', n, m)\n        >>> C = MatrixSymbol(\'C\', m, n)\n        >>> D = MatrixSymbol(\'D\', m, m)\n        >>> X = BlockMatrix([[A, B], [C, D]])\n\n        The default Schur Complement is evaluated with "A"\n\n        >>> X.schur()\n        -C*A**(-1)*B + D\n        >>> X.schur(\'D\')\n        A - B*D**(-1)*C\n\n        Schur complement with non-invertible matrices is not\n        defined. Instead, the generalized Schur complement can\n        be calculated which uses the Moore-Penrose Inverse. To\n        achieve this, `generalized` must be set to `True`\n\n        >>> X.schur(\'B\', generalized=True)\n        C - D*(B.T*B)**(-1)*B.T*A\n        >>> X.schur(\'C\', generalized=True)\n        -A*(C.T*C)**(-1)*C.T*D + B\n\n        Returns\n        =======\n\n        M : Matrix\n            The Schur Complement Matrix\n\n        Raises\n        ======\n\n        ShapeError\n            If the block matrix is not a 2x2 matrix\n\n        NonInvertibleMatrixError\n            If given matrix is non-invertible\n\n        References\n        ==========\n\n        .. [1] Wikipedia Article on Schur Component : https://en.wikipedia.org/wiki/Schur_complement\n\n        See Also\n        ========\n\n        sympy.matrices.matrices.MatrixBase.pinv\n        '
        if self.blockshape == (2, 2):
            [[A, B], [C, D]] = self.blocks.tolist()
            d = {'A': A, 'B': B, 'C': C, 'D': D}
            try:
                inv = (d[mat].T * d[mat]).inv() * d[mat].T if generalized else d[mat].inv()
                if mat == 'A':
                    return D - C * inv * B
                elif mat == 'B':
                    return C - D * inv * A
                elif mat == 'C':
                    return B - A * inv * D
                elif mat == 'D':
                    return A - B * inv * C
                return self
            except NonInvertibleMatrixError:
                raise NonInvertibleMatrixError('The given matrix is not invertible. Please set generalized=True             to compute the generalized Schur Complement which uses Moore-Penrose Inverse')
        else:
            raise ShapeError('Schur Complement can only be calculated for 2x2 block matrices')

    def LDUdecomposition(self):
        if False:
            print('Hello World!')
        'Returns the Block LDU decomposition of\n        a 2x2 Block Matrix\n\n        Returns\n        =======\n\n        (L, D, U) : Matrices\n            L : Lower Diagonal Matrix\n            D : Diagonal Matrix\n            U : Upper Diagonal Matrix\n\n        Examples\n        ========\n\n        >>> from sympy import symbols, MatrixSymbol, BlockMatrix, block_collapse\n        >>> m, n = symbols(\'m n\')\n        >>> A = MatrixSymbol(\'A\', n, n)\n        >>> B = MatrixSymbol(\'B\', n, m)\n        >>> C = MatrixSymbol(\'C\', m, n)\n        >>> D = MatrixSymbol(\'D\', m, m)\n        >>> X = BlockMatrix([[A, B], [C, D]])\n        >>> L, D, U = X.LDUdecomposition()\n        >>> block_collapse(L*D*U)\n        Matrix([\n        [A, B],\n        [C, D]])\n\n        Raises\n        ======\n\n        ShapeError\n            If the block matrix is not a 2x2 matrix\n\n        NonInvertibleMatrixError\n            If the matrix "A" is non-invertible\n\n        See Also\n        ========\n        sympy.matrices.expressions.blockmatrix.BlockMatrix.UDLdecomposition\n        sympy.matrices.expressions.blockmatrix.BlockMatrix.LUdecomposition\n        '
        if self.blockshape == (2, 2):
            [[A, B], [C, D]] = self.blocks.tolist()
            try:
                AI = A.I
            except NonInvertibleMatrixError:
                raise NonInvertibleMatrixError('Block LDU decomposition cannot be calculated when                    "A" is singular')
            Ip = Identity(B.shape[0])
            Iq = Identity(B.shape[1])
            Z = ZeroMatrix(*B.shape)
            L = BlockMatrix([[Ip, Z], [C * AI, Iq]])
            D = BlockDiagMatrix(A, self.schur())
            U = BlockMatrix([[Ip, AI * B], [Z.T, Iq]])
            return (L, D, U)
        else:
            raise ShapeError('Block LDU decomposition is supported only for 2x2 block matrices')

    def UDLdecomposition(self):
        if False:
            print('Hello World!')
        'Returns the Block UDL decomposition of\n        a 2x2 Block Matrix\n\n        Returns\n        =======\n\n        (U, D, L) : Matrices\n            U : Upper Diagonal Matrix\n            D : Diagonal Matrix\n            L : Lower Diagonal Matrix\n\n        Examples\n        ========\n\n        >>> from sympy import symbols, MatrixSymbol, BlockMatrix, block_collapse\n        >>> m, n = symbols(\'m n\')\n        >>> A = MatrixSymbol(\'A\', n, n)\n        >>> B = MatrixSymbol(\'B\', n, m)\n        >>> C = MatrixSymbol(\'C\', m, n)\n        >>> D = MatrixSymbol(\'D\', m, m)\n        >>> X = BlockMatrix([[A, B], [C, D]])\n        >>> U, D, L = X.UDLdecomposition()\n        >>> block_collapse(U*D*L)\n        Matrix([\n        [A, B],\n        [C, D]])\n\n        Raises\n        ======\n\n        ShapeError\n            If the block matrix is not a 2x2 matrix\n\n        NonInvertibleMatrixError\n            If the matrix "D" is non-invertible\n\n        See Also\n        ========\n        sympy.matrices.expressions.blockmatrix.BlockMatrix.LDUdecomposition\n        sympy.matrices.expressions.blockmatrix.BlockMatrix.LUdecomposition\n        '
        if self.blockshape == (2, 2):
            [[A, B], [C, D]] = self.blocks.tolist()
            try:
                DI = D.I
            except NonInvertibleMatrixError:
                raise NonInvertibleMatrixError('Block UDL decomposition cannot be calculated when                    "D" is singular')
            Ip = Identity(A.shape[0])
            Iq = Identity(B.shape[1])
            Z = ZeroMatrix(*B.shape)
            U = BlockMatrix([[Ip, B * DI], [Z.T, Iq]])
            D = BlockDiagMatrix(self.schur('D'), D)
            L = BlockMatrix([[Ip, Z], [DI * C, Iq]])
            return (U, D, L)
        else:
            raise ShapeError('Block UDL decomposition is supported only for 2x2 block matrices')

    def LUdecomposition(self):
        if False:
            print('Hello World!')
        'Returns the Block LU decomposition of\n        a 2x2 Block Matrix\n\n        Returns\n        =======\n\n        (L, U) : Matrices\n            L : Lower Diagonal Matrix\n            U : Upper Diagonal Matrix\n\n        Examples\n        ========\n\n        >>> from sympy import symbols, MatrixSymbol, BlockMatrix, block_collapse\n        >>> m, n = symbols(\'m n\')\n        >>> A = MatrixSymbol(\'A\', n, n)\n        >>> B = MatrixSymbol(\'B\', n, m)\n        >>> C = MatrixSymbol(\'C\', m, n)\n        >>> D = MatrixSymbol(\'D\', m, m)\n        >>> X = BlockMatrix([[A, B], [C, D]])\n        >>> L, U = X.LUdecomposition()\n        >>> block_collapse(L*U)\n        Matrix([\n        [A, B],\n        [C, D]])\n\n        Raises\n        ======\n\n        ShapeError\n            If the block matrix is not a 2x2 matrix\n\n        NonInvertibleMatrixError\n            If the matrix "A" is non-invertible\n\n        See Also\n        ========\n        sympy.matrices.expressions.blockmatrix.BlockMatrix.UDLdecomposition\n        sympy.matrices.expressions.blockmatrix.BlockMatrix.LDUdecomposition\n        '
        if self.blockshape == (2, 2):
            [[A, B], [C, D]] = self.blocks.tolist()
            try:
                A = A ** S.Half
                AI = A.I
            except NonInvertibleMatrixError:
                raise NonInvertibleMatrixError('Block LU decomposition cannot be calculated when                    "A" is singular')
            Z = ZeroMatrix(*B.shape)
            Q = self.schur() ** S.Half
            L = BlockMatrix([[A, Z], [C * AI, Q]])
            U = BlockMatrix([[A, AI * B], [Z.T, Q]])
            return (L, U)
        else:
            raise ShapeError('Block LU decomposition is supported only for 2x2 block matrices')

    def _entry(self, i, j, **kwargs):
        if False:
            return 10
        (orig_i, orig_j) = (i, j)
        for (row_block, numrows) in enumerate(self.rowblocksizes):
            cmp = i < numrows
            if cmp == True:
                break
            elif cmp == False:
                i -= numrows
            elif row_block < self.blockshape[0] - 1:
                return MatrixElement(self, orig_i, orig_j)
        for (col_block, numcols) in enumerate(self.colblocksizes):
            cmp = j < numcols
            if cmp == True:
                break
            elif cmp == False:
                j -= numcols
            elif col_block < self.blockshape[1] - 1:
                return MatrixElement(self, orig_i, orig_j)
        return self.blocks[row_block, col_block][i, j]

    @property
    def is_Identity(self):
        if False:
            print('Hello World!')
        if self.blockshape[0] != self.blockshape[1]:
            return False
        for i in range(self.blockshape[0]):
            for j in range(self.blockshape[1]):
                if i == j and (not self.blocks[i, j].is_Identity):
                    return False
                if i != j and (not self.blocks[i, j].is_ZeroMatrix):
                    return False
        return True

    @property
    def is_structurally_symmetric(self):
        if False:
            for i in range(10):
                print('nop')
        return self.rowblocksizes == self.colblocksizes

    def equals(self, other):
        if False:
            print('Hello World!')
        if self == other:
            return True
        if isinstance(other, BlockMatrix) and self.blocks == other.blocks:
            return True
        return super().equals(other)

class BlockDiagMatrix(BlockMatrix):
    """A sparse matrix with block matrices along its diagonals

    Examples
    ========

    >>> from sympy import MatrixSymbol, BlockDiagMatrix, symbols
    >>> n, m, l = symbols('n m l')
    >>> X = MatrixSymbol('X', n, n)
    >>> Y = MatrixSymbol('Y', m, m)
    >>> BlockDiagMatrix(X, Y)
    Matrix([
    [X, 0],
    [0, Y]])

    Notes
    =====

    If you want to get the individual diagonal blocks, use
    :meth:`get_diag_blocks`.

    See Also
    ========

    sympy.matrices.dense.diag
    """

    def __new__(cls, *mats):
        if False:
            while True:
                i = 10
        return Basic.__new__(BlockDiagMatrix, *[_sympify(m) for m in mats])

    @property
    def diag(self):
        if False:
            print('Hello World!')
        return self.args

    @property
    def blocks(self):
        if False:
            i = 10
            return i + 15
        from sympy.matrices.immutable import ImmutableDenseMatrix
        mats = self.args
        data = [[mats[i] if i == j else ZeroMatrix(mats[i].rows, mats[j].cols) for j in range(len(mats))] for i in range(len(mats))]
        return ImmutableDenseMatrix(data, evaluate=False)

    @property
    def shape(self):
        if False:
            for i in range(10):
                print('nop')
        return (sum((block.rows for block in self.args)), sum((block.cols for block in self.args)))

    @property
    def blockshape(self):
        if False:
            i = 10
            return i + 15
        n = len(self.args)
        return (n, n)

    @property
    def rowblocksizes(self):
        if False:
            print('Hello World!')
        return [block.rows for block in self.args]

    @property
    def colblocksizes(self):
        if False:
            i = 10
            return i + 15
        return [block.cols for block in self.args]

    def _all_square_blocks(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns true if all blocks are square'
        return all((mat.is_square for mat in self.args))

    def _eval_determinant(self):
        if False:
            while True:
                i = 10
        if self._all_square_blocks():
            return Mul(*[det(mat) for mat in self.args])
        return S.Zero

    def _eval_inverse(self, expand='ignored'):
        if False:
            print('Hello World!')
        if self._all_square_blocks():
            return BlockDiagMatrix(*[mat.inverse() for mat in self.args])
        raise NonInvertibleMatrixError('Matrix det == 0; not invertible.')

    def _eval_transpose(self):
        if False:
            while True:
                i = 10
        return BlockDiagMatrix(*[mat.transpose() for mat in self.args])

    def _blockmul(self, other):
        if False:
            i = 10
            return i + 15
        if isinstance(other, BlockDiagMatrix) and self.colblocksizes == other.rowblocksizes:
            return BlockDiagMatrix(*[a * b for (a, b) in zip(self.args, other.args)])
        else:
            return BlockMatrix._blockmul(self, other)

    def _blockadd(self, other):
        if False:
            print('Hello World!')
        if isinstance(other, BlockDiagMatrix) and self.blockshape == other.blockshape and (self.rowblocksizes == other.rowblocksizes) and (self.colblocksizes == other.colblocksizes):
            return BlockDiagMatrix(*[a + b for (a, b) in zip(self.args, other.args)])
        else:
            return BlockMatrix._blockadd(self, other)

    def get_diag_blocks(self):
        if False:
            while True:
                i = 10
        'Return the list of diagonal blocks of the matrix.\n\n        Examples\n        ========\n\n        >>> from sympy import BlockDiagMatrix, Matrix\n\n        >>> A = Matrix([[1, 2], [3, 4]])\n        >>> B = Matrix([[5, 6], [7, 8]])\n        >>> M = BlockDiagMatrix(A, B)\n\n        How to get diagonal blocks from the block diagonal matrix:\n\n        >>> diag_blocks = M.get_diag_blocks()\n        >>> diag_blocks[0]\n        Matrix([\n        [1, 2],\n        [3, 4]])\n        >>> diag_blocks[1]\n        Matrix([\n        [5, 6],\n        [7, 8]])\n        '
        return self.args

def block_collapse(expr):
    if False:
        for i in range(10):
            print('nop')
    "Evaluates a block matrix expression\n\n    >>> from sympy import MatrixSymbol, BlockMatrix, symbols, Identity, ZeroMatrix, block_collapse\n    >>> n,m,l = symbols('n m l')\n    >>> X = MatrixSymbol('X', n, n)\n    >>> Y = MatrixSymbol('Y', m, m)\n    >>> Z = MatrixSymbol('Z', n, m)\n    >>> B = BlockMatrix([[X, Z], [ZeroMatrix(m, n), Y]])\n    >>> print(B)\n    Matrix([\n    [X, Z],\n    [0, Y]])\n\n    >>> C = BlockMatrix([[Identity(n), Z]])\n    >>> print(C)\n    Matrix([[I, Z]])\n\n    >>> print(block_collapse(C*B))\n    Matrix([[X, Z + Z*Y]])\n    "
    from sympy.strategies.util import expr_fns
    hasbm = lambda expr: isinstance(expr, MatrixExpr) and expr.has(BlockMatrix)
    conditioned_rl = condition(hasbm, typed({MatAdd: do_one(bc_matadd, bc_block_plus_ident), MatMul: do_one(bc_matmul, bc_dist), MatPow: bc_matmul, Transpose: bc_transpose, Inverse: bc_inverse, BlockMatrix: do_one(bc_unpack, deblock)}))
    rule = exhaust(bottom_up(exhaust(conditioned_rl), fns=expr_fns))
    result = rule(expr)
    doit = getattr(result, 'doit', None)
    if doit is not None:
        return doit()
    else:
        return result

def bc_unpack(expr):
    if False:
        while True:
            i = 10
    if expr.blockshape == (1, 1):
        return expr.blocks[0, 0]
    return expr

def bc_matadd(expr):
    if False:
        while True:
            i = 10
    args = sift(expr.args, lambda M: isinstance(M, BlockMatrix))
    blocks = args[True]
    if not blocks:
        return expr
    nonblocks = args[False]
    block = blocks[0]
    for b in blocks[1:]:
        block = block._blockadd(b)
    if nonblocks:
        return MatAdd(*nonblocks) + block
    else:
        return block

def bc_block_plus_ident(expr):
    if False:
        i = 10
        return i + 15
    idents = [arg for arg in expr.args if arg.is_Identity]
    if not idents:
        return expr
    blocks = [arg for arg in expr.args if isinstance(arg, BlockMatrix)]
    if blocks and all((b.structurally_equal(blocks[0]) for b in blocks)) and blocks[0].is_structurally_symmetric:
        block_id = BlockDiagMatrix(*[Identity(k) for k in blocks[0].rowblocksizes])
        rest = [arg for arg in expr.args if not arg.is_Identity and (not isinstance(arg, BlockMatrix))]
        return MatAdd(block_id * len(idents), *blocks, *rest).doit()
    return expr

def bc_dist(expr):
    if False:
        return 10
    ' Turn  a*[X, Y] into [a*X, a*Y] '
    (factor, mat) = expr.as_coeff_mmul()
    if factor == 1:
        return expr
    unpacked = unpack(mat)
    if isinstance(unpacked, BlockDiagMatrix):
        B = unpacked.diag
        new_B = [factor * mat for mat in B]
        return BlockDiagMatrix(*new_B)
    elif isinstance(unpacked, BlockMatrix):
        B = unpacked.blocks
        new_B = [[factor * B[i, j] for j in range(B.cols)] for i in range(B.rows)]
        return BlockMatrix(new_B)
    return expr

def bc_matmul(expr):
    if False:
        while True:
            i = 10
    if isinstance(expr, MatPow):
        if expr.args[1].is_Integer and expr.args[1] > 0:
            (factor, matrices) = (1, [expr.args[0]] * expr.args[1])
        else:
            return expr
    else:
        (factor, matrices) = expr.as_coeff_matrices()
    i = 0
    while i + 1 < len(matrices):
        (A, B) = matrices[i:i + 2]
        if isinstance(A, BlockMatrix) and isinstance(B, BlockMatrix):
            matrices[i] = A._blockmul(B)
            matrices.pop(i + 1)
        elif isinstance(A, BlockMatrix):
            matrices[i] = A._blockmul(BlockMatrix([[B]]))
            matrices.pop(i + 1)
        elif isinstance(B, BlockMatrix):
            matrices[i] = BlockMatrix([[A]])._blockmul(B)
            matrices.pop(i + 1)
        else:
            i += 1
    return MatMul(factor, *matrices).doit()

def bc_transpose(expr):
    if False:
        i = 10
        return i + 15
    collapse = block_collapse(expr.arg)
    return collapse._eval_transpose()

def bc_inverse(expr):
    if False:
        i = 10
        return i + 15
    if isinstance(expr.arg, BlockDiagMatrix):
        return expr.inverse()
    expr2 = blockinverse_1x1(expr)
    if expr != expr2:
        return expr2
    return blockinverse_2x2(Inverse(reblock_2x2(expr.arg)))

def blockinverse_1x1(expr):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(expr.arg, BlockMatrix) and expr.arg.blockshape == (1, 1):
        mat = Matrix([[expr.arg.blocks[0].inverse()]])
        return BlockMatrix(mat)
    return expr

def blockinverse_2x2(expr):
    if False:
        print('Hello World!')
    if isinstance(expr.arg, BlockMatrix) and expr.arg.blockshape == (2, 2):
        [[A, B], [C, D]] = expr.arg.blocks.tolist()
        formula = _choose_2x2_inversion_formula(A, B, C, D)
        if formula != None:
            MI = expr.arg.schur(formula).I
        if formula == 'A':
            AI = A.I
            return BlockMatrix([[AI + AI * B * MI * C * AI, -AI * B * MI], [-MI * C * AI, MI]])
        if formula == 'B':
            BI = B.I
            return BlockMatrix([[-MI * D * BI, MI], [BI + BI * A * MI * D * BI, -BI * A * MI]])
        if formula == 'C':
            CI = C.I
            return BlockMatrix([[-CI * D * MI, CI + CI * D * MI * A * CI], [MI, -MI * A * CI]])
        if formula == 'D':
            DI = D.I
            return BlockMatrix([[MI, -MI * B * DI], [-DI * C * MI, DI + DI * C * MI * B * DI]])
    return expr

def _choose_2x2_inversion_formula(A, B, C, D):
    if False:
        i = 10
        return i + 15
    "\n    Assuming [[A, B], [C, D]] would form a valid square block matrix, find\n    which of the classical 2x2 block matrix inversion formulas would be\n    best suited.\n\n    Returns 'A', 'B', 'C', 'D' to represent the algorithm involving inversion\n    of the given argument or None if the matrix cannot be inverted using\n    any of those formulas.\n    "
    A_inv = ask(Q.invertible(A))
    if A_inv == True:
        return 'A'
    B_inv = ask(Q.invertible(B))
    if B_inv == True:
        return 'B'
    C_inv = ask(Q.invertible(C))
    if C_inv == True:
        return 'C'
    D_inv = ask(Q.invertible(D))
    if D_inv == True:
        return 'D'
    if A_inv != False:
        return 'A'
    if B_inv != False:
        return 'B'
    if C_inv != False:
        return 'C'
    if D_inv != False:
        return 'D'
    return None

def deblock(B):
    if False:
        i = 10
        return i + 15
    ' Flatten a BlockMatrix of BlockMatrices '
    if not isinstance(B, BlockMatrix) or not B.blocks.has(BlockMatrix):
        return B
    wrap = lambda x: x if isinstance(x, BlockMatrix) else BlockMatrix([[x]])
    bb = B.blocks.applyfunc(wrap)
    try:
        MM = Matrix(0, sum((bb[0, i].blocks.shape[1] for i in range(bb.shape[1]))), [])
        for row in range(0, bb.shape[0]):
            M = Matrix(bb[row, 0].blocks)
            for col in range(1, bb.shape[1]):
                M = M.row_join(bb[row, col].blocks)
            MM = MM.col_join(M)
        return BlockMatrix(MM)
    except ShapeError:
        return B

def reblock_2x2(expr):
    if False:
        return 10
    '\n    Reblock a BlockMatrix so that it has 2x2 blocks of block matrices.  If\n    possible in such a way that the matrix continues to be invertible using the\n    classical 2x2 block inversion formulas.\n    '
    if not isinstance(expr, BlockMatrix) or not all((d > 2 for d in expr.blockshape)):
        return expr
    BM = BlockMatrix
    (rowblocks, colblocks) = expr.blockshape
    blocks = expr.blocks
    for i in range(1, rowblocks):
        for j in range(1, colblocks):
            A = bc_unpack(BM(blocks[:i, :j]))
            B = bc_unpack(BM(blocks[:i, j:]))
            C = bc_unpack(BM(blocks[i:, :j]))
            D = bc_unpack(BM(blocks[i:, j:]))
            formula = _choose_2x2_inversion_formula(A, B, C, D)
            if formula is not None:
                return BlockMatrix([[A, B], [C, D]])
    return BM([[blocks[0, 0], BM(blocks[0, 1:])], [BM(blocks[1:, 0]), BM(blocks[1:, 1:])]])

def bounds(sizes):
    if False:
        i = 10
        return i + 15
    ' Convert sequence of numbers into pairs of low-high pairs\n\n    >>> from sympy.matrices.expressions.blockmatrix import bounds\n    >>> bounds((1, 10, 50))\n    [(0, 1), (1, 11), (11, 61)]\n    '
    low = 0
    rv = []
    for size in sizes:
        rv.append((low, low + size))
        low += size
    return rv

def blockcut(expr, rowsizes, colsizes):
    if False:
        i = 10
        return i + 15
    " Cut a matrix expression into Blocks\n\n    >>> from sympy import ImmutableMatrix, blockcut\n    >>> M = ImmutableMatrix(4, 4, range(16))\n    >>> B = blockcut(M, (1, 3), (1, 3))\n    >>> type(B).__name__\n    'BlockMatrix'\n    >>> ImmutableMatrix(B.blocks[0, 1])\n    Matrix([[1, 2, 3]])\n    "
    rowbounds = bounds(rowsizes)
    colbounds = bounds(colsizes)
    return BlockMatrix([[MatrixSlice(expr, rowbound, colbound) for colbound in colbounds] for rowbound in rowbounds])