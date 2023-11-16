from sympy.core.relational import Eq
from sympy.core.expr import Expr
from sympy.core.numbers import Integer
from sympy.logic.boolalg import Boolean, And
from sympy.matrices.expressions.matexpr import MatrixExpr
from sympy.matrices.common import ShapeError
from typing import Union

def is_matadd_valid(*args: MatrixExpr) -> Boolean:
    if False:
        while True:
            i = 10
    "Return the symbolic condition how ``MatAdd``, ``HadamardProduct``\n    makes sense.\n\n    Parameters\n    ==========\n\n    args\n        The list of arguments of matrices to be tested for.\n\n    Examples\n    ========\n\n    >>> from sympy import MatrixSymbol, symbols\n    >>> from sympy.matrices.expressions._shape import is_matadd_valid\n\n    >>> m, n, p, q = symbols('m n p q')\n    >>> A = MatrixSymbol('A', m, n)\n    >>> B = MatrixSymbol('B', p, q)\n    >>> is_matadd_valid(A, B)\n    Eq(m, p) & Eq(n, q)\n    "
    (rows, cols) = zip(*(arg.shape for arg in args))
    return And(*(Eq(i, j) for (i, j) in zip(rows[:-1], rows[1:])), *(Eq(i, j) for (i, j) in zip(cols[:-1], cols[1:])))

def is_matmul_valid(*args: Union[MatrixExpr, Expr]) -> Boolean:
    if False:
        return 10
    "Return the symbolic condition how ``MatMul`` makes sense\n\n    Parameters\n    ==========\n\n    args\n        The list of arguments of matrices and scalar expressions to be tested\n        for.\n\n    Examples\n    ========\n\n    >>> from sympy import MatrixSymbol, symbols\n    >>> from sympy.matrices.expressions._shape import is_matmul_valid\n\n    >>> m, n, p, q = symbols('m n p q')\n    >>> A = MatrixSymbol('A', m, n)\n    >>> B = MatrixSymbol('B', p, q)\n    >>> is_matmul_valid(A, B)\n    Eq(n, p)\n    "
    (rows, cols) = zip(*(arg.shape for arg in args if isinstance(arg, MatrixExpr)))
    return And(*(Eq(i, j) for (i, j) in zip(cols[:-1], rows[1:])))

def is_square(arg: MatrixExpr, /) -> Boolean:
    if False:
        for i in range(10):
            print('nop')
    "Return the symbolic condition how the matrix is assumed to be square\n\n    Parameters\n    ==========\n\n    arg\n        The matrix to be tested for.\n\n    Examples\n    ========\n\n    >>> from sympy import MatrixSymbol, symbols\n    >>> from sympy.matrices.expressions._shape import is_square\n\n    >>> m, n = symbols('m n')\n    >>> A = MatrixSymbol('A', m, n)\n    >>> is_square(A)\n    Eq(m, n)\n    "
    return Eq(arg.rows, arg.cols)

def validate_matadd_integer(*args: MatrixExpr) -> None:
    if False:
        i = 10
        return i + 15
    'Validate matrix shape for addition only for integer values'
    (rows, cols) = zip(*(x.shape for x in args))
    if len(set(filter(lambda x: isinstance(x, (int, Integer)), rows))) > 1:
        raise ShapeError(f'Matrices have mismatching shape: {rows}')
    if len(set(filter(lambda x: isinstance(x, (int, Integer)), cols))) > 1:
        raise ShapeError(f'Matrices have mismatching shape: {cols}')

def validate_matmul_integer(*args: MatrixExpr) -> None:
    if False:
        i = 10
        return i + 15
    'Validate matrix shape for multiplication only for integer values'
    for (A, B) in zip(args[:-1], args[1:]):
        (i, j) = (A.cols, B.rows)
        if isinstance(i, (int, Integer)) and isinstance(j, (int, Integer)) and (i != j):
            raise ShapeError('Matrices are not aligned', i, j)