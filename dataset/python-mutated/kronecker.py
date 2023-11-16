"""Implementation of the Kronecker product"""
from functools import reduce
from math import prod
from sympy.core import Mul, sympify
from sympy.functions import adjoint
from sympy.matrices.common import ShapeError
from sympy.matrices.expressions.matexpr import MatrixExpr
from sympy.matrices.expressions.transpose import transpose
from sympy.matrices.expressions.special import Identity
from sympy.matrices.matrices import MatrixBase
from sympy.strategies import canon, condition, distribute, do_one, exhaust, flatten, typed, unpack
from sympy.strategies.traverse import bottom_up
from sympy.utilities import sift
from .matadd import MatAdd
from .matmul import MatMul
from .matpow import MatPow

def kronecker_product(*matrices):
    if False:
        print('Hello World!')
    "\n    The Kronecker product of two or more arguments.\n\n    This computes the explicit Kronecker product for subclasses of\n    ``MatrixBase`` i.e. explicit matrices. Otherwise, a symbolic\n    ``KroneckerProduct`` object is returned.\n\n\n    Examples\n    ========\n\n    For ``MatrixSymbol`` arguments a ``KroneckerProduct`` object is returned.\n    Elements of this matrix can be obtained by indexing, or for MatrixSymbols\n    with known dimension the explicit matrix can be obtained with\n    ``.as_explicit()``\n\n    >>> from sympy import kronecker_product, MatrixSymbol\n    >>> A = MatrixSymbol('A', 2, 2)\n    >>> B = MatrixSymbol('B', 2, 2)\n    >>> kronecker_product(A)\n    A\n    >>> kronecker_product(A, B)\n    KroneckerProduct(A, B)\n    >>> kronecker_product(A, B)[0, 1]\n    A[0, 0]*B[0, 1]\n    >>> kronecker_product(A, B).as_explicit()\n    Matrix([\n        [A[0, 0]*B[0, 0], A[0, 0]*B[0, 1], A[0, 1]*B[0, 0], A[0, 1]*B[0, 1]],\n        [A[0, 0]*B[1, 0], A[0, 0]*B[1, 1], A[0, 1]*B[1, 0], A[0, 1]*B[1, 1]],\n        [A[1, 0]*B[0, 0], A[1, 0]*B[0, 1], A[1, 1]*B[0, 0], A[1, 1]*B[0, 1]],\n        [A[1, 0]*B[1, 0], A[1, 0]*B[1, 1], A[1, 1]*B[1, 0], A[1, 1]*B[1, 1]]])\n\n    For explicit matrices the Kronecker product is returned as a Matrix\n\n    >>> from sympy import Matrix, kronecker_product\n    >>> sigma_x = Matrix([\n    ... [0, 1],\n    ... [1, 0]])\n    ...\n    >>> Isigma_y = Matrix([\n    ... [0, 1],\n    ... [-1, 0]])\n    ...\n    >>> kronecker_product(sigma_x, Isigma_y)\n    Matrix([\n    [ 0, 0,  0, 1],\n    [ 0, 0, -1, 0],\n    [ 0, 1,  0, 0],\n    [-1, 0,  0, 0]])\n\n    See Also\n    ========\n        KroneckerProduct\n\n    "
    if not matrices:
        raise TypeError('Empty Kronecker product is undefined')
    if len(matrices) == 1:
        return matrices[0]
    else:
        return KroneckerProduct(*matrices).doit()

class KroneckerProduct(MatrixExpr):
    """
    The Kronecker product of two or more arguments.

    The Kronecker product is a non-commutative product of matrices.
    Given two matrices of dimension (m, n) and (s, t) it produces a matrix
    of dimension (m s, n t).

    This is a symbolic object that simply stores its argument without
    evaluating it. To actually compute the product, use the function
    ``kronecker_product()`` or call the ``.doit()`` or  ``.as_explicit()``
    methods.

    >>> from sympy import KroneckerProduct, MatrixSymbol
    >>> A = MatrixSymbol('A', 5, 5)
    >>> B = MatrixSymbol('B', 5, 5)
    >>> isinstance(KroneckerProduct(A, B), KroneckerProduct)
    True
    """
    is_KroneckerProduct = True

    def __new__(cls, *args, check=True):
        if False:
            i = 10
            return i + 15
        args = list(map(sympify, args))
        if all((a.is_Identity for a in args)):
            ret = Identity(prod((a.rows for a in args)))
            if all((isinstance(a, MatrixBase) for a in args)):
                return ret.as_explicit()
            else:
                return ret
        if check:
            validate(*args)
        return super().__new__(cls, *args)

    @property
    def shape(self):
        if False:
            while True:
                i = 10
        (rows, cols) = self.args[0].shape
        for mat in self.args[1:]:
            rows *= mat.rows
            cols *= mat.cols
        return (rows, cols)

    def _entry(self, i, j, **kwargs):
        if False:
            print('Hello World!')
        result = 1
        for mat in reversed(self.args):
            (i, m) = divmod(i, mat.rows)
            (j, n) = divmod(j, mat.cols)
            result *= mat[m, n]
        return result

    def _eval_adjoint(self):
        if False:
            i = 10
            return i + 15
        return KroneckerProduct(*list(map(adjoint, self.args))).doit()

    def _eval_conjugate(self):
        if False:
            while True:
                i = 10
        return KroneckerProduct(*[a.conjugate() for a in self.args]).doit()

    def _eval_transpose(self):
        if False:
            for i in range(10):
                print('nop')
        return KroneckerProduct(*list(map(transpose, self.args))).doit()

    def _eval_trace(self):
        if False:
            print('Hello World!')
        from .trace import trace
        return Mul(*[trace(a) for a in self.args])

    def _eval_determinant(self):
        if False:
            for i in range(10):
                print('nop')
        from .determinant import det, Determinant
        if not all((a.is_square for a in self.args)):
            return Determinant(self)
        m = self.rows
        return Mul(*[det(a) ** (m / a.rows) for a in self.args])

    def _eval_inverse(self):
        if False:
            return 10
        try:
            return KroneckerProduct(*[a.inverse() for a in self.args])
        except ShapeError:
            from sympy.matrices.expressions.inverse import Inverse
            return Inverse(self)

    def structurally_equal(self, other):
        if False:
            for i in range(10):
                print('nop')
        "Determine whether two matrices have the same Kronecker product structure\n\n        Examples\n        ========\n\n        >>> from sympy import KroneckerProduct, MatrixSymbol, symbols\n        >>> m, n = symbols(r'm, n', integer=True)\n        >>> A = MatrixSymbol('A', m, m)\n        >>> B = MatrixSymbol('B', n, n)\n        >>> C = MatrixSymbol('C', m, m)\n        >>> D = MatrixSymbol('D', n, n)\n        >>> KroneckerProduct(A, B).structurally_equal(KroneckerProduct(C, D))\n        True\n        >>> KroneckerProduct(A, B).structurally_equal(KroneckerProduct(D, C))\n        False\n        >>> KroneckerProduct(A, B).structurally_equal(C)\n        False\n        "
        return isinstance(other, KroneckerProduct) and self.shape == other.shape and (len(self.args) == len(other.args)) and all((a.shape == b.shape for (a, b) in zip(self.args, other.args)))

    def has_matching_shape(self, other):
        if False:
            return 10
        "Determine whether two matrices have the appropriate structure to bring matrix\n        multiplication inside the KroneckerProdut\n\n        Examples\n        ========\n        >>> from sympy import KroneckerProduct, MatrixSymbol, symbols\n        >>> m, n = symbols(r'm, n', integer=True)\n        >>> A = MatrixSymbol('A', m, n)\n        >>> B = MatrixSymbol('B', n, m)\n        >>> KroneckerProduct(A, B).has_matching_shape(KroneckerProduct(B, A))\n        True\n        >>> KroneckerProduct(A, B).has_matching_shape(KroneckerProduct(A, B))\n        False\n        >>> KroneckerProduct(A, B).has_matching_shape(A)\n        False\n        "
        return isinstance(other, KroneckerProduct) and self.cols == other.rows and (len(self.args) == len(other.args)) and all((a.cols == b.rows for (a, b) in zip(self.args, other.args)))

    def _eval_expand_kroneckerproduct(self, **hints):
        if False:
            while True:
                i = 10
        return flatten(canon(typed({KroneckerProduct: distribute(KroneckerProduct, MatAdd)}))(self))

    def _kronecker_add(self, other):
        if False:
            i = 10
            return i + 15
        if self.structurally_equal(other):
            return self.__class__(*[a + b for (a, b) in zip(self.args, other.args)])
        else:
            return self + other

    def _kronecker_mul(self, other):
        if False:
            for i in range(10):
                print('nop')
        if self.has_matching_shape(other):
            return self.__class__(*[a * b for (a, b) in zip(self.args, other.args)])
        else:
            return self * other

    def doit(self, **hints):
        if False:
            for i in range(10):
                print('nop')
        deep = hints.get('deep', True)
        if deep:
            args = [arg.doit(**hints) for arg in self.args]
        else:
            args = self.args
        return canonicalize(KroneckerProduct(*args))

def validate(*args):
    if False:
        for i in range(10):
            print('nop')
    if not all((arg.is_Matrix for arg in args)):
        raise TypeError('Mix of Matrix and Scalar symbols')

def extract_commutative(kron):
    if False:
        i = 10
        return i + 15
    c_part = []
    nc_part = []
    for arg in kron.args:
        (c, nc) = arg.args_cnc()
        c_part.extend(c)
        nc_part.append(Mul._from_args(nc))
    c_part = Mul(*c_part)
    if c_part != 1:
        return c_part * KroneckerProduct(*nc_part)
    return kron

def matrix_kronecker_product(*matrices):
    if False:
        print('Hello World!')
    'Compute the Kronecker product of a sequence of SymPy Matrices.\n\n    This is the standard Kronecker product of matrices [1].\n\n    Parameters\n    ==========\n\n    matrices : tuple of MatrixBase instances\n        The matrices to take the Kronecker product of.\n\n    Returns\n    =======\n\n    matrix : MatrixBase\n        The Kronecker product matrix.\n\n    Examples\n    ========\n\n    >>> from sympy import Matrix\n    >>> from sympy.matrices.expressions.kronecker import (\n    ... matrix_kronecker_product)\n\n    >>> m1 = Matrix([[1,2],[3,4]])\n    >>> m2 = Matrix([[1,0],[0,1]])\n    >>> matrix_kronecker_product(m1, m2)\n    Matrix([\n    [1, 0, 2, 0],\n    [0, 1, 0, 2],\n    [3, 0, 4, 0],\n    [0, 3, 0, 4]])\n    >>> matrix_kronecker_product(m2, m1)\n    Matrix([\n    [1, 2, 0, 0],\n    [3, 4, 0, 0],\n    [0, 0, 1, 2],\n    [0, 0, 3, 4]])\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Kronecker_product\n    '
    if not all((isinstance(m, MatrixBase) for m in matrices)):
        raise TypeError('Sequence of Matrices expected, got: %s' % repr(matrices))
    matrix_expansion = matrices[-1]
    for mat in reversed(matrices[:-1]):
        rows = mat.rows
        cols = mat.cols
        for i in range(rows):
            start = matrix_expansion * mat[i * cols]
            for j in range(cols - 1):
                start = start.row_join(matrix_expansion * mat[i * cols + j + 1])
            if i == 0:
                next = start
            else:
                next = next.col_join(start)
        matrix_expansion = next
    MatrixClass = max(matrices, key=lambda M: M._class_priority).__class__
    if isinstance(matrix_expansion, MatrixClass):
        return matrix_expansion
    else:
        return MatrixClass(matrix_expansion)

def explicit_kronecker_product(kron):
    if False:
        return 10
    if not all((isinstance(m, MatrixBase) for m in kron.args)):
        return kron
    return matrix_kronecker_product(*kron.args)
rules = (unpack, explicit_kronecker_product, flatten, extract_commutative)
canonicalize = exhaust(condition(lambda x: isinstance(x, KroneckerProduct), do_one(*rules)))

def _kronecker_dims_key(expr):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(expr, KroneckerProduct):
        return tuple((a.shape for a in expr.args))
    else:
        return (0,)

def kronecker_mat_add(expr):
    if False:
        print('Hello World!')
    args = sift(expr.args, _kronecker_dims_key)
    nonkrons = args.pop((0,), None)
    if not args:
        return expr
    krons = [reduce(lambda x, y: x._kronecker_add(y), group) for group in args.values()]
    if not nonkrons:
        return MatAdd(*krons)
    else:
        return MatAdd(*krons) + nonkrons

def kronecker_mat_mul(expr):
    if False:
        print('Hello World!')
    (factor, matrices) = expr.as_coeff_matrices()
    i = 0
    while i < len(matrices) - 1:
        (A, B) = matrices[i:i + 2]
        if isinstance(A, KroneckerProduct) and isinstance(B, KroneckerProduct):
            matrices[i] = A._kronecker_mul(B)
            matrices.pop(i + 1)
        else:
            i += 1
    return factor * MatMul(*matrices)

def kronecker_mat_pow(expr):
    if False:
        while True:
            i = 10
    if isinstance(expr.base, KroneckerProduct) and all((a.is_square for a in expr.base.args)):
        return KroneckerProduct(*[MatPow(a, expr.exp) for a in expr.base.args])
    else:
        return expr

def combine_kronecker(expr):
    if False:
        for i in range(10):
            print('nop')
    "Combine KronekeckerProduct with expression.\n\n    If possible write operations on KroneckerProducts of compatible shapes\n    as a single KroneckerProduct.\n\n    Examples\n    ========\n\n    >>> from sympy.matrices.expressions import combine_kronecker\n    >>> from sympy import MatrixSymbol, KroneckerProduct, symbols\n    >>> m, n = symbols(r'm, n', integer=True)\n    >>> A = MatrixSymbol('A', m, n)\n    >>> B = MatrixSymbol('B', n, m)\n    >>> combine_kronecker(KroneckerProduct(A, B)*KroneckerProduct(B, A))\n    KroneckerProduct(A*B, B*A)\n    >>> combine_kronecker(KroneckerProduct(A, B)+KroneckerProduct(B.T, A.T))\n    KroneckerProduct(A + B.T, B + A.T)\n    >>> C = MatrixSymbol('C', n, n)\n    >>> D = MatrixSymbol('D', m, m)\n    >>> combine_kronecker(KroneckerProduct(C, D)**m)\n    KroneckerProduct(C**m, D**m)\n    "

    def haskron(expr):
        if False:
            while True:
                i = 10
        return isinstance(expr, MatrixExpr) and expr.has(KroneckerProduct)
    rule = exhaust(bottom_up(exhaust(condition(haskron, typed({MatAdd: kronecker_mat_add, MatMul: kronecker_mat_mul, MatPow: kronecker_mat_pow})))))
    result = rule(expr)
    doit = getattr(result, 'doit', None)
    if doit is not None:
        return doit()
    else:
        return result