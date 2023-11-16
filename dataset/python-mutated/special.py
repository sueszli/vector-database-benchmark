from sympy.assumptions.ask import ask, Q
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.sympify import _sympify
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.matrices.common import NonInvertibleMatrixError
from .matexpr import MatrixExpr

class ZeroMatrix(MatrixExpr):
    """The Matrix Zero 0 - additive identity

    Examples
    ========

    >>> from sympy import MatrixSymbol, ZeroMatrix
    >>> A = MatrixSymbol('A', 3, 5)
    >>> Z = ZeroMatrix(3, 5)
    >>> A + Z
    A
    >>> Z*A.T
    0
    """
    is_ZeroMatrix = True

    def __new__(cls, m, n):
        if False:
            print('Hello World!')
        (m, n) = (_sympify(m), _sympify(n))
        cls._check_dim(m)
        cls._check_dim(n)
        return super().__new__(cls, m, n)

    @property
    def shape(self):
        if False:
            i = 10
            return i + 15
        return (self.args[0], self.args[1])

    def _eval_power(self, exp):
        if False:
            print('Hello World!')
        if (exp < 0) == True:
            raise NonInvertibleMatrixError('Matrix det == 0; not invertible')
        return self

    def _eval_transpose(self):
        if False:
            for i in range(10):
                print('nop')
        return ZeroMatrix(self.cols, self.rows)

    def _eval_adjoint(self):
        if False:
            i = 10
            return i + 15
        return ZeroMatrix(self.cols, self.rows)

    def _eval_trace(self):
        if False:
            print('Hello World!')
        return S.Zero

    def _eval_determinant(self):
        if False:
            while True:
                i = 10
        return S.Zero

    def _eval_inverse(self):
        if False:
            for i in range(10):
                print('nop')
        raise NonInvertibleMatrixError('Matrix det == 0; not invertible.')

    def _eval_as_real_imag(self):
        if False:
            print('Hello World!')
        return (self, self)

    def _eval_conjugate(self):
        if False:
            return 10
        return self

    def _entry(self, i, j, **kwargs):
        if False:
            i = 10
            return i + 15
        return S.Zero

class GenericZeroMatrix(ZeroMatrix):
    """
    A zero matrix without a specified shape

    This exists primarily so MatAdd() with no arguments can return something
    meaningful.
    """

    def __new__(cls):
        if False:
            while True:
                i = 10
        return super(ZeroMatrix, cls).__new__(cls)

    @property
    def rows(self):
        if False:
            print('Hello World!')
        raise TypeError('GenericZeroMatrix does not have a specified shape')

    @property
    def cols(self):
        if False:
            return 10
        raise TypeError('GenericZeroMatrix does not have a specified shape')

    @property
    def shape(self):
        if False:
            return 10
        raise TypeError('GenericZeroMatrix does not have a specified shape')

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return isinstance(other, GenericZeroMatrix)

    def __ne__(self, other):
        if False:
            i = 10
            return i + 15
        return not self == other

    def __hash__(self):
        if False:
            for i in range(10):
                print('nop')
        return super().__hash__()

class Identity(MatrixExpr):
    """The Matrix Identity I - multiplicative identity

    Examples
    ========

    >>> from sympy import Identity, MatrixSymbol
    >>> A = MatrixSymbol('A', 3, 5)
    >>> I = Identity(3)
    >>> I*A
    A
    """
    is_Identity = True

    def __new__(cls, n):
        if False:
            while True:
                i = 10
        n = _sympify(n)
        cls._check_dim(n)
        return super().__new__(cls, n)

    @property
    def rows(self):
        if False:
            print('Hello World!')
        return self.args[0]

    @property
    def cols(self):
        if False:
            print('Hello World!')
        return self.args[0]

    @property
    def shape(self):
        if False:
            return 10
        return (self.args[0], self.args[0])

    @property
    def is_square(self):
        if False:
            print('Hello World!')
        return True

    def _eval_transpose(self):
        if False:
            for i in range(10):
                print('nop')
        return self

    def _eval_trace(self):
        if False:
            i = 10
            return i + 15
        return self.rows

    def _eval_inverse(self):
        if False:
            print('Hello World!')
        return self

    def _eval_as_real_imag(self):
        if False:
            i = 10
            return i + 15
        return (self, ZeroMatrix(*self.shape))

    def _eval_conjugate(self):
        if False:
            for i in range(10):
                print('nop')
        return self

    def _eval_adjoint(self):
        if False:
            while True:
                i = 10
        return self

    def _entry(self, i, j, **kwargs):
        if False:
            print('Hello World!')
        eq = Eq(i, j)
        if eq is S.true:
            return S.One
        elif eq is S.false:
            return S.Zero
        return KroneckerDelta(i, j, (0, self.cols - 1))

    def _eval_determinant(self):
        if False:
            while True:
                i = 10
        return S.One

    def _eval_power(self, exp):
        if False:
            while True:
                i = 10
        return self

class GenericIdentity(Identity):
    """
    An identity matrix without a specified shape

    This exists primarily so MatMul() with no arguments can return something
    meaningful.
    """

    def __new__(cls):
        if False:
            i = 10
            return i + 15
        return super(Identity, cls).__new__(cls)

    @property
    def rows(self):
        if False:
            return 10
        raise TypeError('GenericIdentity does not have a specified shape')

    @property
    def cols(self):
        if False:
            print('Hello World!')
        raise TypeError('GenericIdentity does not have a specified shape')

    @property
    def shape(self):
        if False:
            for i in range(10):
                print('nop')
        raise TypeError('GenericIdentity does not have a specified shape')

    @property
    def is_square(self):
        if False:
            print('Hello World!')
        return True

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return isinstance(other, GenericIdentity)

    def __ne__(self, other):
        if False:
            i = 10
            return i + 15
        return not self == other

    def __hash__(self):
        if False:
            for i in range(10):
                print('nop')
        return super().__hash__()

class OneMatrix(MatrixExpr):
    """
    Matrix whose all entries are ones.
    """

    def __new__(cls, m, n, evaluate=False):
        if False:
            while True:
                i = 10
        (m, n) = (_sympify(m), _sympify(n))
        cls._check_dim(m)
        cls._check_dim(n)
        if evaluate:
            condition = Eq(m, 1) & Eq(n, 1)
            if condition == True:
                return Identity(1)
        obj = super().__new__(cls, m, n)
        return obj

    @property
    def shape(self):
        if False:
            return 10
        return self._args

    @property
    def is_Identity(self):
        if False:
            while True:
                i = 10
        return self._is_1x1() == True

    def as_explicit(self):
        if False:
            while True:
                i = 10
        from sympy.matrices.immutable import ImmutableDenseMatrix
        return ImmutableDenseMatrix.ones(*self.shape)

    def doit(self, **hints):
        if False:
            for i in range(10):
                print('nop')
        args = self.args
        if hints.get('deep', True):
            args = [a.doit(**hints) for a in args]
        return self.func(*args, evaluate=True)

    def _eval_power(self, exp):
        if False:
            print('Hello World!')
        if self._is_1x1() == True:
            return Identity(1)
        if (exp < 0) == True:
            raise NonInvertibleMatrixError('Matrix det == 0; not invertible')
        if ask(Q.integer(exp)):
            return self.shape[0] ** (exp - 1) * OneMatrix(*self.shape)
        return super()._eval_power(exp)

    def _eval_transpose(self):
        if False:
            i = 10
            return i + 15
        return OneMatrix(self.cols, self.rows)

    def _eval_adjoint(self):
        if False:
            for i in range(10):
                print('nop')
        return OneMatrix(self.cols, self.rows)

    def _eval_trace(self):
        if False:
            return 10
        return S.One * self.rows

    def _is_1x1(self):
        if False:
            while True:
                i = 10
        'Returns true if the matrix is known to be 1x1'
        shape = self.shape
        return Eq(shape[0], 1) & Eq(shape[1], 1)

    def _eval_determinant(self):
        if False:
            print('Hello World!')
        condition = self._is_1x1()
        if condition == True:
            return S.One
        elif condition == False:
            return S.Zero
        else:
            from sympy.matrices.expressions.determinant import Determinant
            return Determinant(self)

    def _eval_inverse(self):
        if False:
            print('Hello World!')
        condition = self._is_1x1()
        if condition == True:
            return Identity(1)
        elif condition == False:
            raise NonInvertibleMatrixError('Matrix det == 0; not invertible.')
        else:
            from .inverse import Inverse
            return Inverse(self)

    def _eval_as_real_imag(self):
        if False:
            while True:
                i = 10
        return (self, ZeroMatrix(*self.shape))

    def _eval_conjugate(self):
        if False:
            for i in range(10):
                print('nop')
        return self

    def _entry(self, i, j, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return S.One