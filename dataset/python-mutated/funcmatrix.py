from .matexpr import MatrixExpr
from sympy.core.function import FunctionClass, Lambda
from sympy.core.symbol import Dummy
from sympy.core.sympify import _sympify, sympify
from sympy.matrices import Matrix
from sympy.functions.elementary.complexes import re, im

class FunctionMatrix(MatrixExpr):
    """Represents a matrix using a function (``Lambda``) which gives
    outputs according to the coordinates of each matrix entries.

    Parameters
    ==========

    rows : nonnegative integer. Can be symbolic.

    cols : nonnegative integer. Can be symbolic.

    lamda : Function, Lambda or str
        If it is a SymPy ``Function`` or ``Lambda`` instance,
        it should be able to accept two arguments which represents the
        matrix coordinates.

        If it is a pure string containing Python ``lambda`` semantics,
        it is interpreted by the SymPy parser and casted into a SymPy
        ``Lambda`` instance.

    Examples
    ========

    Creating a ``FunctionMatrix`` from ``Lambda``:

    >>> from sympy import FunctionMatrix, symbols, Lambda, MatPow
    >>> i, j, n, m = symbols('i,j,n,m')
    >>> FunctionMatrix(n, m, Lambda((i, j), i + j))
    FunctionMatrix(n, m, Lambda((i, j), i + j))

    Creating a ``FunctionMatrix`` from a SymPy function:

    >>> from sympy import KroneckerDelta
    >>> X = FunctionMatrix(3, 3, KroneckerDelta)
    >>> X.as_explicit()
    Matrix([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]])

    Creating a ``FunctionMatrix`` from a SymPy undefined function:

    >>> from sympy import Function
    >>> f = Function('f')
    >>> X = FunctionMatrix(3, 3, f)
    >>> X.as_explicit()
    Matrix([
    [f(0, 0), f(0, 1), f(0, 2)],
    [f(1, 0), f(1, 1), f(1, 2)],
    [f(2, 0), f(2, 1), f(2, 2)]])

    Creating a ``FunctionMatrix`` from Python ``lambda``:

    >>> FunctionMatrix(n, m, 'lambda i, j: i + j')
    FunctionMatrix(n, m, Lambda((i, j), i + j))

    Example of lazy evaluation of matrix product:

    >>> Y = FunctionMatrix(1000, 1000, Lambda((i, j), i + j))
    >>> isinstance(Y*Y, MatPow) # this is an expression object
    True
    >>> (Y**2)[10,10] # So this is evaluated lazily
    342923500

    Notes
    =====

    This class provides an alternative way to represent an extremely
    dense matrix with entries in some form of a sequence, in a most
    sparse way.
    """

    def __new__(cls, rows, cols, lamda):
        if False:
            i = 10
            return i + 15
        (rows, cols) = (_sympify(rows), _sympify(cols))
        cls._check_dim(rows)
        cls._check_dim(cols)
        lamda = sympify(lamda)
        if not isinstance(lamda, (FunctionClass, Lambda)):
            raise ValueError('{} should be compatible with SymPy function classes.'.format(lamda))
        if 2 not in lamda.nargs:
            raise ValueError('{} should be able to accept 2 arguments.'.format(lamda))
        if not isinstance(lamda, Lambda):
            (i, j) = (Dummy('i'), Dummy('j'))
            lamda = Lambda((i, j), lamda(i, j))
        return super().__new__(cls, rows, cols, lamda)

    @property
    def shape(self):
        if False:
            return 10
        return self.args[0:2]

    @property
    def lamda(self):
        if False:
            for i in range(10):
                print('nop')
        return self.args[2]

    def _entry(self, i, j, **kwargs):
        if False:
            while True:
                i = 10
        return self.lamda(i, j)

    def _eval_trace(self):
        if False:
            i = 10
            return i + 15
        from sympy.matrices.expressions.trace import Trace
        from sympy.concrete.summations import Sum
        return Trace(self).rewrite(Sum).doit()

    def _eval_as_real_imag(self):
        if False:
            return 10
        return (re(Matrix(self)), im(Matrix(self)))