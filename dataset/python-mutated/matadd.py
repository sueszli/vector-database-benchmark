from functools import reduce
import operator
from sympy.core import Basic, sympify
from sympy.core.add import add, Add, _could_extract_minus_sign
from sympy.core.sorting import default_sort_key
from sympy.functions import adjoint
from sympy.matrices.matrices import MatrixBase
from sympy.matrices.expressions.transpose import transpose
from sympy.strategies import rm_id, unpack, flatten, sort, condition, exhaust, do_one, glom
from sympy.matrices.expressions.matexpr import MatrixExpr
from sympy.matrices.expressions.special import ZeroMatrix, GenericZeroMatrix
from sympy.matrices.expressions._shape import validate_matadd_integer as validate
from sympy.utilities.iterables import sift
from sympy.utilities.exceptions import sympy_deprecation_warning

class MatAdd(MatrixExpr, Add):
    """A Sum of Matrix Expressions

    MatAdd inherits from and operates like SymPy Add

    Examples
    ========

    >>> from sympy import MatAdd, MatrixSymbol
    >>> A = MatrixSymbol('A', 5, 5)
    >>> B = MatrixSymbol('B', 5, 5)
    >>> C = MatrixSymbol('C', 5, 5)
    >>> MatAdd(A, B, C)
    A + B + C
    """
    is_MatAdd = True
    identity = GenericZeroMatrix()

    def __new__(cls, *args, evaluate=False, check=None, _sympify=True):
        if False:
            return 10
        if not args:
            return cls.identity
        args = list(filter(lambda i: cls.identity != i, args))
        if _sympify:
            args = list(map(sympify, args))
        if not all((isinstance(arg, MatrixExpr) for arg in args)):
            raise TypeError('Mix of Matrix and Scalar symbols')
        obj = Basic.__new__(cls, *args)
        if check is not None:
            sympy_deprecation_warning('Passing check to MatAdd is deprecated and the check argument will be removed in a future version.', deprecated_since_version='1.11', active_deprecations_target='remove-check-argument-from-matrix-operations')
        if check is not False:
            validate(*args)
        if evaluate:
            obj = cls._evaluate(obj)
        return obj

    @classmethod
    def _evaluate(cls, expr):
        if False:
            return 10
        return canonicalize(expr)

    @property
    def shape(self):
        if False:
            print('Hello World!')
        return self.args[0].shape

    def could_extract_minus_sign(self):
        if False:
            print('Hello World!')
        return _could_extract_minus_sign(self)

    def expand(self, **kwargs):
        if False:
            i = 10
            return i + 15
        expanded = super(MatAdd, self).expand(**kwargs)
        return self._evaluate(expanded)

    def _entry(self, i, j, **kwargs):
        if False:
            i = 10
            return i + 15
        return Add(*[arg._entry(i, j, **kwargs) for arg in self.args])

    def _eval_transpose(self):
        if False:
            for i in range(10):
                print('nop')
        return MatAdd(*[transpose(arg) for arg in self.args]).doit()

    def _eval_adjoint(self):
        if False:
            i = 10
            return i + 15
        return MatAdd(*[adjoint(arg) for arg in self.args]).doit()

    def _eval_trace(self):
        if False:
            while True:
                i = 10
        from .trace import trace
        return Add(*[trace(arg) for arg in self.args]).doit()

    def doit(self, **hints):
        if False:
            i = 10
            return i + 15
        deep = hints.get('deep', True)
        if deep:
            args = [arg.doit(**hints) for arg in self.args]
        else:
            args = self.args
        return canonicalize(MatAdd(*args))

    def _eval_derivative_matrix_lines(self, x):
        if False:
            return 10
        add_lines = [arg._eval_derivative_matrix_lines(x) for arg in self.args]
        return [j for i in add_lines for j in i]
add.register_handlerclass((Add, MatAdd), MatAdd)
factor_of = lambda arg: arg.as_coeff_mmul()[0]
matrix_of = lambda arg: unpack(arg.as_coeff_mmul()[1])

def combine(cnt, mat):
    if False:
        i = 10
        return i + 15
    if cnt == 1:
        return mat
    else:
        return cnt * mat

def merge_explicit(matadd):
    if False:
        print('Hello World!')
    " Merge explicit MatrixBase arguments\n\n    Examples\n    ========\n\n    >>> from sympy import MatrixSymbol, eye, Matrix, MatAdd, pprint\n    >>> from sympy.matrices.expressions.matadd import merge_explicit\n    >>> A = MatrixSymbol('A', 2, 2)\n    >>> B = eye(2)\n    >>> C = Matrix([[1, 2], [3, 4]])\n    >>> X = MatAdd(A, B, C)\n    >>> pprint(X)\n        [1  0]   [1  2]\n    A + [    ] + [    ]\n        [0  1]   [3  4]\n    >>> pprint(merge_explicit(X))\n        [2  2]\n    A + [    ]\n        [3  5]\n    "
    groups = sift(matadd.args, lambda arg: isinstance(arg, MatrixBase))
    if len(groups[True]) > 1:
        return MatAdd(*groups[False] + [reduce(operator.add, groups[True])])
    else:
        return matadd
rules = (rm_id(lambda x: x == 0 or isinstance(x, ZeroMatrix)), unpack, flatten, glom(matrix_of, factor_of, combine), merge_explicit, sort(default_sort_key))
canonicalize = exhaust(condition(lambda x: isinstance(x, MatAdd), do_one(*rules)))