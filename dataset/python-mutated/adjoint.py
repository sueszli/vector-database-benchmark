from sympy.core import Basic
from sympy.functions import adjoint, conjugate
from sympy.matrices.expressions.matexpr import MatrixExpr

class Adjoint(MatrixExpr):
    """
    The Hermitian adjoint of a matrix expression.

    This is a symbolic object that simply stores its argument without
    evaluating it. To actually compute the adjoint, use the ``adjoint()``
    function.

    Examples
    ========

    >>> from sympy import MatrixSymbol, Adjoint, adjoint
    >>> A = MatrixSymbol('A', 3, 5)
    >>> B = MatrixSymbol('B', 5, 3)
    >>> Adjoint(A*B)
    Adjoint(A*B)
    >>> adjoint(A*B)
    Adjoint(B)*Adjoint(A)
    >>> adjoint(A*B) == Adjoint(A*B)
    False
    >>> adjoint(A*B) == Adjoint(A*B).doit()
    True
    """
    is_Adjoint = True

    def doit(self, **hints):
        if False:
            return 10
        arg = self.arg
        if hints.get('deep', True) and isinstance(arg, Basic):
            return adjoint(arg.doit(**hints))
        else:
            return adjoint(self.arg)

    @property
    def arg(self):
        if False:
            while True:
                i = 10
        return self.args[0]

    @property
    def shape(self):
        if False:
            while True:
                i = 10
        return self.arg.shape[::-1]

    def _entry(self, i, j, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return conjugate(self.arg._entry(j, i, **kwargs))

    def _eval_adjoint(self):
        if False:
            while True:
                i = 10
        return self.arg

    def _eval_transpose(self):
        if False:
            i = 10
            return i + 15
        return self.arg.conjugate()

    def _eval_conjugate(self):
        if False:
            return 10
        return self.arg.transpose()

    def _eval_trace(self):
        if False:
            return 10
        from sympy.matrices.expressions.trace import Trace
        return conjugate(Trace(self.arg))