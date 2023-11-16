from sympy.concrete.summations import Sum
from sympy.core.expr import Expr
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import sin
from sympy.matrices.dense import MutableDenseMatrix as Matrix
from sympy.sets.sets import Interval
from sympy.utilities.lambdify import lambdify
from sympy.testing.pytest import raises
from sympy.printing.tensorflow import TensorflowPrinter
from sympy.printing.lambdarepr import lambdarepr, LambdaPrinter, NumExprPrinter
(x, y, z) = symbols('x,y,z')
(i, a, b) = symbols('i,a,b')
(j, c, d) = symbols('j,c,d')

def test_basic():
    if False:
        print('Hello World!')
    assert lambdarepr(x * y) == 'x*y'
    assert lambdarepr(x + y) in ['y + x', 'x + y']
    assert lambdarepr(x ** y) == 'x**y'

def test_matrix():
    if False:
        return 10
    e = x % 2
    assert lambdarepr(e) != str(e)
    assert lambdarepr(Matrix([e])) == 'ImmutableDenseMatrix([[x % 2]])'

def test_piecewise():
    if False:
        i = 10
        return i + 15
    h = 'lambda x: '
    p = Piecewise((x, x < 0))
    l = lambdarepr(p)
    eval(h + l)
    assert l == '((x) if (x < 0) else None)'
    p = Piecewise((1, x < 1), (2, x < 2), (0, True))
    l = lambdarepr(p)
    eval(h + l)
    assert l == '((1) if (x < 1) else (2) if (x < 2) else (0))'
    p = Piecewise((1, x < 1), (2, x < 2))
    l = lambdarepr(p)
    eval(h + l)
    assert l == '((1) if (x < 1) else (2) if (x < 2) else None)'
    p = Piecewise((x, x < 1), (x ** 2, Interval(3, 4, True, False).contains(x)), (0, True))
    l = lambdarepr(p)
    eval(h + l)
    assert l == '((x) if (x < 1) else (x**2) if (((x <= 4)) and ((x > 3))) else (0))'
    p = Piecewise((x ** 2, x < 0), (x, x < 1), (2 - x, x >= 1), (0, True), evaluate=False)
    l = lambdarepr(p)
    eval(h + l)
    assert l == '((x**2) if (x < 0) else (x) if (x < 1) else (2 - x) if (x >= 1) else (0))'
    p = Piecewise((x ** 2, x < 0), (x, x < 1), (2 - x, x >= 1), evaluate=False)
    l = lambdarepr(p)
    eval(h + l)
    assert l == '((x**2) if (x < 0) else (x) if (x < 1) else (2 - x) if (x >= 1) else None)'
    p = Piecewise((1, x >= 1), (2, x >= 2), (3, x >= 3), (4, x >= 4), (5, x >= 5), (6, True))
    l = lambdarepr(p)
    eval(h + l)
    assert l == '((1) if (x >= 1) else (2) if (x >= 2) else (3) if (x >= 3) else (4) if (x >= 4) else (5) if (x >= 5) else (6))'
    p = Piecewise((1, x <= 1), (2, x <= 2), (3, x <= 3), (4, x <= 4), (5, x <= 5), (6, True))
    l = lambdarepr(p)
    eval(h + l)
    assert l == '((1) if (x <= 1) else (2) if (x <= 2) else (3) if (x <= 3) else (4) if (x <= 4) else (5) if (x <= 5) else (6))'
    p = Piecewise((1, x > 1), (2, x > 2), (3, x > 3), (4, x > 4), (5, x > 5), (6, True))
    l = lambdarepr(p)
    eval(h + l)
    assert l == '((1) if (x > 1) else (2) if (x > 2) else (3) if (x > 3) else (4) if (x > 4) else (5) if (x > 5) else (6))'
    p = Piecewise((1, x < 1), (2, x < 2), (3, x < 3), (4, x < 4), (5, x < 5), (6, True))
    l = lambdarepr(p)
    eval(h + l)
    assert l == '((1) if (x < 1) else (2) if (x < 2) else (3) if (x < 3) else (4) if (x < 4) else (5) if (x < 5) else (6))'
    p = Piecewise((Piecewise((1, x > 0), (2, True)), y > 0), (3, True))
    l = lambdarepr(p)
    eval(h + l)
    assert l == '((((1) if (x > 0) else (2))) if (y > 0) else (3))'

def test_sum__1():
    if False:
        for i in range(10):
            print('nop')
    s = Sum(x ** i, (i, a, b))
    l = lambdarepr(s)
    assert l == '(builtins.sum(x**i for i in range(a, b+1)))'
    args = (x, a, b)
    f = lambdify(args, s)
    v = (2, 3, 8)
    assert f(*v) == s.subs(zip(args, v)).doit()

def test_sum__2():
    if False:
        i = 10
        return i + 15
    s = Sum(i * x, (i, a, b))
    l = lambdarepr(s)
    assert l == '(builtins.sum(i*x for i in range(a, b+1)))'
    args = (x, a, b)
    f = lambdify(args, s)
    v = (2, 3, 8)
    assert f(*v) == s.subs(zip(args, v)).doit()

def test_multiple_sums():
    if False:
        while True:
            i = 10
    s = Sum(i * x + j, (i, a, b), (j, c, d))
    l = lambdarepr(s)
    assert l == '(builtins.sum(i*x + j for i in range(a, b+1) for j in range(c, d+1)))'
    args = (x, a, b, c, d)
    f = lambdify(args, s)
    vals = (2, 3, 4, 5, 6)
    f_ref = s.subs(zip(args, vals)).doit()
    f_res = f(*vals)
    assert f_res == f_ref

def test_sqrt():
    if False:
        print('Hello World!')
    prntr = LambdaPrinter({'standard': 'python3'})
    assert prntr._print_Pow(sqrt(x), rational=False) == 'sqrt(x)'
    assert prntr._print_Pow(sqrt(x), rational=True) == 'x**(1/2)'

def test_settings():
    if False:
        return 10
    raises(TypeError, lambda : lambdarepr(sin(x), method='garbage'))

def test_numexpr():
    if False:
        while True:
            i = 10
    from sympy.logic.boolalg import ITE
    expr = ITE(x > 0, True, False, evaluate=False)
    assert NumExprPrinter().doprint(expr) == "numexpr.evaluate('where((x > 0), True, False)', truediv=True)"
    from sympy.codegen.ast import Return, FunctionDefinition, Variable, Assignment
    func_def = FunctionDefinition(None, 'foo', [Variable(x)], [Assignment(y, x), Return(y ** 2)])
    expected = "def foo(x):\n    y = numexpr.evaluate('x', truediv=True)\n    return numexpr.evaluate('y**2', truediv=True)"
    assert NumExprPrinter().doprint(func_def) == expected

class CustomPrintedObject(Expr):

    def _lambdacode(self, printer):
        if False:
            i = 10
            return i + 15
        return 'lambda'

    def _tensorflowcode(self, printer):
        if False:
            i = 10
            return i + 15
        return 'tensorflow'

    def _numpycode(self, printer):
        if False:
            i = 10
            return i + 15
        return 'numpy'

    def _numexprcode(self, printer):
        if False:
            print('Hello World!')
        return 'numexpr'

    def _mpmathcode(self, printer):
        if False:
            while True:
                i = 10
        return 'mpmath'

def test_printmethod():
    if False:
        i = 10
        return i + 15
    obj = CustomPrintedObject()
    assert LambdaPrinter().doprint(obj) == 'lambda'
    assert TensorflowPrinter().doprint(obj) == 'tensorflow'
    assert NumExprPrinter().doprint(obj) == "numexpr.evaluate('numexpr', truediv=True)"
    assert NumExprPrinter().doprint(Piecewise((y, x >= 0), (z, x < 0))) == "numexpr.evaluate('where((x >= 0), y, z)', truediv=True)"