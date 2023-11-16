from sympy.core.function import nfloat
from sympy.core.numbers import Float, I, Rational, pi
from sympy.core.relational import Eq
from sympy.core.symbol import Symbol, symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import sin
from sympy.integrals.integrals import Integral
from sympy.matrices.dense import Matrix
from mpmath import mnorm, mpf
from sympy.solvers import nsolve
from sympy.utilities.lambdify import lambdify
from sympy.testing.pytest import raises, XFAIL
from sympy.utilities.decorator import conserve_mpmath_dps

@XFAIL
def test_nsolve_fail():
    if False:
        for i in range(10):
            print('nop')
    x = symbols('x')
    ans = nsolve(x ** 2 / (1 - x) / (1 - 2 * x) ** 2 - 100, x, 0)
    assert ans > 0.46 and ans < 0.47

def test_nsolve_denominator():
    if False:
        while True:
            i = 10
    x = symbols('x')
    ans = nsolve((x ** 2 + 3 * x + 2) / (x + 2), -2.1)
    assert ans == -1.0

def test_nsolve():
    if False:
        print('Hello World!')
    x = Symbol('x')
    assert nsolve(sin(x), 2) - pi.evalf() < 1e-15
    assert nsolve(Eq(2 * x, 2), x, -10) == nsolve(2 * x - 2, -10)
    raises(TypeError, lambda : nsolve(Eq(2 * x, 2)))
    raises(TypeError, lambda : nsolve(Eq(2 * x, 2), x, 1, 2))
    x1 = Symbol('x1')
    x2 = Symbol('x2')
    f1 = 3 * x1 ** 2 - 2 * x2 ** 2 - 1
    f2 = x1 ** 2 - 2 * x1 + x2 ** 2 + 2 * x2 - 8
    f = Matrix((f1, f2)).T
    F = lambdify((x1, x2), f.T, modules='mpmath')
    for x0 in [(-1, 1), (1, -2), (4, 4), (-4, -4)]:
        x = nsolve(f, (x1, x2), x0, tol=1e-08)
        assert mnorm(F(*x), 1) <= 1e-10
    x = Symbol('x')
    y = Symbol('y')
    z = Symbol('z')
    f1 = -x + 2 * y
    f2 = (x ** 2 + x * (y ** 2 - 2) - 4 * y) / (x + 4)
    f3 = sqrt(x ** 2 + y ** 2) * z
    f = Matrix((f1, f2, f3)).T
    F = lambdify((x, y, z), f.T, modules='mpmath')

    def getroot(x0):
        if False:
            while True:
                i = 10
        root = nsolve(f, (x, y, z), x0)
        assert mnorm(F(*root), 1) <= 1e-08
        return root
    assert list(map(round, getroot((1, 1, 1)))) == [2, 1, 0]
    assert nsolve([Eq(f1, 0), Eq(f2, 0), Eq(f3, 0)], [x, y, z], (1, 1, 1))
    a = Symbol('a')
    assert abs(nsolve(1 / (0.001 + a) ** 3 - 6 / (0.9 - a) ** 3, a, 0.3) - mpf('0.31883011387318591')) < 1e-15

def test_issue_6408():
    if False:
        print('Hello World!')
    x = Symbol('x')
    assert nsolve(Piecewise((x, x < 1), (x ** 2, True)), x, 2) == 0.0

def test_issue_6408_integral():
    if False:
        while True:
            i = 10
    (x, y) = symbols('x y')
    assert nsolve(Integral(x * y, (x, 0, 5)), y, 2) == 0.0

@conserve_mpmath_dps
def test_increased_dps():
    if False:
        for i in range(10):
            print('nop')
    import mpmath
    mpmath.mp.dps = 128
    x = Symbol('x')
    e1 = x ** 2 - pi
    q = nsolve(e1, x, 3.0)
    assert abs(sqrt(pi).evalf(128) - q) < 1e-128

def test_nsolve_precision():
    if False:
        return 10
    (x, y) = symbols('x y')
    sol = nsolve(x ** 2 - pi, x, 3, prec=128)
    assert abs(sqrt(pi).evalf(128) - sol) < 1e-128
    assert isinstance(sol, Float)
    sols = nsolve((y ** 2 - x, x ** 2 - pi), (x, y), (3, 3), prec=128)
    assert isinstance(sols, Matrix)
    assert sols.shape == (2, 1)
    assert abs(sqrt(pi).evalf(128) - sols[0]) < 1e-128
    assert abs(sqrt(sqrt(pi)).evalf(128) - sols[1]) < 1e-128
    assert all((isinstance(i, Float) for i in sols))

def test_nsolve_complex():
    if False:
        while True:
            i = 10
    (x, y) = symbols('x y')
    assert nsolve(x ** 2 + 2, 1j) == sqrt(2.0) * I
    assert nsolve(x ** 2 + 2, I) == sqrt(2.0) * I
    assert nsolve([x ** 2 + 2, y ** 2 + 2], [x, y], [I, I]) == Matrix([sqrt(2.0) * I, sqrt(2.0) * I])
    assert nsolve([x ** 2 + 2, y ** 2 + 2], [x, y], [I, I]) == Matrix([sqrt(2.0) * I, sqrt(2.0) * I])

def test_nsolve_dict_kwarg():
    if False:
        while True:
            i = 10
    (x, y) = symbols('x y')
    assert nsolve(x ** 2 - 2, 1, dict=True) == [{x: sqrt(2.0)}]
    assert nsolve(x ** 2 + 2, I, dict=True) == [{x: sqrt(2.0) * I}]
    assert nsolve([x ** 2 + y ** 2 - 5, x ** 2 - y ** 2 + 1], [x, y], [1, 1], dict=True) == [{x: sqrt(2.0), y: sqrt(3.0)}]

def test_nsolve_rational():
    if False:
        for i in range(10):
            print('nop')
    x = symbols('x')
    assert nsolve(x - Rational(1, 3), 0, prec=100) == Rational(1, 3).evalf(100)

def test_issue_14950():
    if False:
        for i in range(10):
            print('nop')
    x = Matrix(symbols('t s'))
    x0 = Matrix([17, 23])
    eqn = x + x0
    assert nsolve(eqn, x, x0) == nfloat(-x0)
    assert nsolve(eqn.T, x.T, x0.T) == nfloat(-x0)