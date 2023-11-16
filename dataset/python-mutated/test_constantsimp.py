"""
If the arbitrary constant class from issue 4435 is ever implemented, this
should serve as a set of test cases.
"""
from sympy.core.function import Function
from sympy.core.numbers import I
from sympy.core.power import Pow
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.hyperbolic import cosh, sinh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import acos, cos, sin
from sympy.integrals.integrals import Integral
from sympy.solvers.ode.ode import constantsimp, constant_renumber
from sympy.testing.pytest import XFAIL
x = Symbol('x')
y = Symbol('y')
z = Symbol('z')
u2 = Symbol('u2')
_a = Symbol('_a')
C1 = Symbol('C1')
C2 = Symbol('C2')
C3 = Symbol('C3')
f = Function('f')

def test_constant_mul():
    if False:
        print('Hello World!')
    assert constant_renumber(constantsimp(y * C1, [C1])) == C1 * y
    assert constant_renumber(constantsimp(C1 * y, [C1])) == C1 * y
    assert constant_renumber(constantsimp(x * C1, [C1])) == x * C1
    assert constant_renumber(constantsimp(C1 * x, [C1])) == x * C1
    assert constant_renumber(constantsimp(2 * C1, [C1])) == C1
    assert constant_renumber(constantsimp(C1 * 2, [C1])) == C1
    assert constant_renumber(constantsimp(y * C1 * x, [C1, y])) == C1 * x
    assert constant_renumber(constantsimp(x * y * C1, [C1, y])) == x * C1
    assert constant_renumber(constantsimp(y * x * C1, [C1, y])) == x * C1
    assert constant_renumber(constantsimp(C1 * x * y, [C1, y])) == C1 * x
    assert constant_renumber(constantsimp(x * C1 * y, [C1, y])) == x * C1
    assert constant_renumber(constantsimp(C1 * y * (y + 1), [C1])) == C1 * y * (y + 1)
    assert constant_renumber(constantsimp(y * C1 * (y + 1), [C1])) == C1 * y * (y + 1)
    assert constant_renumber(constantsimp(x * (y * C1), [C1])) == x * y * C1
    assert constant_renumber(constantsimp(x * (C1 * y), [C1])) == x * y * C1
    assert constant_renumber(constantsimp(C1 * (x * y), [C1, y])) == C1 * x
    assert constant_renumber(constantsimp(x * y * C1, [C1, y])) == x * C1
    assert constant_renumber(constantsimp(y * x * C1, [C1, y])) == x * C1
    assert constant_renumber(constantsimp(y * (y + 1) * C1, [C1, y])) == C1
    assert constant_renumber(constantsimp(C1 * x * y, [C1, y])) == C1 * x
    assert constant_renumber(constantsimp(y * (x * C1), [C1, y])) == x * C1
    assert constant_renumber(constantsimp(x * C1 * y, [C1, y])) == x * C1
    assert constant_renumber(constantsimp(C1 * x * y * x * y * 2, [C1, y])) == C1 * x ** 2
    assert constant_renumber(constantsimp(C1 * x * y * z, [C1, y, z])) == C1 * x
    assert constant_renumber(constantsimp(C1 * x * y ** 2 * sin(z), [C1, y, z])) == C1 * x
    assert constant_renumber(constantsimp(C1 * C1, [C1])) == C1
    assert constant_renumber(constantsimp(C1 * C2, [C1, C2])) == C1
    assert constant_renumber(constantsimp(C2 * C2, [C1, C2])) == C1
    assert constant_renumber(constantsimp(C1 * C1 * C2, [C1, C2])) == C1
    assert constant_renumber(constantsimp(C1 * x * 2 ** x, [C1])) == C1 * x * 2 ** x

def test_constant_add():
    if False:
        for i in range(10):
            print('nop')
    assert constant_renumber(constantsimp(C1 + C1, [C1])) == C1
    assert constant_renumber(constantsimp(C1 + 2, [C1])) == C1
    assert constant_renumber(constantsimp(2 + C1, [C1])) == C1
    assert constant_renumber(constantsimp(C1 + y, [C1, y])) == C1
    assert constant_renumber(constantsimp(C1 + x, [C1])) == C1 + x
    assert constant_renumber(constantsimp(C1 + C1, [C1])) == C1
    assert constant_renumber(constantsimp(C1 + C2, [C1, C2])) == C1
    assert constant_renumber(constantsimp(C2 + C1, [C1, C2])) == C1
    assert constant_renumber(constantsimp(C1 + C2 + C1, [C1, C2])) == C1

def test_constant_power_as_base():
    if False:
        i = 10
        return i + 15
    assert constant_renumber(constantsimp(C1 ** C1, [C1])) == C1
    assert constant_renumber(constantsimp(Pow(C1, C1), [C1])) == C1
    assert constant_renumber(constantsimp(C1 ** C1, [C1])) == C1
    assert constant_renumber(constantsimp(C1 ** C2, [C1, C2])) == C1
    assert constant_renumber(constantsimp(C2 ** C1, [C1, C2])) == C1
    assert constant_renumber(constantsimp(C2 ** C2, [C1, C2])) == C1
    assert constant_renumber(constantsimp(C1 ** y, [C1, y])) == C1
    assert constant_renumber(constantsimp(C1 ** x, [C1])) == C1 ** x
    assert constant_renumber(constantsimp(C1 ** 2, [C1])) == C1
    assert constant_renumber(constantsimp(C1 ** (x * y), [C1])) == C1 ** (x * y)

def test_constant_power_as_exp():
    if False:
        print('Hello World!')
    assert constant_renumber(constantsimp(x ** C1, [C1])) == x ** C1
    assert constant_renumber(constantsimp(y ** C1, [C1, y])) == C1
    assert constant_renumber(constantsimp(x ** y ** C1, [C1, y])) == x ** C1
    assert constant_renumber(constantsimp((x ** y) ** C1, [C1])) == (x ** y) ** C1
    assert constant_renumber(constantsimp(x ** y ** C1, [C1, y])) == x ** C1
    assert constant_renumber(constantsimp(x ** C1 ** y, [C1, y])) == x ** C1
    assert constant_renumber(constantsimp(x ** C1 ** y, [C1, y])) == x ** C1
    assert constant_renumber(constantsimp((x ** C1) ** y, [C1])) == (x ** C1) ** y
    assert constant_renumber(constantsimp(2 ** C1, [C1])) == C1
    assert constant_renumber(constantsimp(S(2) ** C1, [C1])) == C1
    assert constant_renumber(constantsimp(exp(C1), [C1])) == C1
    assert constant_renumber(constantsimp(exp(C1 + x), [C1])) == C1 * exp(x)
    assert constant_renumber(constantsimp(Pow(2, C1), [C1])) == C1

def test_constant_function():
    if False:
        while True:
            i = 10
    assert constant_renumber(constantsimp(sin(C1), [C1])) == C1
    assert constant_renumber(constantsimp(f(C1), [C1])) == C1
    assert constant_renumber(constantsimp(f(C1, C1), [C1])) == C1
    assert constant_renumber(constantsimp(f(C1, C2), [C1, C2])) == C1
    assert constant_renumber(constantsimp(f(C2, C1), [C1, C2])) == C1
    assert constant_renumber(constantsimp(f(C2, C2), [C1, C2])) == C1
    assert constant_renumber(constantsimp(f(C1, x), [C1])) == f(C1, x)
    assert constant_renumber(constantsimp(f(C1, y), [C1, y])) == C1
    assert constant_renumber(constantsimp(f(y, C1), [C1, y])) == C1
    assert constant_renumber(constantsimp(f(C1, y, C2), [C1, C2, y])) == C1

def test_constant_function_multiple():
    if False:
        return 10
    assert constant_renumber(constantsimp(f(C1, C1, x), [C1])) == f(C1, C1, x)

def test_constant_multiple():
    if False:
        return 10
    assert constant_renumber(constantsimp(C1 * 2 + 2, [C1])) == C1
    assert constant_renumber(constantsimp(x * 2 / C1, [C1])) == C1 * x
    assert constant_renumber(constantsimp(C1 ** 2 * 2 + 2, [C1])) == C1
    assert constant_renumber(constantsimp(sin(2 * C1) + x + sqrt(2), [C1])) == C1 + x
    assert constant_renumber(constantsimp(2 * C1 + C2, [C1, C2])) == C1

def test_constant_repeated():
    if False:
        return 10
    assert C1 + C1 * x == constant_renumber(C1 + C1 * x)

def test_ode_solutions():
    if False:
        for i in range(10):
            print('nop')
    assert constant_renumber(constantsimp(C1 * exp(2 * x) + exp(x) * (C2 + C3), [C1, C2, C3])) == constant_renumber(C1 * exp(x) + C2 * exp(2 * x))
    assert constant_renumber(constantsimp(Eq(f(x), I * C1 * sinh(x / 3) + C2 * cosh(x / 3)), [C1, C2])) == constant_renumber(Eq(f(x), C1 * sinh(x / 3) + C2 * cosh(x / 3)))
    assert constant_renumber(constantsimp(Eq(f(x), acos(-C1 / cos(x))), [C1])) == Eq(f(x), acos(C1 / cos(x)))
    assert constant_renumber(constantsimp(Eq(log(f(x) / C1) + 2 * exp(x / f(x)), 0), [C1])) == Eq(log(C1 * f(x)) + 2 * exp(x / f(x)), 0)
    assert constant_renumber(constantsimp(Eq(log(x * sqrt(2) * sqrt(1 / x) * sqrt(f(x)) / C1) + x ** 2 / (2 * f(x) ** 2), 0), [C1])) == Eq(log(C1 * sqrt(x) * sqrt(f(x))) + x ** 2 / (2 * f(x) ** 2), 0)
    assert constant_renumber(constantsimp(Eq(-exp(-f(x) / x) * sin(f(x) / x) / 2 + log(x / C1) - cos(f(x) / x) * exp(-f(x) / x) / 2, 0), [C1])) == Eq(-exp(-f(x) / x) * sin(f(x) / x) / 2 + log(C1 * x) - cos(f(x) / x) * exp(-f(x) / x) / 2, 0)
    assert constant_renumber(constantsimp(Eq(-Integral(-1 / (sqrt(1 - u2 ** 2) * u2), (u2, _a, x / f(x))) + log(f(x) / C1), 0), [C1])) == Eq(-Integral(-1 / (u2 * sqrt(1 - u2 ** 2)), (u2, _a, x / f(x))) + log(C1 * f(x)), 0)
    assert [constantsimp(i, [C1]) for i in [Eq(f(x), sqrt(-C1 * x + x ** 2)), Eq(f(x), -sqrt(-C1 * x + x ** 2))]] == [Eq(f(x), sqrt(x * (C1 + x))), Eq(f(x), -sqrt(x * (C1 + x)))]

@XFAIL
def test_nonlocal_simplification():
    if False:
        while True:
            i = 10
    assert constantsimp(C1 + C2 + x * C2, [C1, C2]) == C1 + C2 * x

def test_constant_Eq():
    if False:
        print('Hello World!')
    assert constantsimp(Eq(C1, 3 + f(x) * x), [C1]) == Eq(x * f(x), C1)
    assert constantsimp(Eq(C1, 3 * f(x) * x), [C1]) == Eq(f(x) * x, C1)