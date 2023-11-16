from sympy.core.numbers import Rational, oo, pi
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.miscellaneous import root, sqrt
from sympy.functions.elementary.trigonometric import asin, cos, sin, tan
from sympy.polys.rationaltools import together
from sympy.series.limits import limit
x = Symbol('x')

def test_leadterm():
    if False:
        print('Hello World!')
    assert (3 + 2 * x ** (log(3) / log(2) - 1)).leadterm(x) == (3, 0)

def root3(x):
    if False:
        for i in range(10):
            print('nop')
    return root(x, 3)

def root4(x):
    if False:
        i = 10
        return i + 15
    return root(x, 4)

def test_Limits_simple_0():
    if False:
        return 10
    assert limit((2 ** (x + 1) + 3 ** (x + 1)) / (2 ** x + 3 ** x), x, oo) == 3

def test_Limits_simple_1():
    if False:
        i = 10
        return i + 15
    assert limit((x + 1) * (x + 2) * (x + 3) / x ** 3, x, oo) == 1
    assert limit(sqrt(x + 1) - sqrt(x), x, oo) == 0
    assert limit((2 * x - 3) * (3 * x + 5) * (4 * x - 6) / (3 * x ** 3 + x - 1), x, oo) == 8
    assert limit(x / root3(x ** 3 + 10), x, oo) == 1
    assert limit((x + 1) ** 2 / (x ** 2 + 1), x, oo) == 1

def test_Limits_simple_2():
    if False:
        i = 10
        return i + 15
    assert limit(1000 * x / (x ** 2 - 1), x, oo) == 0
    assert limit((x ** 2 - 5 * x + 1) / (3 * x + 7), x, oo) is oo
    assert limit((2 * x ** 2 - x + 3) / (x ** 3 - 8 * x + 5), x, oo) == 0
    assert limit((2 * x ** 2 - 3 * x - 4) / sqrt(x ** 4 + 1), x, oo) == 2
    assert limit((2 * x + 3) / (x + root3(x)), x, oo) == 2
    assert limit(x ** 2 / (10 + x * sqrt(x)), x, oo) is oo
    assert limit(root3(x ** 2 + 1) / (x + 1), x, oo) == 0
    assert limit(sqrt(x) / sqrt(x + sqrt(x + sqrt(x))), x, oo) == 1

def test_Limits_simple_3a():
    if False:
        while True:
            i = 10
    a = Symbol('a')
    assert together(limit((x ** 2 - (a + 1) * x + a) / (x ** 3 - a ** 3), x, a)) == (a - 1) / (3 * a ** 2)

def test_Limits_simple_3b():
    if False:
        while True:
            i = 10
    h = Symbol('h')
    assert limit(((x + h) ** 3 - x ** 3) / h, h, 0) == 3 * x ** 2
    assert limit(1 / (1 - x) - 3 / (1 - x ** 3), x, 1) == -1
    assert limit((sqrt(1 + x) - 1) / (root3(1 + x) - 1), x, 0) == Rational(3) / 2
    assert limit((sqrt(x) - 1) / (x - 1), x, 1) == Rational(1) / 2
    assert limit((sqrt(x) - 8) / (root3(x) - 4), x, 64) == 3
    assert limit((root3(x) - 1) / (root4(x) - 1), x, 1) == Rational(4) / 3
    assert limit((root3(x ** 2) - 2 * root3(x) + 1) / (x - 1) ** 2, x, 1) == Rational(1) / 9

def test_Limits_simple_4a():
    if False:
        while True:
            i = 10
    a = Symbol('a')
    assert limit((sqrt(x) - sqrt(a)) / (x - a), x, a) == 1 / (2 * sqrt(a))
    assert limit((sqrt(x) - 1) / (root3(x) - 1), x, 1) == Rational(3, 2)
    assert limit((sqrt(1 + x) - sqrt(1 - x)) / x, x, 0) == 1
    assert limit(sqrt(x ** 2 - 5 * x + 6) - x, x, oo) == Rational(-5, 2)

def test_limits_simple_4aa():
    if False:
        return 10
    assert limit(x * (sqrt(x ** 2 + 1) - x), x, oo) == Rational(1) / 2

def test_Limits_simple_4b():
    if False:
        while True:
            i = 10
    assert limit(x - root3(x ** 3 - 1), x, oo) == 0

def test_Limits_simple_4c():
    if False:
        for i in range(10):
            print('nop')
    assert limit(log(1 + exp(x)) / x, x, -oo) == 0
    assert limit(log(1 + exp(x)) / x, x, oo) == 1

def test_bounded():
    if False:
        return 10
    assert limit(sin(x) / x, x, oo) == 0
    assert limit(x * sin(1 / x), x, 0) == 0

def test_f1a():
    if False:
        while True:
            i = 10
    assert limit((sin(2 * x) / x) ** (1 + x), x, 0) == 2

def test_f1a2():
    if False:
        for i in range(10):
            print('nop')
    assert limit(((x - 1) / (x + 1)) ** x, x, oo) == exp(-2)

def test_f1b():
    if False:
        return 10
    m = Symbol('m')
    n = Symbol('n')
    h = Symbol('h')
    a = Symbol('a')
    assert limit(sin(x) / x, x, 2) == sin(2) / 2
    assert limit(sin(3 * x) / x, x, 0) == 3
    assert limit(sin(5 * x) / sin(2 * x), x, 0) == Rational(5, 2)
    assert limit(sin(pi * x) / sin(3 * pi * x), x, 0) == Rational(1, 3)
    assert limit(x * sin(pi / x), x, oo) == pi
    assert limit((1 - cos(x)) / x ** 2, x, 0) == S.Half
    assert limit(x * sin(1 / x), x, oo) == 1
    assert limit((cos(m * x) - cos(n * x)) / x ** 2, x, 0) == -m ** 2 / 2 + n ** 2 / 2
    assert limit((tan(x) - sin(x)) / x ** 3, x, 0) == S.Half
    assert limit((x - sin(2 * x)) / (x + sin(3 * x)), x, 0) == -Rational(1, 4)
    assert limit((1 - sqrt(cos(x))) / x ** 2, x, 0) == Rational(1, 4)
    assert limit((sqrt(1 + sin(x)) - sqrt(1 - sin(x))) / x, x, 0) == 1
    assert limit((1 + h / x) ** x, x, oo) == exp(h)
    assert limit((sin(x) - sin(a)) / (x - a), x, a) == cos(a)
    assert limit((cos(x) - cos(a)) / (x - a), x, a) == -sin(a)
    assert limit((sin(x + h) - sin(x)) / h, h, 0) == cos(x)

def test_f2a():
    if False:
        while True:
            i = 10
    assert limit(((x + 1) / (2 * x + 1)) ** x ** 2, x, oo) == 0

def test_f2():
    if False:
        return 10
    assert limit((sqrt(cos(x)) - root3(cos(x))) / sin(x) ** 2, x, 0) == -Rational(1, 12)

def test_f3():
    if False:
        return 10
    a = Symbol('a')
    assert limit(asin(a * x) / x, x, 0) == a