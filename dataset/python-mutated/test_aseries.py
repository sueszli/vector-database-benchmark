from sympy.core.function import PoleError
from sympy.core.numbers import oo
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.series.order import O
from sympy.abc import x
from sympy.testing.pytest import raises

def test_simple():
    if False:
        for i in range(10):
            print('nop')
    e = sin(1 / x + exp(-x)) - sin(1 / x)
    assert e.aseries(x) == (1 / (24 * x ** 4) - 1 / (2 * x ** 2) + 1 + O(x ** (-6), (x, oo))) * exp(-x)
    e = exp(x) * (exp(1 / x + exp(-x)) - exp(1 / x))
    assert e.aseries(x, n=4) == 1 / (6 * x ** 3) + 1 / (2 * x ** 2) + 1 / x + 1 + O(x ** (-4), (x, oo))
    e = exp(exp(x) / (1 - 1 / x))
    assert e.aseries(x) == exp(exp(x) / (1 - 1 / x))
    e = exp(sin(1 / x + exp(-exp(x)))) - exp(sin(1 / x))
    assert e.aseries(x, n=4) == (-1 / (2 * x ** 3) + 1 / x + 1 + O(x ** (-4), (x, oo))) * exp(-exp(x))
    e3 = lambda x: exp(exp(exp(x)))
    e = e3(x) / e3(x - 1 / e3(x))
    assert e.aseries(x, n=3) == 1 + exp(x + exp(x)) * exp(-exp(exp(x))) + ((-exp(x) / 2 - S.Half) * exp(x + exp(x)) + exp(2 * x + 2 * exp(x)) / 2) * exp(-2 * exp(exp(x))) + O(exp(-3 * exp(exp(x))), (x, oo))
    e = exp(exp(x)) * (exp(sin(1 / x + 1 / exp(exp(x)))) - exp(sin(1 / x)))
    assert e.aseries(x, n=4) == -1 / (2 * x ** 3) + 1 / x + 1 + O(x ** (-4), (x, oo))
    n = Symbol('n', integer=True)
    e = sqrt(n) * log(n) ** 2 * exp(sqrt(log(n)) * log(log(n)) ** 2 * exp(sqrt(log(log(n))) * log(log(log(n))) ** 3)) / n
    assert e.aseries(n) == exp(exp(sqrt(log(log(n))) * log(log(log(n))) ** 3) * sqrt(log(n)) * log(log(n)) ** 2) * log(n) ** 2 / sqrt(n)

def test_hierarchical():
    if False:
        print('Hello World!')
    e = sin(1 / x + exp(-x))
    assert e.aseries(x, n=3, hir=True) == -exp(-2 * x) * sin(1 / x) / 2 + exp(-x) * cos(1 / x) + sin(1 / x) + O(exp(-3 * x), (x, oo))
    e = sin(x) * cos(exp(-x))
    assert e.aseries(x, hir=True) == exp(-4 * x) * sin(x) / 24 - exp(-2 * x) * sin(x) / 2 + sin(x) + O(exp(-6 * x), (x, oo))
    raises(PoleError, lambda : e.aseries(x))