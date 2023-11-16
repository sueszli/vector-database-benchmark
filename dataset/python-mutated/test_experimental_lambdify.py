from sympy.core.symbol import symbols, Symbol
from sympy.functions import Max
from sympy.plotting.experimental_lambdify import experimental_lambdify
from sympy.plotting.intervalmath.interval_arithmetic import interval, intervalMembership

def test_experimental_lambify():
    if False:
        for i in range(10):
            print('nop')
    x = Symbol('x')
    f = experimental_lambdify([x], Max(x, 5))
    assert Max(2, 5) == 5
    assert Max(5, 7) == 7
    x = Symbol('x-3')
    f = experimental_lambdify([x], x + 1)
    assert f(1) == 2

def test_composite_boolean_region():
    if False:
        while True:
            i = 10
    (x, y) = symbols('x y')
    r1 = (x - 1) ** 2 + y ** 2 < 2
    r2 = (x + 1) ** 2 + y ** 2 < 2
    f = experimental_lambdify((x, y), r1 & r2)
    a = (interval(-0.1, 0.1), interval(-0.1, 0.1))
    assert f(*a) == intervalMembership(True, True)
    a = (interval(-1.1, -0.9), interval(-0.1, 0.1))
    assert f(*a) == intervalMembership(False, True)
    a = (interval(0.9, 1.1), interval(-0.1, 0.1))
    assert f(*a) == intervalMembership(False, True)
    a = (interval(-0.1, 0.1), interval(1.9, 2.1))
    assert f(*a) == intervalMembership(False, True)
    f = experimental_lambdify((x, y), r1 | r2)
    a = (interval(-0.1, 0.1), interval(-0.1, 0.1))
    assert f(*a) == intervalMembership(True, True)
    a = (interval(-1.1, -0.9), interval(-0.1, 0.1))
    assert f(*a) == intervalMembership(True, True)
    a = (interval(0.9, 1.1), interval(-0.1, 0.1))
    assert f(*a) == intervalMembership(True, True)
    a = (interval(-0.1, 0.1), interval(1.9, 2.1))
    assert f(*a) == intervalMembership(False, True)
    f = experimental_lambdify((x, y), r1 & ~r2)
    a = (interval(-0.1, 0.1), interval(-0.1, 0.1))
    assert f(*a) == intervalMembership(False, True)
    a = (interval(-1.1, -0.9), interval(-0.1, 0.1))
    assert f(*a) == intervalMembership(False, True)
    a = (interval(0.9, 1.1), interval(-0.1, 0.1))
    assert f(*a) == intervalMembership(True, True)
    a = (interval(-0.1, 0.1), interval(1.9, 2.1))
    assert f(*a) == intervalMembership(False, True)
    f = experimental_lambdify((x, y), ~r1 & r2)
    a = (interval(-0.1, 0.1), interval(-0.1, 0.1))
    assert f(*a) == intervalMembership(False, True)
    a = (interval(-1.1, -0.9), interval(-0.1, 0.1))
    assert f(*a) == intervalMembership(True, True)
    a = (interval(0.9, 1.1), interval(-0.1, 0.1))
    assert f(*a) == intervalMembership(False, True)
    a = (interval(-0.1, 0.1), interval(1.9, 2.1))
    assert f(*a) == intervalMembership(False, True)
    f = experimental_lambdify((x, y), ~r1 & ~r2)
    a = (interval(-0.1, 0.1), interval(-0.1, 0.1))
    assert f(*a) == intervalMembership(False, True)
    a = (interval(-1.1, -0.9), interval(-0.1, 0.1))
    assert f(*a) == intervalMembership(False, True)
    a = (interval(0.9, 1.1), interval(-0.1, 0.1))
    assert f(*a) == intervalMembership(False, True)
    a = (interval(-0.1, 0.1), interval(1.9, 2.1))
    assert f(*a) == intervalMembership(True, True)