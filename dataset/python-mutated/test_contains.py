from sympy.core.expr import unchanged
from sympy.core.numbers import oo
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.sets.contains import Contains
from sympy.sets.sets import FiniteSet, Interval
from sympy.testing.pytest import raises

def test_contains_basic():
    if False:
        i = 10
        return i + 15
    raises(TypeError, lambda : Contains(S.Integers, 1))
    assert Contains(2, S.Integers) is S.true
    assert Contains(-2, S.Naturals) is S.false
    i = Symbol('i', integer=True)
    assert Contains(i, S.Naturals) == Contains(i, S.Naturals, evaluate=False)

def test_issue_6194():
    if False:
        i = 10
        return i + 15
    x = Symbol('x')
    assert unchanged(Contains, x, Interval(0, 1))
    assert Interval(0, 1).contains(x) == (S.Zero <= x) & (x <= 1)
    assert Contains(x, FiniteSet(0)) != S.false
    assert Contains(x, Interval(1, 1)) != S.false
    assert Contains(x, S.Integers) != S.false

def test_issue_10326():
    if False:
        for i in range(10):
            print('nop')
    assert Contains(oo, Interval(-oo, oo)) == False
    assert Contains(-oo, Interval(-oo, oo)) == False

def test_binary_symbols():
    if False:
        return 10
    x = Symbol('x')
    y = Symbol('y')
    z = Symbol('z')
    assert Contains(x, FiniteSet(y, Eq(z, True))).binary_symbols == {y, z}

def test_as_set():
    if False:
        for i in range(10):
            print('nop')
    x = Symbol('x')
    y = Symbol('y')
    assert Contains(x, FiniteSet(y)).as_set() == FiniteSet(y)
    assert Contains(x, S.Integers).as_set() == S.Integers
    assert Contains(x, S.Reals).as_set() == S.Reals

def test_type_error():
    if False:
        for i in range(10):
            print('nop')
    raises(TypeError, lambda : Contains(2, None))