"""
rename this to test_assumptions.py when the old assumptions system is deleted
"""
from sympy.abc import x, y
from sympy.assumptions.assume import global_assumptions
from sympy.assumptions.ask import Q
from sympy.printing import pretty

def test_equal():
    if False:
        i = 10
        return i + 15
    'Test for equality'
    assert Q.positive(x) == Q.positive(x)
    assert Q.positive(x) != ~Q.positive(x)
    assert ~Q.positive(x) == ~Q.positive(x)

def test_pretty():
    if False:
        for i in range(10):
            print('nop')
    assert pretty(Q.positive(x)) == 'Q.positive(x)'
    assert pretty({Q.positive, Q.integer}) == '{Q.integer, Q.positive}'

def test_global():
    if False:
        while True:
            i = 10
    'Test for global assumptions'
    global_assumptions.add(x > 0)
    assert (x > 0) in global_assumptions
    global_assumptions.remove(x > 0)
    assert not (x > 0) in global_assumptions
    global_assumptions.add(x > 0, y > 0)
    assert (x > 0) in global_assumptions
    assert (y > 0) in global_assumptions
    global_assumptions.clear()
    assert not (x > 0) in global_assumptions
    assert not (y > 0) in global_assumptions