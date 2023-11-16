import math
import sys
from random import Random
import pytest
from hypothesis.internal.conjecture.shrinking import Integer, Lexical, Ordering

def measure_baseline(cls, value, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    shrinker = cls(value, lambda x: x == value, random=Random(0), **kwargs)
    shrinker.run()
    return shrinker.calls

@pytest.mark.parametrize('cls', [Lexical, Ordering])
@pytest.mark.parametrize('example', [[255] * 8])
def test_meets_budgetary_requirements(cls, example):
    if False:
        while True:
            i = 10
    n = len(example)
    budget = n * math.ceil(math.log(n, 2)) + 5
    assert measure_baseline(cls, example) <= budget

def test_integer_shrinking_is_parsimonious():
    if False:
        for i in range(10):
            print('nop')
    assert measure_baseline(Integer, int(sys.float_info.max)) <= 10