import pytest
from dpcontracts import require
from hypothesis import given
from hypothesis.errors import InvalidArgument
from hypothesis.extra.dpcontracts import fulfill
from hypothesis.strategies import builds, integers

def identity(x):
    if False:
        print('Hello World!')
    return x

@require('division is undefined for zero', lambda args: args.n != 0)
def invert(n):
    if False:
        while True:
            i = 10
    return 1 / n

@given(builds(fulfill(invert), integers()))
def test_contract_filter_builds(x):
    if False:
        for i in range(10):
            print('nop')
    assert -1 <= x <= 1

@given(integers())
def test_contract_filter_inline(n):
    if False:
        i = 10
        return i + 15
    assert -1 <= fulfill(invert)(n) <= 1

@pytest.mark.parametrize('f', [int, identity, lambda x: None])
def test_no_vacuous_fulfill(f):
    if False:
        return 10
    with pytest.raises(InvalidArgument):
        fulfill(f)