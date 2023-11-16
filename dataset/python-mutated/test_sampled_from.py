import enum
import pytest
from hypothesis import given, strategies as st
from hypothesis.errors import InvalidArgument
from tests.common.utils import counts_calls, fails_with

@pytest.mark.parametrize('n', [100, 10 ** 5, 10 ** 6, 2 ** 25])
def test_filter_large_lists(n):
    if False:
        print('Hello World!')
    filter_limit = 100 * 10000

    @counts_calls
    def cond(x):
        if False:
            return 10
        assert cond.calls < filter_limit
        return x % 2 != 0
    s = st.sampled_from(range(n)).filter(cond)

    @given(s)
    def run(x):
        if False:
            i = 10
            return i + 15
        assert x % 2 != 0
    run()
    assert cond.calls < filter_limit

def rare_value_strategy(n, target):
    if False:
        while True:
            i = 10

    def forbid(s, forbidden):
        if False:
            for i in range(10):
                print('nop')
        'Helper function to avoid Python variable scoping issues.'
        return s.filter(lambda x: x != forbidden)
    s = st.sampled_from(range(n))
    for i in range(n):
        if i != target:
            s = forbid(s, i)
    return s

@given(rare_value_strategy(n=128, target=80))
def test_chained_filters_find_rare_value(x):
    if False:
        while True:
            i = 10
    assert x == 80

@fails_with(InvalidArgument)
@given(st.sets(st.sampled_from(range(10)), min_size=11))
def test_unsat_sets_of_samples(x):
    if False:
        i = 10
        return i + 15
    raise AssertionError

@given(st.sets(st.sampled_from(range(50)), min_size=50))
def test_efficient_sets_of_samples(x):
    if False:
        print('Hello World!')
    assert x == set(range(50))

class AnEnum(enum.Enum):
    a = enum.auto()
    b = enum.auto()

def test_enum_repr_uses_class_not_a_list():
    if False:
        for i in range(10):
            print('nop')
    lazy_repr = repr(st.sampled_from(AnEnum))
    assert lazy_repr == 'sampled_from(tests.nocover.test_sampled_from.AnEnum)'

class AFlag(enum.Flag):
    a = enum.auto()
    b = enum.auto()

def test_flag_enum_repr_uses_class_not_a_list():
    if False:
        print('Hello World!')
    lazy_repr = repr(st.sampled_from(AFlag))
    assert lazy_repr == 'sampled_from(tests.nocover.test_sampled_from.AFlag)'