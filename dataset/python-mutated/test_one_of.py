import re
import pytest
from hypothesis import given, strategies as st
from hypothesis.errors import InvalidArgument
from tests.common.debug import assert_no_examples

def test_one_of_empty():
    if False:
        while True:
            i = 10
    e = st.one_of()
    assert e.is_empty
    assert_no_examples(e)

@given(st.one_of(st.integers().filter(bool)))
def test_one_of_filtered(i):
    if False:
        i = 10
        return i + 15
    assert bool(i)

@given(st.one_of(st.just(100).flatmap(st.integers)))
def test_one_of_flatmapped(i):
    if False:
        i = 10
        return i + 15
    assert i >= 100

def test_one_of_single_strategy_is_noop():
    if False:
        print('Hello World!')
    s = st.integers()
    assert st.one_of(s) is s
    assert st.one_of([s]) is s

def test_one_of_without_strategies_suggests_sampled_from():
    if False:
        i = 10
        return i + 15
    with pytest.raises(InvalidArgument, match=re.escape('Did you mean st.sampled_from([1, 2, 3])?')):
        st.one_of(1, 2, 3)