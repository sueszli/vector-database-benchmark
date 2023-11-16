import pytest
from hypothesis import given, strategies as st
from hypothesis.errors import InvalidArgument
from tests.common.debug import assert_no_examples, minimal

def test_resampling():
    if False:
        for i in range(10):
            print('nop')
    x = minimal(st.lists(st.integers(), min_size=1).flatmap(lambda x: st.lists(st.sampled_from(x))), lambda x: len(x) >= 10 and len(set(x)) == 1)
    assert x == [0] * 10

@given(st.lists(st.nothing()))
def test_list_of_nothing(xs):
    if False:
        i = 10
        return i + 15
    assert xs == []

@given(st.sets(st.nothing()))
def test_set_of_nothing(xs):
    if False:
        print('Hello World!')
    assert xs == set()

def test_validates_min_size():
    if False:
        return 10
    with pytest.raises(InvalidArgument):
        st.lists(st.nothing(), min_size=1).validate()

def test_function_composition():
    if False:
        i = 10
        return i + 15
    assert st.nothing().map(lambda x: 'hi').is_empty
    assert st.nothing().filter(lambda x: True).is_empty
    assert st.nothing().flatmap(lambda x: st.integers()).is_empty

def test_tuples_detect_empty_elements():
    if False:
        while True:
            i = 10
    assert st.tuples(st.nothing()).is_empty

def test_fixed_dictionaries_detect_empty_values():
    if False:
        print('Hello World!')
    assert st.fixed_dictionaries({'a': st.nothing()}).is_empty

def test_no_examples():
    if False:
        i = 10
        return i + 15
    assert_no_examples(st.nothing())

@pytest.mark.parametrize('s', [st.nothing(), st.nothing().map(lambda x: x), st.nothing().filter(lambda x: True), st.nothing().flatmap(lambda x: st.integers())])
def test_empty(s):
    if False:
        while True:
            i = 10
    assert s.is_empty