import pytest
from hypothesis import assume, given, strategies as st
from hypothesis.internal.conjecture.junkdrawer import IntList
non_neg_lists = st.lists(st.integers(min_value=0, max_value=2 ** 63 - 1))

@given(non_neg_lists)
def test_intlist_is_equal_to_itself(ls):
    if False:
        for i in range(10):
            print('nop')
    assert IntList(ls) == IntList(ls)

@given(non_neg_lists, non_neg_lists)
def test_distinct_int_lists_are_not_equal(x, y):
    if False:
        return 10
    assume(x != y)
    assert IntList(x) != IntList(y)

def test_basic_equality():
    if False:
        print('Hello World!')
    x = IntList([1, 2, 3])
    assert x == x
    t = x != x
    assert not t
    assert x != 'foo'
    s = x == 'foo'
    assert not s

def test_error_on_invalid_value():
    if False:
        i = 10
        return i + 15
    with pytest.raises(ValueError):
        IntList([-1])

def test_extend_by_too_large():
    if False:
        while True:
            i = 10
    x = IntList()
    ls = [1, 10 ** 6]
    x.extend(ls)
    assert list(x) == ls