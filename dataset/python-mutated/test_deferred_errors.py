import pytest
from hypothesis import find, given, strategies as st
from hypothesis.errors import InvalidArgument
from hypothesis.strategies._internal.core import defines_strategy

def test_does_not_error_on_initial_calculation():
    if False:
        return 10
    st.floats(max_value=float('nan'))
    st.sampled_from([])
    st.lists(st.integers(), min_size=5, max_size=2)
    st.floats(min_value=2.0, max_value=1.0)

def test_errors_each_time():
    if False:
        return 10
    s = st.integers(max_value=1, min_value=3)
    with pytest.raises(InvalidArgument):
        s.example()
    with pytest.raises(InvalidArgument):
        s.example()

def test_errors_on_test_invocation():
    if False:
        i = 10
        return i + 15

    @given(st.integers(max_value=1, min_value=3))
    def test(x):
        if False:
            while True:
                i = 10
        pass
    with pytest.raises(InvalidArgument):
        test()

def test_errors_on_find():
    if False:
        for i in range(10):
            print('nop')
    s = st.lists(st.integers(), min_size=5, max_size=2)
    with pytest.raises(InvalidArgument):
        find(s, lambda x: True)

def test_errors_on_example():
    if False:
        i = 10
        return i + 15
    s = st.floats(min_value=2.0, max_value=1.0)
    with pytest.raises(InvalidArgument):
        s.example()

def test_does_not_recalculate_the_strategy():
    if False:
        while True:
            i = 10
    calls = [0]

    @defines_strategy()
    def foo():
        if False:
            print('Hello World!')
        calls[0] += 1
        return st.just(1)
    f = foo()
    assert calls == [0]
    f.example()
    assert calls == [1]
    f.example()
    assert calls == [1]