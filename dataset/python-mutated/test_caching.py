import pytest
from hypothesis import strategies as st
from hypothesis.errors import InvalidArgument

def test_no_args():
    if False:
        i = 10
        return i + 15
    assert st.text() is st.text()

def test_tuple_lengths():
    if False:
        for i in range(10):
            print('nop')
    assert st.tuples(st.integers()) is st.tuples(st.integers())
    assert st.tuples(st.integers()) is not st.tuples(st.integers(), st.integers())

def test_values():
    if False:
        for i in range(10):
            print('nop')
    assert st.integers() is not st.integers(min_value=1)

def test_alphabet_key():
    if False:
        i = 10
        return i + 15
    assert st.text(alphabet='abcs') is st.text(alphabet='abcs')

def test_does_not_error_on_unhashable_posarg():
    if False:
        return 10
    st.text(['a', 'b', 'c'])

def test_does_not_error_on_unhashable_kwarg():
    if False:
        i = 10
        return i + 15
    with pytest.raises(InvalidArgument):
        st.builds(lambda alphabet: 1, alphabet=['a', 'b', 'c']).validate()

def test_caches_floats_sensitively():
    if False:
        return 10
    assert st.floats(min_value=0.0) is st.floats(min_value=0.0)
    assert st.floats(min_value=0.0) is not st.floats(min_value=0)
    assert st.floats(min_value=0.0) is not st.floats(min_value=-0.0)