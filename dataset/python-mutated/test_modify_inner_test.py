from functools import wraps
import pytest
from hypothesis import given, strategies as st
from hypothesis.errors import InvalidArgument

def always_passes(*args, **kwargs):
    if False:
        print('Hello World!')
    'Stand-in for a fixed version of an inner test.\n\n    For example, pytest-trio would take the inner test, wrap it in an\n    async-to-sync converter, and use the new func (not always_passes).\n    '

@given(st.integers())
def test_can_replace_inner_test(x):
    if False:
        for i in range(10):
            print('nop')
    raise AssertionError('This should be replaced')
test_can_replace_inner_test.hypothesis.inner_test = always_passes

def decorator(func):
    if False:
        while True:
            i = 10
    'An example of a common decorator pattern.'

    @wraps(func)
    def inner(*args, **kwargs):
        if False:
            return 10
        return func(*args, **kwargs)
    return inner

@decorator
@given(st.integers())
def test_can_replace_when_decorated(x):
    if False:
        return 10
    raise AssertionError('This should be replaced')
test_can_replace_when_decorated.hypothesis.inner_test = always_passes

@pytest.mark.parametrize('x', [1, 2])
@given(y=st.integers())
def test_can_replace_when_parametrized(x, y):
    if False:
        print('Hello World!')
    raise AssertionError('This should be replaced')
test_can_replace_when_parametrized.hypothesis.inner_test = always_passes

def test_can_replace_when_original_is_invalid():
    if False:
        for i in range(10):
            print('nop')

    @given(st.integers(), st.integers())
    def invalid_test(x):
        if False:
            while True:
                i = 10
        raise AssertionError
    invalid_test.hypothesis.inner_test = always_passes
    with pytest.raises(InvalidArgument, match='Too many positional arguments'):
        invalid_test()

def test_inner_is_original_even_when_invalid():
    if False:
        while True:
            i = 10

    def invalid_test(x):
        if False:
            print('Hello World!')
        raise AssertionError
    original = invalid_test
    invalid_test = given()(invalid_test)
    with pytest.raises(InvalidArgument, match='given must be called with at least one argument'):
        invalid_test()
    assert invalid_test.hypothesis.inner_test == original

def test_invokes_inner_function_with_args_by_name():
    if False:
        return 10

    @given(st.integers())
    def test(x):
        if False:
            return 10
        pass
    f = test.hypothesis.inner_test
    test.hypothesis.inner_test = wraps(f)(lambda **kw: f(**kw))
    test()