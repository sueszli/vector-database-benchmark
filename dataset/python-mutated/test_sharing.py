from hypothesis import given, strategies as st
from tests.common.debug import find_any, minimal
x = st.shared(st.integers())

@given(x, x)
def test_sharing_is_by_instance_by_default(a, b):
    if False:
        for i in range(10):
            print('nop')
    assert a == b

@given(st.shared(st.integers(), key='hi'), st.shared(st.integers(), key='hi'))
def test_different_instances_with_the_same_key_are_shared(a, b):
    if False:
        for i in range(10):
            print('nop')
    assert a == b

def test_different_instances_are_not_shared():
    if False:
        for i in range(10):
            print('nop')
    find_any(st.tuples(st.shared(st.integers()), st.shared(st.integers())), lambda x: x[0] != x[1])

def test_different_keys_are_not_shared():
    if False:
        return 10
    find_any(st.tuples(st.shared(st.integers(), key=1), st.shared(st.integers(), key=2)), lambda x: x[0] != x[1])

def test_keys_and_default_are_not_shared():
    if False:
        print('Hello World!')
    find_any(st.tuples(st.shared(st.integers(), key=1), st.shared(st.integers())), lambda x: x[0] != x[1])

def test_can_simplify_shared_lists():
    if False:
        for i in range(10):
            print('nop')
    xs = minimal(st.lists(st.shared(st.integers())), lambda x: len(x) >= 10 and x[0] != 0)
    assert xs == [1] * 10

def test_simplify_shared_linked_to_size():
    if False:
        return 10
    xs = minimal(st.lists(st.shared(st.integers())), lambda t: sum(t) >= 1000)
    assert sum(xs[:-1]) < 1000
    assert (xs[0] - 1) * len(xs) < 1000