from hypothesis import assume, given, strategies as st
from hypothesis.strategies._internal.lazy import unwrap_strategies
from tests.common.debug import assert_no_examples

@given(st.integers().map(lambda x: assume(x % 3 != 0) and x))
def test_can_assume_in_map(x):
    if False:
        i = 10
        return i + 15
    assert x % 3 != 0

def test_assume_in_just_raises_immediately():
    if False:
        return 10
    assert_no_examples(st.just(1).map(lambda x: assume(x == 2)))

def test_identity_map_is_noop():
    if False:
        print('Hello World!')
    s = unwrap_strategies(st.integers())
    assert s.map(lambda x: x) is s