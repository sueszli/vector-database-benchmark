import pytest
from hypothesis import given, strategies as st
from hypothesis.errors import HypothesisWarning

def test_error_is_in_finally():
    if False:
        print('Hello World!')

    @given(st.data())
    def test(d):
        if False:
            for i in range(10):
                print('nop')
        try:
            d.draw(st.lists(st.integers(), min_size=3, unique=True))
        finally:
            raise ValueError
    with pytest.raises(ValueError) as err:
        test()
    assert '[0, 1, -1]' in '\n'.join(err.value.__notes__)

@given(st.data())
def test_warns_on_bool_strategy(data):
    if False:
        while True:
            i = 10
    with pytest.warns(HypothesisWarning, match='bool\\(.+\\) is always True, did you mean to draw a value\\?'):
        if st.booleans():
            pass