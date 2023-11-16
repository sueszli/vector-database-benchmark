import pytest
from hypothesis import given, strategies as st

@st.composite
def strat(draw, x=0, /):
    if False:
        for i in range(10):
            print('nop')
    return draw(st.integers(min_value=x))

@given(st.data(), st.integers())
def test_composite_with_posonly_args(data, min_value):
    if False:
        return 10
    v = data.draw(strat(min_value))
    assert min_value <= v

def test_preserves_signature():
    if False:
        while True:
            i = 10
    with pytest.raises(TypeError):
        strat(x=1)

def test_builds_real_pos_only():
    if False:
        return 10
    with pytest.raises(TypeError):
        st.builds()