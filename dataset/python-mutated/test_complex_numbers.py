import pytest
from hypothesis import given, settings, strategies as st

@pytest.mark.parametrize('width', [32, 64, 128])
@pytest.mark.parametrize('keyword', ['min_magnitude', 'max_magnitude'])
@given(data=st.data())
@settings(max_examples=1000)
def test_magnitude_validates(width, keyword, data):
    if False:
        i = 10
        return i + 15
    component_width = width / 2
    magnitude = data.draw(st.floats(0, width=component_width) | st.just(1.8), label=keyword)
    strat = st.complex_numbers(width=width, **{keyword: magnitude})
    data.draw(strat)