import pytest
from hypothesis import given, strategies as st
from hypothesis.strategies import integers, lists

@pytest.mark.parametrize(('specifier', 'condition'), [(integers(), lambda x: x > 1), (lists(integers()), bool)])
def test_filter_correctly(specifier, condition):
    if False:
        i = 10
        return i + 15

    @given(specifier.filter(condition))
    def test_is_filtered(x):
        if False:
            return 10
        assert condition(x)
    test_is_filtered()
one_to_twenty_strategies = [st.integers(1, 20), st.integers(0, 19).map(lambda x: x + 1), st.sampled_from(range(1, 21)), st.sampled_from(range(20)).map(lambda x: x + 1)]

@pytest.mark.parametrize('base', one_to_twenty_strategies)
@given(data=st.data(), forbidden_values=st.lists(st.integers(1, 20), max_size=19, unique=True))
def test_chained_filters_agree(data, forbidden_values, base):
    if False:
        while True:
            i = 10

    def forbid(s, forbidden):
        if False:
            print('Hello World!')
        'Helper function to avoid Python variable scoping issues.'
        return s.filter(lambda x: x != forbidden)
    s = base
    for forbidden in forbidden_values:
        s = forbid(s, forbidden)
    x = data.draw(s)
    assert 1 <= x <= 20
    assert x not in forbidden_values

@pytest.mark.parametrize('base', one_to_twenty_strategies)
def test_chained_filters_repr(base):
    if False:
        while True:
            i = 10

    def foo(x):
        if False:
            print('Hello World!')
        return x != 0

    def bar(x):
        if False:
            while True:
                i = 10
        return x != 2
    filtered = base.filter(foo)
    chained = filtered.filter(bar)
    assert repr(chained) == f'{base!r}.filter(foo).filter(bar)'
    assert repr(filtered) == f'{base!r}.filter(foo)'