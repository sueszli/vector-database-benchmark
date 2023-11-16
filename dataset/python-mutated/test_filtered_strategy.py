import hypothesis.strategies as st
from hypothesis.control import BuildContext
from hypothesis.internal.conjecture.data import ConjectureData
from hypothesis.strategies._internal.strategies import FilteredStrategy

def test_filter_iterations_are_marked_as_discarded():
    if False:
        for i in range(10):
            print('nop')
    variable_equal_to_zero = 0
    x = st.integers(0, 255).filter(lambda x: x == variable_equal_to_zero)
    data = ConjectureData.for_buffer([0, 2, 1, 0])
    with BuildContext(data):
        assert data.draw(x) == 0
    assert data.has_discards

def test_filtered_branches_are_all_filtered():
    if False:
        while True:
            i = 10
    s = FilteredStrategy(st.integers() | st.text(), (bool,))
    assert all((isinstance(x, FilteredStrategy) for x in s.branches))

def test_filter_conditions_may_be_empty():
    if False:
        for i in range(10):
            print('nop')
    s = FilteredStrategy(st.integers(), conditions=())
    s.condition(0)

def test_nested_filteredstrategy_flattens_conditions():
    if False:
        i = 10
        return i + 15
    s = FilteredStrategy(FilteredStrategy(st.text(), conditions=(bool,)), conditions=(len,))
    assert s.filtered_strategy is st.text()
    assert s.flat_conditions == (bool, len)