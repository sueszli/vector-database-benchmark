from hypothesis import given, settings
from hypothesis.strategies import floats, integers, sets
from tests.common.debug import find_any

def test_can_draw_sets_of_hard_to_find_elements():
    if False:
        print('Hello World!')
    rarebool = floats(0, 1).map(lambda x: x <= 0.05)
    find_any(sets(rarebool, min_size=2), settings=settings(deadline=None))

@given(sets(integers(), max_size=0))
def test_empty_sets(x):
    if False:
        return 10
    assert x == set()

@given(sets(integers(), max_size=2))
def test_bounded_size_sets(x):
    if False:
        return 10
    assert len(x) <= 2