from hypothesis import given, strategies as st
from hypothesis.strategies._internal.types import _global_type_lookup
TYPES = sorted((x for x in _global_type_lookup if x.__module__ != 'typing' and x.__name__ != 'ByteString'), key=str)

def everything_except(excluded_types):
    if False:
        while True:
            i = 10
    'Recipe copied from the docstring of ``from_type``'
    return st.from_type(type).flatmap(st.from_type).filter(lambda x: not isinstance(x, excluded_types))

@given(excluded_types=st.lists(st.sampled_from(TYPES), min_size=1, max_size=3, unique=True).map(tuple), data=st.data())
def test_recipe_for_everything_except(excluded_types, data):
    if False:
        print('Hello World!')
    value = data.draw(everything_except(excluded_types))
    assert not isinstance(value, excluded_types)