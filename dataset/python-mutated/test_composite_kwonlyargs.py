from hypothesis import given, strategies as st

@st.composite
def kwonlyargs_composites(draw, *, kwarg1=None):
    if False:
        print('Hello World!')
    return draw(st.fixed_dictionaries({'kwarg1': st.just(kwarg1), 'i': st.integers()}))

@given(st.lists(st.one_of(kwonlyargs_composites(kwarg1='test')), unique_by=lambda x: x['i']))
def test_composite_with_keyword_only_args(a):
    if False:
        for i in range(10):
            print('nop')
    pass