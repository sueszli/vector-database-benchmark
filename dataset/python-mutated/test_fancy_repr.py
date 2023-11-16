from hypothesis import strategies as st

def test_floats_is_floats():
    if False:
        return 10
    assert repr(st.floats()) == 'floats()'

def test_includes_non_default_values():
    if False:
        return 10
    assert repr(st.floats(max_value=1.0)) == 'floats(max_value=1.0)'

def foo(*args, **kwargs):
    if False:
        while True:
            i = 10
    pass

def test_builds_repr():
    if False:
        for i in range(10):
            print('nop')
    assert repr(st.builds(foo, st.just(1), x=st.just(10))) == 'builds(foo, just(1), x=just(10))'

def test_map_repr():
    if False:
        i = 10
        return i + 15
    assert repr(st.integers().map(abs)) == 'integers().map(abs)'
    assert repr(st.integers().map(lambda x: x * 2)) == 'integers().map(lambda x: x * 2)'

def test_filter_repr():
    if False:
        print('Hello World!')
    assert repr(st.integers().filter(lambda x: x != 3)) == 'integers().filter(lambda x: x != 3)'

def test_flatmap_repr():
    if False:
        i = 10
        return i + 15
    assert repr(st.integers().flatmap(lambda x: st.booleans())) == 'integers().flatmap(lambda x: st.booleans())'