import gc
import weakref
import pytest
from hypothesis import given, settings, strategies as st

@pytest.mark.parametrize('s', [st.floats(), st.tuples(st.integers()), st.tuples(), st.one_of(st.integers(), st.text())])
def test_is_cacheable(s):
    if False:
        print('Hello World!')
    assert s.is_cacheable

@pytest.mark.parametrize('s', [st.just([]), st.tuples(st.integers(), st.just([])), st.one_of(st.integers(), st.text(), st.just([]))])
def test_is_not_cacheable(s):
    if False:
        return 10
    assert not s.is_cacheable

def test_non_cacheable_things_are_not_cached():
    if False:
        return 10
    x = st.just([])
    assert st.tuples(x) != st.tuples(x)

def test_cacheable_things_are_cached():
    if False:
        return 10
    x = st.just(())
    assert st.tuples(x) == st.tuples(x)

def test_local_types_are_garbage_collected_issue_493():
    if False:
        i = 10
        return i + 15
    store = [None]

    def run_locally():
        if False:
            i = 10
            return i + 15

        class Test:

            @settings(database=None)
            @given(st.integers())
            def test(self, i):
                if False:
                    return 10
                pass
        store[0] = weakref.ref(Test)
        Test().test()
    run_locally()
    del run_locally
    assert store[0]() is not None
    gc.collect()
    assert store[0]() is None