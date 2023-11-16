from pytest import raises
from hypothesis import Verbosity, core, find, given, settings, strategies as st
from tests.common.utils import no_shrink

def test_seeds_off_internal_random():
    if False:
        while True:
            i = 10
    s = settings(phases=no_shrink, database=None)
    r = core._hypothesis_global_random.getstate()
    x = find(st.integers(), lambda x: True, settings=s)
    core._hypothesis_global_random.setstate(r)
    y = find(st.integers(), lambda x: True, settings=s)
    assert x == y

def test_nesting_with_control_passes_health_check():
    if False:
        print('Hello World!')

    @given(st.integers(0, 100), st.random_module())
    @settings(max_examples=5, database=None, deadline=None)
    def test_blah(x, rnd):
        if False:
            for i in range(10):
                print('nop')

        @given(st.integers())
        @settings(max_examples=100, phases=no_shrink, database=None, verbosity=Verbosity.quiet)
        def test_nest(y):
            if False:
                print('Hello World!')
            assert y < x
        with raises(AssertionError):
            test_nest()
    test_blah()