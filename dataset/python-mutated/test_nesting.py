from pytest import raises
from hypothesis import Verbosity, given, settings, strategies as st
from tests.common.utils import no_shrink

def test_nesting_1():
    if False:
        while True:
            i = 10

    @given(st.integers(0, 100))
    @settings(max_examples=5, database=None, deadline=None)
    def test_blah(x):
        if False:
            i = 10
            return i + 15

        @given(st.integers())
        @settings(max_examples=100, phases=no_shrink, database=None, verbosity=Verbosity.quiet)
        def test_nest(y):
            if False:
                print('Hello World!')
            if y >= x:
                raise ValueError
        with raises(ValueError):
            test_nest()
    test_blah()