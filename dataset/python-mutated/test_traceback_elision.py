import traceback
import pytest
from hypothesis import Verbosity, given, settings, strategies as st

@pytest.mark.parametrize('verbosity', [Verbosity.normal, Verbosity.debug])
def test_tracebacks_omit_hypothesis_internals(verbosity):
    if False:
        print('Hello World!')

    @settings(verbosity=verbosity)
    @given(st.just(False))
    def simplest_failure(x):
        if False:
            i = 10
            return i + 15
        raise ValueError
    try:
        simplest_failure()
    except ValueError as e:
        tb = traceback.extract_tb(e.__traceback__)
        if verbosity < Verbosity.debug:
            assert len(tb) == 4
        else:
            assert len(tb) >= 5