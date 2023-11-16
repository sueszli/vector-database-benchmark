import pytest
from hypothesis import HealthCheck, Verbosity, given, reject, settings, strategies as st
from hypothesis.errors import Unsatisfiable
from tests.common.debug import minimal
from tests.common.utils import no_shrink

@pytest.mark.parametrize('strat', [st.text(min_size=5)])
@settings(phases=no_shrink, deadline=None, suppress_health_check=list(HealthCheck))
@given(st.data())
def test_explore_arbitrary_function(strat, data):
    if False:
        while True:
            i = 10
    cache = {}

    def predicate(x):
        if False:
            i = 10
            return i + 15
        try:
            return cache[x]
        except KeyError:
            return cache.setdefault(x, data.draw(st.booleans(), label=repr(x)))
    try:
        minimal(strat, predicate, settings=settings(max_examples=10, database=None, verbosity=Verbosity.quiet))
    except Unsatisfiable:
        reject()