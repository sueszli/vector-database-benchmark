import pytest
from hypothesis import Phase, settings
from hypothesis.errors import Unsatisfiable
from hypothesis.strategies import lists
from tests.common import standard_types
from tests.common.debug import minimal
from tests.common.utils import flaky

@pytest.mark.parametrize('spec', standard_types, ids=list(map(repr, standard_types)))
@flaky(min_passes=1, max_runs=2)
def test_can_collectively_minimize(spec):
    if False:
        i = 10
        return i + 15
    "This should generally exercise strategies' strictly_simpler heuristic by\n    putting us in a state where example cloning is required to get to the\n    answer fast enough."
    n = 10
    try:
        xs = minimal(lists(spec, min_size=n, max_size=n), lambda x: len(set(map(repr, x))) >= 2, settings(max_examples=2000, phases=(Phase.generate, Phase.shrink)))
        assert len(xs) == n
        assert 2 <= len(set(map(repr, xs))) <= 3
    except Unsatisfiable:
        pass