import pytest
from hypothesis import HealthCheck, given, reject, settings
from hypothesis.errors import Unsatisfiable
from hypothesis.strategies import integers

def test_raises_unsatisfiable_if_all_false():
    if False:
        return 10

    @given(integers())
    @settings(max_examples=50, suppress_health_check=list(HealthCheck))
    def test_assume_false(x):
        if False:
            print('Hello World!')
        reject()
    with pytest.raises(Unsatisfiable):
        test_assume_false()