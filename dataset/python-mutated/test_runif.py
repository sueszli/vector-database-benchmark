import pytest
from tests_fabric.helpers.runif import RunIf

@RunIf(min_torch='99')
def test_always_skip():
    if False:
        i = 10
        return i + 15
    exit(1)

@pytest.mark.parametrize('arg1', [0.5, 1.0, 2.0])
@RunIf(min_torch='0.0')
def test_wrapper(arg1: float):
    if False:
        while True:
            i = 10
    assert arg1 > 0.0