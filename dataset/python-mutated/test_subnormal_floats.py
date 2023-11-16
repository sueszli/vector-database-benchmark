import math
from sys import float_info
import pytest
from hypothesis.internal.floats import width_smallest_normals
from hypothesis.strategies import floats
from tests.common.debug import assert_all_examples, find_any
from tests.common.utils import PYTHON_FTZ

def test_python_compiled_with_sane_math_options():
    if False:
        for i in range(10):
            print('nop')
    "Python does not flush-to-zero, which violates IEEE-754\n\n    The other tests that rely on subnormals are skipped when Python is FTZ\n    (otherwise pytest will be very noisy), so this meta test ensures CI jobs\n    still fail as we currently don't care to support such builds of Python.\n    "
    assert not PYTHON_FTZ
skipif_ftz = pytest.mark.skipif(PYTHON_FTZ, reason='broken by unsafe compiler flags')

@skipif_ftz
def test_can_generate_subnormals():
    if False:
        i = 10
        return i + 15
    find_any(floats().filter(lambda x: x > 0), lambda x: x < float_info.min)
    find_any(floats().filter(lambda x: x < 0), lambda x: x > -float_info.min)

@skipif_ftz
@pytest.mark.parametrize('min_value, max_value', [(None, None), (-1, 0), (0, 1), (-1, 1)])
@pytest.mark.parametrize('width', [16, 32, 64])
def test_does_not_generate_subnormals_when_disallowed(width, min_value, max_value):
    if False:
        return 10
    strat = floats(min_value=min_value, max_value=max_value, allow_subnormal=False, width=width)
    strat = strat.filter(lambda x: x != 0.0 and math.isfinite(x))
    smallest_normal = width_smallest_normals[width]
    assert_all_examples(strat, lambda x: x <= -smallest_normal or x >= smallest_normal)