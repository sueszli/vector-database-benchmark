import math
import warnings
import numpy as np
import pytest
from hypothesis.internal.compat import ceil, floor

@pytest.mark.parametrize('value', ['2**64+1', '2**64-1', '2**63+1', '2**53+1', '-2**53-1', '-2**63+1', '-2**63-1', '-2**64+1', '-2**64-1'])
def test_our_floor_and_ceil_avoid_numpy_rounding(value):
    if False:
        i = 10
        return i + 15
    a = np.array([eval(value)])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        f = floor(a)
        c = ceil(a)
    assert type(f) == int
    assert type(c) == int
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        assert math.floor(a) > a or math.ceil(a) < a
    assert f <= a <= c
    assert f + 1 > a > c - 1