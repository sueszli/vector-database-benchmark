import numpy as np
import pytest
from sklearn.utils._testing import assert_allclose
from sklearn.utils.arrayfuncs import min_pos

def test_min_pos():
    if False:
        while True:
            i = 10
    X = np.random.RandomState(0).randn(100)
    min_double = min_pos(X)
    min_float = min_pos(X.astype(np.float32))
    assert_allclose(min_double, min_float)
    assert min_double >= 0

@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_min_pos_no_positive(dtype):
    if False:
        print('Hello World!')
    X = np.full(100, -1.0).astype(dtype, copy=False)
    assert min_pos(X) == np.finfo(dtype).max