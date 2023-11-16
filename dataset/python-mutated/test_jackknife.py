import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal
from astropy.stats.jackknife import jackknife_resampling, jackknife_stats
from astropy.utils.compat.optional_deps import HAS_SCIPY

def test_jackknife_resampling():
    if False:
        while True:
            i = 10
    data = np.array([1, 2, 3, 4])
    answer = np.array([[2, 3, 4], [1, 3, 4], [1, 2, 4], [1, 2, 3]])
    assert_equal(answer, jackknife_resampling(data))

@pytest.mark.skipif(not HAS_SCIPY, reason='requires scipy')
def test_jackknife_stats():
    if False:
        i = 10
        return i + 15
    data = np.array((115, 170, 142, 138, 280, 470, 480, 141, 390))
    answer = (258.4444, 0.0, 50.25936)
    assert_allclose(answer, jackknife_stats(data, np.mean)[0:3], atol=0.0001)

@pytest.mark.skipif(not HAS_SCIPY, reason='requires scipy')
def test_jackknife_stats_conf_interval():
    if False:
        for i in range(10):
            print('nop')
    data = np.array([48, 42, 36, 33, 20, 16, 29, 39, 42, 38, 42, 36, 20, 15, 42, 33, 22, 20, 41, 43, 45, 34, 14, 22, 6, 7, 0, 15, 33, 34, 28, 29, 34, 41, 4, 13, 32, 38, 24, 25, 47, 27, 41, 41, 24, 28, 26, 14, 30, 28, 41, 40])
    data = np.reshape(data, (-1, 2))
    data = data[:, 1]
    answer = (113.7862, -4.376391, 22.26572)

    def mle_var(x):
        if False:
            return 10
        return np.sum((x - np.mean(x)) * (x - np.mean(x))) / len(x)
    assert_allclose(answer, jackknife_stats(data, mle_var, 0.95)[0:3], atol=0.0001)
    answer = np.array((70.14615, 157.42616))
    assert_allclose(answer, jackknife_stats(data, mle_var, 0.95)[3], atol=0.0001)

def test_jackknife_stats_exceptions():
    if False:
        while True:
            i = 10
    with pytest.raises(ValueError):
        jackknife_stats(np.arange(2), np.mean, confidence_level=42)