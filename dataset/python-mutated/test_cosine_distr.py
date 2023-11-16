import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.special._ufuncs import _cosine_cdf, _cosine_invcdf
_coscdf_exact = [(-4.0, 0.0), (0, 0.5), (np.pi, 1.0), (4.0, 1.0)]

@pytest.mark.parametrize('x, expected', _coscdf_exact)
def test_cosine_cdf_exact(x, expected):
    if False:
        return 10
    assert _cosine_cdf(x) == expected
_coscdf_close = [(3.1409, 0.999999999991185), (2.25, 0.9819328173287907), (-1.599, 0.08641959838382553), (-1.601, 0.086110582992713), (-2.0, 0.0369709335961611), (-3.0, 7.522387241801384e-05), (-3.1415, 2.109869685443648e-14), (-3.14159, 4.956444476505336e-19), (-np.pi, 4.871934450264861e-50)]

@pytest.mark.parametrize('x, expected', _coscdf_close)
def test_cosine_cdf(x, expected):
    if False:
        i = 10
        return i + 15
    assert_allclose(_cosine_cdf(x), expected, rtol=5e-15)
_cosinvcdf_exact = [(0.0, -np.pi), (0.5, 0.0), (1.0, np.pi)]

@pytest.mark.parametrize('p, expected', _cosinvcdf_exact)
def test_cosine_invcdf_exact(p, expected):
    if False:
        return 10
    assert _cosine_invcdf(p) == expected

def test_cosine_invcdf_invalid_p():
    if False:
        print('Hello World!')
    assert np.isnan(_cosine_invcdf([-0.1, 1.1])).all()
_cosinvcdf_close = [(1e-50, -np.pi), (1e-14, -3.1415204137058454), (1e-08, -3.1343686589124524), (0.0018001, -2.732563923138336), (0.01, -2.41276589008678), (0.06, -1.7881244975330157), (0.125, -1.3752523669869274), (0.25, -0.831711193579736), (0.4, -0.3167954512395289), (0.419, -0.25586025626919906), (0.421, -0.24947570750445663), (0.75, 0.831711193579736), (0.94, 1.7881244975330153), (0.9999999996, 3.1391220839917167)]

@pytest.mark.parametrize('p, expected', _cosinvcdf_close)
def test_cosine_invcdf(p, expected):
    if False:
        for i in range(10):
            print('nop')
    assert_allclose(_cosine_invcdf(p), expected, rtol=1e-14)