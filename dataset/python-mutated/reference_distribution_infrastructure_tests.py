import numpy as np
import pytest
from scipy import stats
from numpy.testing import assert_allclose
import scipy.stats.tests.test_generation.reference_distributions as rd
import mpmath
from mpmath import mp

def test_basic():
    if False:
        print('Hello World!')
    message = '`mpmath.mp.dps <= 15`. Set a higher precision...'
    with pytest.raises(RuntimeError, match=message):
        rd.Normal()
    mpmath.dps = 20
    message = '`mpmath.dps` has been assigned. This is not intended usage...'
    with pytest.raises(RuntimeError, match=message):
        rd.Normal()
    del mpmath.dps
    mp.dps = 20
    rng = np.random.default_rng(6716188855217730280)
    x = rng.random(size=3)
    a = rng.random(size=(2, 1))
    rtol = 1e-15
    dist = rd.SkewNormal(a=a)
    dist_ref = stats.skewnorm(a)
    assert_allclose(dist.pdf(x), dist_ref.pdf(x), rtol=rtol)
    assert_allclose(dist.cdf(x), dist_ref.cdf(x), rtol=rtol)
    assert_allclose(dist.sf(x), dist_ref.sf(x), rtol=rtol)
    assert_allclose(dist.ppf(x), dist_ref.ppf(x), rtol=rtol)
    assert_allclose(dist.isf(x), dist_ref.isf(x), rtol=rtol)
    assert_allclose(dist.logpdf(x), dist_ref.logpdf(x), rtol=rtol)
    assert_allclose(dist.logcdf(x), dist_ref.logcdf(x), rtol=rtol)
    assert_allclose(dist.logsf(x), dist_ref.logsf(x), rtol=rtol)
    assert_allclose(dist.support(), dist_ref.support(), rtol=rtol)
    assert_allclose(dist.entropy(), dist_ref.entropy(), rtol=rtol)
    assert_allclose(dist.mean(), dist_ref.mean(), rtol=rtol)
    assert_allclose(dist.var(), dist_ref.var(), rtol=rtol)
    assert_allclose(dist.skew(), dist_ref.stats('s'), rtol=rtol)
    assert_allclose(dist.kurtosis(), dist_ref.stats('k'), rtol=rtol)

def test_complementary_method_use():
    if False:
        i = 10
        return i + 15
    mp.dps = 50
    x = np.linspace(0, 1, 10)

    class MyDist(rd.ReferenceDistribution):

        def _cdf(self, x):
            if False:
                print('Hello World!')
            return x
    dist = MyDist()
    assert_allclose(dist.sf(x), 1 - dist.cdf(x))

    class MyDist(rd.ReferenceDistribution):

        def _sf(self, x):
            if False:
                i = 10
                return i + 15
            return 1 - x
    dist = MyDist()
    assert_allclose(dist.cdf(x), 1 - dist.sf(x))

    class MyDist(rd.ReferenceDistribution):

        def _ppf(self, x, guess):
            if False:
                i = 10
                return i + 15
            return x
    dist = MyDist()
    assert_allclose(dist.isf(x), dist.ppf(1 - x))

    class MyDist(rd.ReferenceDistribution):

        def _isf(self, x, guess):
            if False:
                print('Hello World!')
            return 1 - x
    dist = MyDist()
    assert_allclose(dist.ppf(x), dist.isf(1 - x))