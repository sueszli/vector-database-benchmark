import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal
from scipy import stats
from statsmodels.stats._lilliefors import lilliefors, get_lilliefors_table, kstest_fit, ksstat

class TestLilliefors:

    def test_normal(self):
        if False:
            print('Hello World!')
        np.random.seed(3975)
        x_n = stats.norm.rvs(size=500)
        (d_ks_norm, p_norm) = lilliefors(x_n, dist='norm', pvalmethod='approx')
        (d_ks_exp, p_exp) = lilliefors(x_n + np.abs(x_n.min()) + 0.001, dist='exp', pvalmethod='approx')
        assert_almost_equal(d_ks_norm, 0.025957, decimal=3)
        assert_almost_equal(p_norm, 0.64175, decimal=3)
        assert_almost_equal(d_ks_exp, 0.3436007, decimal=3)
        assert_almost_equal(p_exp, 0.001, decimal=3)

    def test_normal_table(self):
        if False:
            return 10
        np.random.seed(3975)
        x_n = stats.norm.rvs(size=500)
        (d_ks_norm, p_norm) = lilliefors(x_n, dist='norm', pvalmethod='table')
        assert_almost_equal(d_ks_norm, 0.025957, decimal=3)
        assert_almost_equal(p_norm, 0.64175, decimal=3)

    def test_expon(self):
        if False:
            i = 10
            return i + 15
        np.random.seed(3975)
        x_e = stats.expon.rvs(size=500)
        (d_ks_norm, p_norm) = lilliefors(x_e, dist='norm', pvalmethod='approx')
        (d_ks_exp, p_exp) = lilliefors(x_e, dist='exp', pvalmethod='approx')
        assert_almost_equal(d_ks_norm, 0.15581, decimal=3)
        assert_almost_equal(p_norm, 2.2e-16, decimal=3)
        assert_almost_equal(d_ks_exp, 0.02763748, decimal=3)
        assert_almost_equal(p_exp, 0.7254, decimal=3)

    def test_pval_bounds(self):
        if False:
            print('Hello World!')
        x = stats.norm.ppf((np.arange(10.0) + 0.5) / 10)
        (d_ks_n, p_n) = lilliefors(x, dist='norm', pvalmethod='approx')
        x = stats.expon.ppf((np.arange(10.0) + 0.5) / 10)
        (d_ks_e, p_e) = lilliefors(x, dist='exp', pvalmethod='approx')
        assert_almost_equal(p_n, 0.99, decimal=7)
        assert_almost_equal(p_e, 0.99, decimal=7)

    def test_min_nobs(self):
        if False:
            for i in range(10):
                print('nop')
        x = np.arange(3.0)
        with pytest.raises(ValueError):
            lilliefors(x, dist='norm', pvalmethod='approx')
        x = np.arange(2.0)
        with pytest.raises(ValueError):
            lilliefors(x, dist='exp', pvalmethod='approx')

    @pytest.mark.smoke
    def test_large_sample(self, reset_randomstate):
        if False:
            return 10
        x = np.random.randn(10000)
        lilliefors(x, pvalmethod='approx')

    def test_x_dims(self):
        if False:
            i = 10
            return i + 15
        np.random.seed(3975)
        x_n = stats.norm.rvs(size=500)
        data = x_n
        (d_ks_norm, p_norm) = lilliefors(data, dist='norm', pvalmethod='approx')
        assert_almost_equal(d_ks_norm, 0.025957, decimal=3)
        assert_almost_equal(p_norm, 0.64175, decimal=3)
        data = x_n.reshape(-1, 1)
        (d_ks_norm, p_norm) = lilliefors(data, dist='norm', pvalmethod='approx')
        assert_almost_equal(d_ks_norm, 0.025957, decimal=3)
        assert_almost_equal(p_norm, 0.64175, decimal=3)
        data = np.array([x_n, x_n]).T
        with pytest.raises(ValueError):
            lilliefors(data, dist='norm', pvalmethod='approx')
        data = pd.DataFrame(data=x_n)
        (d_ks_norm, p_norm) = lilliefors(data, dist='norm', pvalmethod='approx')
        assert_almost_equal(d_ks_norm, 0.025957, decimal=3)
        assert_almost_equal(p_norm, 0.64175, decimal=3)
        data = pd.DataFrame(data=[x_n, x_n])
        with pytest.raises(ValueError):
            lilliefors(data, dist='norm', pvalmethod='approx')
        data = pd.DataFrame(data=x_n.reshape(-1, 1).T)
        with pytest.raises(ValueError):
            lilliefors(data, dist='norm', pvalmethod='approx')

def test_get_lilliefors_errors(reset_randomstate):
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(ValueError):
        get_lilliefors_table(dist='unknown')
    with pytest.raises(ValueError):
        kstest_fit(np.random.standard_normal(100), dist='unknown', pvalmethod='table')

def test_ksstat(reset_randomstate):
    if False:
        print('Hello World!')
    x = np.random.uniform(0, 1, 100)
    two_sided = ksstat(x, 'uniform', alternative='two_sided')
    greater = ksstat(x, 'uniform', alternative='greater')
    lower = ksstat(x, stats.uniform, alternative='lower')
    print(two_sided, greater, lower)
    assert lower <= two_sided
    assert greater <= two_sided