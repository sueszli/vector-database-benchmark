from itertools import product
import numpy as np
import random
import functools
import pytest
from numpy.testing import assert_, assert_equal, assert_allclose, assert_almost_equal
from pytest import raises as assert_raises
import scipy.stats as stats
from scipy.stats import distributions
from scipy.stats._hypotests import epps_singleton_2samp, cramervonmises, _cdf_cvm, cramervonmises_2samp, _pval_cvm_2samp_exact, barnard_exact, boschloo_exact
from scipy.stats._mannwhitneyu import mannwhitneyu, _mwu_state
from .common_tests import check_named_results
from scipy._lib._testutils import _TestPythranFunc

class TestEppsSingleton:

    def test_statistic_1(self):
        if False:
            for i in range(10):
                print('nop')
        x = np.array([-0.35, 2.55, 1.73, 0.73, 0.35, 2.69, 0.46, -0.94, -0.37, 12.07])
        y = np.array([-1.15, -0.15, 2.48, 3.25, 3.71, 4.29, 5.0, 7.74, 8.38, 8.6])
        (w, p) = epps_singleton_2samp(x, y)
        assert_almost_equal(w, 15.14, decimal=1)
        assert_almost_equal(p, 0.00442, decimal=3)

    def test_statistic_2(self):
        if False:
            print('Hello World!')
        x = np.array((0, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 5, 5, 5, 5, 6, 10, 10, 10, 10))
        y = np.array((10, 4, 0, 5, 10, 10, 0, 5, 6, 7, 10, 3, 1, 7, 0, 8, 1, 5, 8, 10))
        (w, p) = epps_singleton_2samp(x, y)
        assert_allclose(w, 8.9, atol=0.001)
        assert_almost_equal(p, 0.06364, decimal=3)

    def test_epps_singleton_array_like(self):
        if False:
            print('Hello World!')
        np.random.seed(1234)
        (x, y) = (np.arange(30), np.arange(28))
        (w1, p1) = epps_singleton_2samp(list(x), list(y))
        (w2, p2) = epps_singleton_2samp(tuple(x), tuple(y))
        (w3, p3) = epps_singleton_2samp(x, y)
        assert_(w1 == w2 == w3)
        assert_(p1 == p2 == p3)

    def test_epps_singleton_size(self):
        if False:
            while True:
                i = 10
        (x, y) = ((1, 2, 3, 4), np.arange(10))
        assert_raises(ValueError, epps_singleton_2samp, x, y)

    def test_epps_singleton_nonfinite(self):
        if False:
            i = 10
            return i + 15
        (x, y) = ((1, 2, 3, 4, 5, np.inf), np.arange(10))
        assert_raises(ValueError, epps_singleton_2samp, x, y)
        (x, y) = (np.arange(10), (1, 2, 3, 4, 5, np.nan))
        assert_raises(ValueError, epps_singleton_2samp, x, y)

    def test_epps_singleton_1d_input(self):
        if False:
            i = 10
            return i + 15
        x = np.arange(100).reshape(-1, 1)
        assert_raises(ValueError, epps_singleton_2samp, x, x)

    def test_names(self):
        if False:
            for i in range(10):
                print('nop')
        (x, y) = (np.arange(20), np.arange(30))
        res = epps_singleton_2samp(x, y)
        attributes = ('statistic', 'pvalue')
        check_named_results(res, attributes)

class TestCvm:

    def test_cdf_4(self):
        if False:
            for i in range(10):
                print('nop')
        assert_allclose(_cdf_cvm([0.02983, 0.04111, 0.12331, 0.94251], 4), [0.01, 0.05, 0.5, 0.999], atol=0.0001)

    def test_cdf_10(self):
        if False:
            while True:
                i = 10
        assert_allclose(_cdf_cvm([0.02657, 0.0383, 0.12068, 0.56643], 10), [0.01, 0.05, 0.5, 0.975], atol=0.0001)

    def test_cdf_1000(self):
        if False:
            for i in range(10):
                print('nop')
        assert_allclose(_cdf_cvm([0.02481, 0.03658, 0.11889, 1.1612], 1000), [0.01, 0.05, 0.5, 0.999], atol=0.0001)

    def test_cdf_inf(self):
        if False:
            print('Hello World!')
        assert_allclose(_cdf_cvm([0.0248, 0.03656, 0.11888, 1.16204]), [0.01, 0.05, 0.5, 0.999], atol=0.0001)

    def test_cdf_support(self):
        if False:
            return 10
        assert_equal(_cdf_cvm([1 / (12 * 533), 533 / 3], 533), [0, 1])
        assert_equal(_cdf_cvm([1 / (12 * (27 + 1)), (27 + 1) / 3], 27), [0, 1])

    def test_cdf_large_n(self):
        if False:
            return 10
        assert_allclose(_cdf_cvm([0.0248, 0.03656, 0.11888, 1.16204, 100], 10000), _cdf_cvm([0.0248, 0.03656, 0.11888, 1.16204, 100]), atol=0.0001)

    def test_large_x(self):
        if False:
            print('Hello World!')
        assert_(0.99999 < _cdf_cvm(333.3, 1000) < 1.0)
        assert_(0.99999 < _cdf_cvm(333.3) < 1.0)

    def test_low_p(self):
        if False:
            i = 10
            return i + 15
        n = 12
        res = cramervonmises(np.ones(n) * 0.8, 'norm')
        assert_(_cdf_cvm(res.statistic, n) > 1.0)
        assert_equal(res.pvalue, 0)

    def test_invalid_input(self):
        if False:
            i = 10
            return i + 15
        x = np.arange(10).reshape((2, 5))
        assert_raises(ValueError, cramervonmises, x, 'norm')
        assert_raises(ValueError, cramervonmises, [1.5], 'norm')
        assert_raises(ValueError, cramervonmises, (), 'norm')

    def test_values_R(self):
        if False:
            print('Hello World!')
        res = cramervonmises([-1.7, 2, 0, 1.3, 4, 0.1, 0.6], 'norm')
        assert_allclose(res.statistic, 0.288156, atol=1e-06)
        assert_allclose(res.pvalue, 0.1453465, atol=1e-06)
        res = cramervonmises([-1.7, 2, 0, 1.3, 4, 0.1, 0.6], 'norm', (3, 1.5))
        assert_allclose(res.statistic, 0.9426685, atol=1e-06)
        assert_allclose(res.pvalue, 0.002026417, atol=1e-06)
        res = cramervonmises([1, 2, 5, 1.4, 0.14, 11, 13, 0.9, 7.5], 'expon')
        assert_allclose(res.statistic, 0.8421854, atol=1e-06)
        assert_allclose(res.pvalue, 0.004433406, atol=1e-06)

    def test_callable_cdf(self):
        if False:
            return 10
        (x, args) = (np.arange(5), (1.4, 0.7))
        r1 = cramervonmises(x, distributions.expon.cdf)
        r2 = cramervonmises(x, 'expon')
        assert_equal((r1.statistic, r1.pvalue), (r2.statistic, r2.pvalue))
        r1 = cramervonmises(x, distributions.beta.cdf, args)
        r2 = cramervonmises(x, 'beta', args)
        assert_equal((r1.statistic, r1.pvalue), (r2.statistic, r2.pvalue))

class TestMannWhitneyU:

    def setup_method(self):
        if False:
            for i in range(10):
                print('nop')
        _mwu_state._recursive = True

    def test_input_validation(self):
        if False:
            return 10
        x = np.array([1, 2])
        y = np.array([3, 4])
        with assert_raises(ValueError, match='`x` and `y` must be of nonzero'):
            mannwhitneyu([], y)
        with assert_raises(ValueError, match='`x` and `y` must be of nonzero'):
            mannwhitneyu(x, [])
        with assert_raises(ValueError, match='`use_continuity` must be one'):
            mannwhitneyu(x, y, use_continuity='ekki')
        with assert_raises(ValueError, match='`alternative` must be one of'):
            mannwhitneyu(x, y, alternative='ekki')
        with assert_raises(ValueError, match='`axis` must be an integer'):
            mannwhitneyu(x, y, axis=1.5)
        with assert_raises(ValueError, match='`method` must be one of'):
            mannwhitneyu(x, y, method='ekki')

    def test_auto(self):
        if False:
            i = 10
            return i + 15
        np.random.seed(1)
        n = 8
        x = np.random.rand(n - 1)
        y = np.random.rand(n - 1)
        auto = mannwhitneyu(x, y)
        asymptotic = mannwhitneyu(x, y, method='asymptotic')
        exact = mannwhitneyu(x, y, method='exact')
        assert auto.pvalue == exact.pvalue
        assert auto.pvalue != asymptotic.pvalue
        x = np.random.rand(n - 1)
        y = np.random.rand(n + 1)
        auto = mannwhitneyu(x, y)
        asymptotic = mannwhitneyu(x, y, method='asymptotic')
        exact = mannwhitneyu(x, y, method='exact')
        assert auto.pvalue == exact.pvalue
        assert auto.pvalue != asymptotic.pvalue
        auto = mannwhitneyu(y, x)
        asymptotic = mannwhitneyu(x, y, method='asymptotic')
        exact = mannwhitneyu(x, y, method='exact')
        assert auto.pvalue == exact.pvalue
        assert auto.pvalue != asymptotic.pvalue
        x = np.random.rand(n + 1)
        y = np.random.rand(n + 1)
        auto = mannwhitneyu(x, y)
        asymptotic = mannwhitneyu(x, y, method='asymptotic')
        exact = mannwhitneyu(x, y, method='exact')
        assert auto.pvalue != exact.pvalue
        assert auto.pvalue == asymptotic.pvalue
        x = np.random.rand(n - 1)
        y = np.random.rand(n - 1)
        y[3] = x[3]
        auto = mannwhitneyu(x, y)
        asymptotic = mannwhitneyu(x, y, method='asymptotic')
        exact = mannwhitneyu(x, y, method='exact')
        assert auto.pvalue != exact.pvalue
        assert auto.pvalue == asymptotic.pvalue
    x = [210.05211, 110.19063, 307.918612]
    y = [436.08811482466416, 416.3739732976819, 179.96975939463582, 197.8118754228619, 34.038757281225756, 138.54220550921517, 128.7769351470246, 265.9272142795185, 275.6617533155341, 592.3408339541626, 448.7317759061702, 300.61495185038905, 187.97508449019588]
    cases_basic = [[{'alternative': 'two-sided', 'method': 'asymptotic'}, (16, 0.6865041817876)], [{'alternative': 'less', 'method': 'asymptotic'}, (16, 0.3432520908938)], [{'alternative': 'greater', 'method': 'asymptotic'}, (16, 0.7047591913255)], [{'alternative': 'two-sided', 'method': 'exact'}, (16, 0.7035714285714)], [{'alternative': 'less', 'method': 'exact'}, (16, 0.3517857142857)], [{'alternative': 'greater', 'method': 'exact'}, (16, 0.6946428571429)]]

    @pytest.mark.parametrize(('kwds', 'expected'), cases_basic)
    def test_basic(self, kwds, expected):
        if False:
            print('Hello World!')
        res = mannwhitneyu(self.x, self.y, **kwds)
        assert_allclose(res, expected)
    cases_continuity = [[{'alternative': 'two-sided', 'use_continuity': True}, (23, 0.6865041817876)], [{'alternative': 'less', 'use_continuity': True}, (23, 0.7047591913255)], [{'alternative': 'greater', 'use_continuity': True}, (23, 0.3432520908938)], [{'alternative': 'two-sided', 'use_continuity': False}, (23, 0.6377328900502)], [{'alternative': 'less', 'use_continuity': False}, (23, 0.6811335549749)], [{'alternative': 'greater', 'use_continuity': False}, (23, 0.3188664450251)]]

    @pytest.mark.parametrize(('kwds', 'expected'), cases_continuity)
    def test_continuity(self, kwds, expected):
        if False:
            return 10
        res = mannwhitneyu(self.y, self.x, method='asymptotic', **kwds)
        assert_allclose(res, expected)

    def test_tie_correct(self):
        if False:
            for i in range(10):
                print('nop')
        x = [1, 2, 3, 4]
        y0 = np.array([1, 2, 3, 4, 5])
        dy = np.array([0, 1, 0, 1, 0]) * 0.01
        dy2 = np.array([0, 0, 1, 0, 0]) * 0.01
        y = [y0 - 0.01, y0 - dy, y0 - dy2, y0, y0 + dy2, y0 + dy, y0 + 0.01]
        res = mannwhitneyu(x, y, axis=-1, method='asymptotic')
        U_expected = [10, 9, 8.5, 8, 7.5, 7, 6]
        p_expected = [1, 0.9017048037317, 0.804080657472, 0.7086240584439, 0.6197963884941, 0.5368784563079, 0.3912672792826]
        assert_equal(res.statistic, U_expected)
        assert_allclose(res.pvalue, p_expected)
    pn3 = {1: [0.25, 0.5, 0.75], 2: [0.1, 0.2, 0.4, 0.6], 3: [0.05, 0.1, 0.2, 0.35, 0.5, 0.65]}
    pn4 = {1: [0.2, 0.4, 0.6], 2: [0.067, 0.133, 0.267, 0.4, 0.6], 3: [0.028, 0.057, 0.114, 0.2, 0.314, 0.429, 0.571], 4: [0.014, 0.029, 0.057, 0.1, 0.171, 0.243, 0.343, 0.443, 0.557]}
    pm5 = {1: [0.167, 0.333, 0.5, 0.667], 2: [0.047, 0.095, 0.19, 0.286, 0.429, 0.571], 3: [0.018, 0.036, 0.071, 0.125, 0.196, 0.286, 0.393, 0.5, 0.607], 4: [0.008, 0.016, 0.032, 0.056, 0.095, 0.143, 0.206, 0.278, 0.365, 0.452, 0.548], 5: [0.004, 0.008, 0.016, 0.028, 0.048, 0.075, 0.111, 0.155, 0.21, 0.274, 0.345, 0.421, 0.5, 0.579]}
    pm6 = {1: [0.143, 0.286, 0.428, 0.571], 2: [0.036, 0.071, 0.143, 0.214, 0.321, 0.429, 0.571], 3: [0.012, 0.024, 0.048, 0.083, 0.131, 0.19, 0.274, 0.357, 0.452, 0.548], 4: [0.005, 0.01, 0.019, 0.033, 0.057, 0.086, 0.129, 0.176, 0.238, 0.305, 0.381, 0.457, 0.543], 5: [0.002, 0.004, 0.009, 0.015, 0.026, 0.041, 0.063, 0.089, 0.123, 0.165, 0.214, 0.268, 0.331, 0.396, 0.465, 0.535], 6: [0.001, 0.002, 0.004, 0.008, 0.013, 0.021, 0.032, 0.047, 0.066, 0.09, 0.12, 0.155, 0.197, 0.242, 0.294, 0.35, 0.409, 0.469, 0.531]}

    def test_exact_distribution(self):
        if False:
            i = 10
            return i + 15
        p_tables = {3: self.pn3, 4: self.pn4, 5: self.pm5, 6: self.pm6}
        for (n, table) in p_tables.items():
            for (m, p) in table.items():
                u = np.arange(0, len(p))
                assert_allclose(_mwu_state.cdf(k=u, m=m, n=n), p, atol=0.001)
                u2 = np.arange(0, m * n + 1)
                assert_allclose(_mwu_state.cdf(k=u2, m=m, n=n) + _mwu_state.sf(k=u2, m=m, n=n) - _mwu_state.pmf(k=u2, m=m, n=n), 1)
                pmf = _mwu_state.pmf(k=u2, m=m, n=n)
                assert_allclose(pmf, pmf[::-1])
                pmf2 = _mwu_state.pmf(k=u2, m=n, n=m)
                assert_allclose(pmf, pmf2)

    def test_asymptotic_behavior(self):
        if False:
            return 10
        np.random.seed(0)
        x = np.random.rand(5)
        y = np.random.rand(5)
        res1 = mannwhitneyu(x, y, method='exact')
        res2 = mannwhitneyu(x, y, method='asymptotic')
        assert res1.statistic == res2.statistic
        assert np.abs(res1.pvalue - res2.pvalue) > 0.01
        x = np.random.rand(40)
        y = np.random.rand(40)
        res1 = mannwhitneyu(x, y, method='exact')
        res2 = mannwhitneyu(x, y, method='asymptotic')
        assert res1.statistic == res2.statistic
        assert np.abs(res1.pvalue - res2.pvalue) < 0.001

    def test_exact_U_equals_mean(self):
        if False:
            for i in range(10):
                print('nop')
        res_l = mannwhitneyu([1, 2, 3], [1.5, 2.5], alternative='less', method='exact')
        res_g = mannwhitneyu([1, 2, 3], [1.5, 2.5], alternative='greater', method='exact')
        assert_equal(res_l.pvalue, res_g.pvalue)
        assert res_l.pvalue > 0.5
        res = mannwhitneyu([1, 2, 3], [1.5, 2.5], alternative='two-sided', method='exact')
        assert_equal(res, (3, 1))
    cases_scalar = [[{'alternative': 'two-sided', 'method': 'asymptotic'}, (0, 1)], [{'alternative': 'less', 'method': 'asymptotic'}, (0, 0.5)], [{'alternative': 'greater', 'method': 'asymptotic'}, (0, 0.977249868052)], [{'alternative': 'two-sided', 'method': 'exact'}, (0, 1)], [{'alternative': 'less', 'method': 'exact'}, (0, 0.5)], [{'alternative': 'greater', 'method': 'exact'}, (0, 1)]]

    @pytest.mark.parametrize(('kwds', 'result'), cases_scalar)
    def test_scalar_data(self, kwds, result):
        if False:
            return 10
        assert_allclose(mannwhitneyu(1, 2, **kwds), result)

    def test_equal_scalar_data(self):
        if False:
            for i in range(10):
                print('nop')
        assert_equal(mannwhitneyu(1, 1, method='exact'), (0.5, 1))
        assert_equal(mannwhitneyu(1, 1, method='asymptotic'), (0.5, 1))
        assert_equal(mannwhitneyu(1, 1, method='asymptotic', use_continuity=False), (0.5, np.nan))

    @pytest.mark.parametrize('method', ['asymptotic', 'exact'])
    def test_gh_12837_11113(self, method):
        if False:
            print('Hello World!')
        np.random.seed(0)
        axis = -3
        (m, n) = (7, 10)
        x = np.random.rand(m, 3, 8)
        y = np.random.rand(6, n, 1, 8) + 0.1
        res = mannwhitneyu(x, y, method=method, axis=axis)
        shape = (6, 3, 8)
        assert res.pvalue.shape == shape
        assert res.statistic.shape == shape
        (x, y) = (np.moveaxis(x, axis, -1), np.moveaxis(y, axis, -1))
        x = x[None, ...]
        assert x.ndim == y.ndim
        x = np.broadcast_to(x, shape + (m,))
        y = np.broadcast_to(y, shape + (n,))
        assert x.shape[:-1] == shape
        assert y.shape[:-1] == shape
        statistics = np.zeros(shape)
        pvalues = np.zeros(shape)
        for indices in product(*[range(i) for i in shape]):
            xi = x[indices]
            yi = y[indices]
            temp = mannwhitneyu(xi, yi, method=method)
            statistics[indices] = temp.statistic
            pvalues[indices] = temp.pvalue
        np.testing.assert_equal(res.pvalue, pvalues)
        np.testing.assert_equal(res.statistic, statistics)

    def test_gh_11355(self):
        if False:
            while True:
                i = 10
        x = [1, 2, 3, 4]
        y = [3, 6, 7, 8, 9, 3, 2, 1, 4, 4, 5]
        res1 = mannwhitneyu(x, y)
        y[4] = np.inf
        res2 = mannwhitneyu(x, y)
        assert_equal(res1.statistic, res2.statistic)
        assert_equal(res1.pvalue, res2.pvalue)
        y[4] = np.nan
        res3 = mannwhitneyu(x, y)
        assert_equal(res3.statistic, np.nan)
        assert_equal(res3.pvalue, np.nan)
    cases_11355 = [([1, 2, 3, 4], [3, 6, 7, 8, np.inf, 3, 2, 1, 4, 4, 5], 10, 0.1297704873477), ([1, 2, 3, 4], [3, 6, 7, 8, np.inf, np.inf, 2, 1, 4, 4, 5], 8.5, 0.08735617507695), ([1, 2, np.inf, 4], [3, 6, 7, 8, np.inf, 3, 2, 1, 4, 4, 5], 17.5, 0.5988856695752), ([1, 2, np.inf, 4], [3, 6, 7, 8, np.inf, np.inf, 2, 1, 4, 4, 5], 16, 0.4687165824462), ([1, np.inf, np.inf, 4], [3, 6, 7, 8, np.inf, np.inf, 2, 1, 4, 4, 5], 24.5, 0.7912517950119)]

    @pytest.mark.parametrize(('x', 'y', 'statistic', 'pvalue'), cases_11355)
    def test_gh_11355b(self, x, y, statistic, pvalue):
        if False:
            while True:
                i = 10
        res = mannwhitneyu(x, y, method='asymptotic')
        assert_allclose(res.statistic, statistic, atol=1e-12)
        assert_allclose(res.pvalue, pvalue, atol=1e-12)
    cases_9184 = [[True, 'less', 'asymptotic', 0.900775348204], [True, 'greater', 'asymptotic', 0.1223118025635], [True, 'two-sided', 'asymptotic', 0.244623605127], [False, 'less', 'asymptotic', 0.8896643190401], [False, 'greater', 'asymptotic', 0.1103356809599], [False, 'two-sided', 'asymptotic', 0.2206713619198], [True, 'less', 'exact', 0.8967698967699], [True, 'greater', 'exact', 0.1272061272061], [True, 'two-sided', 'exact', 0.2544122544123]]

    @pytest.mark.parametrize(('use_continuity', 'alternative', 'method', 'pvalue_exp'), cases_9184)
    def test_gh_9184(self, use_continuity, alternative, method, pvalue_exp):
        if False:
            return 10
        statistic_exp = 35
        x = (0.8, 0.83, 1.89, 1.04, 1.45, 1.38, 1.91, 1.64, 0.73, 1.46)
        y = (1.15, 0.88, 0.9, 0.74, 1.21)
        res = mannwhitneyu(x, y, use_continuity=use_continuity, alternative=alternative, method=method)
        assert_equal(res.statistic, statistic_exp)
        assert_allclose(res.pvalue, pvalue_exp)

    def test_gh_6897(self):
        if False:
            return 10
        with assert_raises(ValueError, match='`x` and `y` must be of nonzero'):
            mannwhitneyu([], [])

    def test_gh_4067(self):
        if False:
            i = 10
            return i + 15
        a = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
        b = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
        res = mannwhitneyu(a, b)
        assert_equal(res.statistic, np.nan)
        assert_equal(res.pvalue, np.nan)
    cases_2118 = [[[1, 2, 3], [1.5, 2.5], 'greater', (3, 0.6135850036578)], [[1, 2, 3], [1.5, 2.5], 'less', (3, 0.6135850036578)], [[1, 2, 3], [1.5, 2.5], 'two-sided', (3, 1.0)], [[1, 2, 3], [2], 'greater', (1.5, 0.681324055883)], [[1, 2, 3], [2], 'less', (1.5, 0.681324055883)], [[1, 2, 3], [2], 'two-sided', (1.5, 1)], [[1, 2], [1, 2], 'greater', (2, 0.667497228949)], [[1, 2], [1, 2], 'less', (2, 0.667497228949)], [[1, 2], [1, 2], 'two-sided', (2, 1)]]

    @pytest.mark.parametrize(['x', 'y', 'alternative', 'expected'], cases_2118)
    def test_gh_2118(self, x, y, alternative, expected):
        if False:
            print('Hello World!')
        res = mannwhitneyu(x, y, use_continuity=True, alternative=alternative, method='asymptotic')
        assert_allclose(res, expected, rtol=1e-12)

    def teardown_method(self):
        if False:
            return 10
        _mwu_state._recursive = None

class TestMannWhitneyU_iterative(TestMannWhitneyU):

    def setup_method(self):
        if False:
            print('Hello World!')
        _mwu_state._recursive = False

    def teardown_method(self):
        if False:
            print('Hello World!')
        _mwu_state._recursive = None

@pytest.mark.xslow
def test_mann_whitney_u_switch():
    if False:
        print('Hello World!')
    _mwu_state._recursive = None
    _mwu_state._fmnks = -np.ones((1, 1, 1))
    rng = np.random.default_rng(9546146887652)
    x = rng.random(5)
    y = rng.random(501)
    stats.mannwhitneyu(x, y, method='exact')
    assert np.all(_mwu_state._fmnks == -1)
    y = rng.random(500)
    stats.mannwhitneyu(x, y, method='exact')
    assert not np.all(_mwu_state._fmnks == -1)

class TestSomersD(_TestPythranFunc):

    def setup_method(self):
        if False:
            while True:
                i = 10
        self.dtypes = self.ALL_INTEGER + self.ALL_FLOAT
        self.arguments = {0: (np.arange(10), self.ALL_INTEGER + self.ALL_FLOAT), 1: (np.arange(10), self.ALL_INTEGER + self.ALL_FLOAT)}
        input_array = [self.arguments[idx][0] for idx in self.arguments]
        self.partialfunc = functools.partial(stats.somersd, alternative='two-sided')
        self.expected = self.partialfunc(*input_array)

    def pythranfunc(self, *args):
        if False:
            return 10
        res = self.partialfunc(*args)
        assert_allclose(res.statistic, self.expected.statistic, atol=1e-15)
        assert_allclose(res.pvalue, self.expected.pvalue, atol=1e-15)

    def test_pythranfunc_keywords(self):
        if False:
            for i in range(10):
                print('nop')
        table = [[27, 25, 14, 7, 0], [7, 14, 18, 35, 12], [1, 3, 2, 7, 17]]
        res1 = stats.somersd(table)
        optional_args = self.get_optional_args(stats.somersd)
        res2 = stats.somersd(table, **optional_args)
        assert_allclose(res1.statistic, res2.statistic, atol=1e-15)
        assert_allclose(res1.pvalue, res2.pvalue, atol=1e-15)

    def test_like_kendalltau(self):
        if False:
            for i in range(10):
                print('nop')
        x = [5, 2, 1, 3, 6, 4, 7, 8]
        y = [5, 2, 6, 3, 1, 8, 7, 4]
        expected = (0.0, 1.0)
        res = stats.somersd(x, y)
        assert_allclose(res.statistic, expected[0], atol=1e-15)
        assert_allclose(res.pvalue, expected[1], atol=1e-15)
        x = [0, 5, 2, 1, 3, 6, 4, 7, 8]
        y = [5, 2, 0, 6, 3, 1, 8, 7, 4]
        expected = (0.0, 1.0)
        res = stats.somersd(x, y)
        assert_allclose(res.statistic, expected[0], atol=1e-15)
        assert_allclose(res.pvalue, expected[1], atol=1e-15)
        x = [5, 2, 1, 3, 6, 4, 7]
        y = [5, 2, 6, 3, 1, 7, 4]
        expected = (-0.14285714285714, 0.63032695315767)
        res = stats.somersd(x, y)
        assert_allclose(res.statistic, expected[0], atol=1e-15)
        assert_allclose(res.pvalue, expected[1], atol=1e-15)
        x = np.arange(10)
        y = np.arange(10)
        expected = (1.0, 0)
        res = stats.somersd(x, y)
        assert_allclose(res.statistic, expected[0], atol=1e-15)
        assert_allclose(res.pvalue, expected[1], atol=1e-15)
        x = np.arange(10)
        y = np.array([0, 2, 1, 3, 4, 6, 5, 7, 8, 9])
        expected = (0.91111111111111, 0.0)
        res = stats.somersd(x, y)
        assert_allclose(res.statistic, expected[0], atol=1e-15)
        assert_allclose(res.pvalue, expected[1], atol=1e-15)
        x = np.arange(10)
        y = np.arange(10)[::-1]
        expected = (-1.0, 0)
        res = stats.somersd(x, y)
        assert_allclose(res.statistic, expected[0], atol=1e-15)
        assert_allclose(res.pvalue, expected[1], atol=1e-15)
        x = np.arange(10)
        y = np.array([9, 7, 8, 6, 5, 3, 4, 2, 1, 0])
        expected = (-0.9111111111111111, 0.0)
        res = stats.somersd(x, y)
        assert_allclose(res.statistic, expected[0], atol=1e-15)
        assert_allclose(res.pvalue, expected[1], atol=1e-15)
        x1 = [12, 2, 1, 12, 2]
        x2 = [1, 4, 7, 1, 0]
        expected = (-0.5, 0.30490178817878)
        res = stats.somersd(x1, x2)
        assert_allclose(res.statistic, expected[0], atol=1e-15)
        assert_allclose(res.pvalue, expected[1], atol=1e-15)
        res = stats.somersd([2, 2, 2], [2, 2, 2])
        assert_allclose(res.statistic, np.nan)
        assert_allclose(res.pvalue, np.nan)
        res = stats.somersd([2, 0, 2], [2, 2, 2])
        assert_allclose(res.statistic, np.nan)
        assert_allclose(res.pvalue, np.nan)
        res = stats.somersd([2, 2, 2], [2, 0, 2])
        assert_allclose(res.statistic, np.nan)
        assert_allclose(res.pvalue, np.nan)
        res = stats.somersd([0], [0])
        assert_allclose(res.statistic, np.nan)
        assert_allclose(res.pvalue, np.nan)
        res = stats.somersd([], [])
        assert_allclose(res.statistic, np.nan)
        assert_allclose(res.pvalue, np.nan)
        x = np.arange(10.0)
        y = np.arange(20.0)
        assert_raises(ValueError, stats.somersd, x, y)

    def test_asymmetry(self):
        if False:
            while True:
                i = 10
        x = [1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3]
        y = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        d_cr = 0.27272727272727
        d_rc = 0.34285714285714
        p = 0.0928919408837
        res = stats.somersd(x, y)
        assert_allclose(res.statistic, d_cr, atol=1e-15)
        assert_allclose(res.pvalue, p, atol=0.0001)
        assert_equal(res.table.shape, (3, 2))
        res = stats.somersd(y, x)
        assert_allclose(res.statistic, d_rc, atol=1e-15)
        assert_allclose(res.pvalue, p, atol=1e-15)
        assert_equal(res.table.shape, (2, 3))

    def test_somers_original(self):
        if False:
            while True:
                i = 10
        table = np.array([[8, 2], [6, 5], [3, 4], [1, 3], [2, 3]])
        table = table.T
        dyx = 129 / 340
        assert_allclose(stats.somersd(table).statistic, dyx)
        table = np.array([[25, 0], [85, 0], [0, 30]])
        (dxy, dyx) = (3300 / 5425, 3300 / 3300)
        assert_allclose(stats.somersd(table).statistic, dxy)
        assert_allclose(stats.somersd(table.T).statistic, dyx)
        table = np.array([[25, 0], [0, 30], [85, 0]])
        dyx = -1800 / 3300
        assert_allclose(stats.somersd(table.T).statistic, dyx)

    def test_contingency_table_with_zero_rows_cols(self):
        if False:
            for i in range(10):
                print('nop')
        N = 100
        shape = (4, 6)
        size = np.prod(shape)
        np.random.seed(0)
        s = stats.multinomial.rvs(N, p=np.ones(size) / size).reshape(shape)
        res = stats.somersd(s)
        s2 = np.insert(s, 2, np.zeros(shape[1]), axis=0)
        res2 = stats.somersd(s2)
        s3 = np.insert(s, 2, np.zeros(shape[0]), axis=1)
        res3 = stats.somersd(s3)
        s4 = np.insert(s2, 2, np.zeros(shape[0] + 1), axis=1)
        res4 = stats.somersd(s4)
        assert_allclose(res.statistic, -0.11698113207547, atol=1e-15)
        assert_allclose(res.statistic, res2.statistic)
        assert_allclose(res.statistic, res3.statistic)
        assert_allclose(res.statistic, res4.statistic)
        assert_allclose(res.pvalue, 0.15637644818815, atol=1e-15)
        assert_allclose(res.pvalue, res2.pvalue)
        assert_allclose(res.pvalue, res3.pvalue)
        assert_allclose(res.pvalue, res4.pvalue)

    def test_invalid_contingency_tables(self):
        if False:
            while True:
                i = 10
        N = 100
        shape = (4, 6)
        size = np.prod(shape)
        np.random.seed(0)
        s = stats.multinomial.rvs(N, p=np.ones(size) / size).reshape(shape)
        s5 = s - 2
        message = 'All elements of the contingency table must be non-negative'
        with assert_raises(ValueError, match=message):
            stats.somersd(s5)
        s6 = s + 0.01
        message = 'All elements of the contingency table must be integer'
        with assert_raises(ValueError, match=message):
            stats.somersd(s6)
        message = 'At least two elements of the contingency table must be nonzero.'
        with assert_raises(ValueError, match=message):
            stats.somersd([[]])
        with assert_raises(ValueError, match=message):
            stats.somersd([[1]])
        s7 = np.zeros((3, 3))
        with assert_raises(ValueError, match=message):
            stats.somersd(s7)
        s7[0, 1] = 1
        with assert_raises(ValueError, match=message):
            stats.somersd(s7)

    def test_only_ranks_matter(self):
        if False:
            print('Hello World!')
        x = [1, 2, 3]
        x2 = [-1, 2.1, np.inf]
        y = [3, 2, 1]
        y2 = [0, -0.5, -np.inf]
        res = stats.somersd(x, y)
        res2 = stats.somersd(x2, y2)
        assert_equal(res.statistic, res2.statistic)
        assert_equal(res.pvalue, res2.pvalue)

    def test_contingency_table_return(self):
        if False:
            while True:
                i = 10
        x = np.arange(10)
        y = np.arange(10)
        res = stats.somersd(x, y)
        assert_equal(res.table, np.eye(10))

    def test_somersd_alternative(self):
        if False:
            return 10
        x1 = [1, 2, 3, 4, 5]
        x2 = [5, 6, 7, 8, 7]
        expected = stats.somersd(x1, x2, alternative='two-sided')
        assert expected.statistic > 0
        res = stats.somersd(x1, x2, alternative='less')
        assert_equal(res.statistic, expected.statistic)
        assert_allclose(res.pvalue, 1 - expected.pvalue / 2)
        res = stats.somersd(x1, x2, alternative='greater')
        assert_equal(res.statistic, expected.statistic)
        assert_allclose(res.pvalue, expected.pvalue / 2)
        x2.reverse()
        expected = stats.somersd(x1, x2, alternative='two-sided')
        assert expected.statistic < 0
        res = stats.somersd(x1, x2, alternative='greater')
        assert_equal(res.statistic, expected.statistic)
        assert_allclose(res.pvalue, 1 - expected.pvalue / 2)
        res = stats.somersd(x1, x2, alternative='less')
        assert_equal(res.statistic, expected.statistic)
        assert_allclose(res.pvalue, expected.pvalue / 2)
        with pytest.raises(ValueError, match="alternative must be 'less'..."):
            stats.somersd(x1, x2, alternative='ekki-ekki')

    @pytest.mark.parametrize('positive_correlation', (False, True))
    def test_somersd_perfect_correlation(self, positive_correlation):
        if False:
            while True:
                i = 10
        x1 = np.arange(10)
        x2 = x1 if positive_correlation else np.flip(x1)
        expected_statistic = 1 if positive_correlation else -1
        res = stats.somersd(x1, x2, alternative='two-sided')
        assert res.statistic == expected_statistic
        assert res.pvalue == 0
        res = stats.somersd(x1, x2, alternative='less')
        assert res.statistic == expected_statistic
        assert res.pvalue == (1 if positive_correlation else 0)
        res = stats.somersd(x1, x2, alternative='greater')
        assert res.statistic == expected_statistic
        assert res.pvalue == (0 if positive_correlation else 1)

    def test_somersd_large_inputs_gh18132(self):
        if False:
            return 10
        classes = [1, 2]
        n_samples = 10 ** 6
        random.seed(6272161)
        x = random.choices(classes, k=n_samples)
        y = random.choices(classes, k=n_samples)
        val_sklearn = -0.001528138777036947
        val_scipy = stats.somersd(x, y).statistic
        assert_allclose(val_sklearn, val_scipy, atol=1e-15)

class TestBarnardExact:
    """Some tests to show that barnard_exact() works correctly."""

    @pytest.mark.parametrize('input_sample,expected', [([[43, 40], [10, 39]], (3.555406779643, 0.000362832367)), ([[100, 2], [1000, 5]], (-1.776382925679, 0.135126970878)), ([[2, 7], [8, 2]], (-2.518474945157, 0.01921081543)), ([[5, 1], [10, 10]], (1.449486150679, 0.156277546306)), ([[5, 15], [20, 20]], (-1.851640199545, 0.066363501421)), ([[5, 16], [20, 25]], (-1.609639949352, 0.116984852192)), ([[10, 5], [10, 1]], (-1.449486150679, 0.177536588915)), ([[5, 0], [1, 4]], (2.581988897472, 0.013671875)), ([[0, 1], [3, 2]], (-1.09544511501, 0.509667991877)), ([[0, 2], [6, 4]], (-1.549193338483, 0.197019618792)), ([[2, 7], [8, 2]], (-2.518474945157, 0.01921081543))])
    def test_precise(self, input_sample, expected):
        if False:
            return 10
        'The expected values have been generated by R, using a resolution\n        for the nuisance parameter of 1e-6 :\n        ```R\n        library(Barnard)\n        options(digits=10)\n        barnard.test(43, 40, 10, 39, dp=1e-6, pooled=TRUE)\n        ```\n        '
        res = barnard_exact(input_sample)
        (statistic, pvalue) = (res.statistic, res.pvalue)
        assert_allclose([statistic, pvalue], expected)

    @pytest.mark.parametrize('input_sample,expected', [([[43, 40], [10, 39]], (3.920362887717, 0.000289470662)), ([[100, 2], [1000, 5]], (-1.139432816087, 0.950272080594)), ([[2, 7], [8, 2]], (-3.079373904042, 0.020172119141)), ([[5, 1], [10, 10]], (1.622375939458, 0.150599922226)), ([[5, 15], [20, 20]], (-1.974771239528, 0.063038448651)), ([[5, 16], [20, 25]], (-1.722122973346, 0.133329494287)), ([[10, 5], [10, 1]], (-1.765469659009, 0.250566655215)), ([[5, 0], [1, 4]], (5.477225575052, 0.0078125)), ([[0, 1], [3, 2]], (-1.224744871392, 0.509667991877)), ([[0, 2], [6, 4]], (-1.732050807569, 0.197019618792)), ([[2, 7], [8, 2]], (-3.079373904042, 0.020172119141))])
    def test_pooled_param(self, input_sample, expected):
        if False:
            for i in range(10):
                print('nop')
        'The expected values have been generated by R, using a resolution\n        for the nuisance parameter of 1e-6 :\n        ```R\n        library(Barnard)\n        options(digits=10)\n        barnard.test(43, 40, 10, 39, dp=1e-6, pooled=FALSE)\n        ```\n        '
        res = barnard_exact(input_sample, pooled=False)
        (statistic, pvalue) = (res.statistic, res.pvalue)
        assert_allclose([statistic, pvalue], expected)

    def test_raises(self):
        if False:
            return 10
        error_msg = 'Number of points `n` must be strictly positive, found 0'
        with assert_raises(ValueError, match=error_msg):
            barnard_exact([[1, 2], [3, 4]], n=0)
        error_msg = 'The input `table` must be of shape \\(2, 2\\).'
        with assert_raises(ValueError, match=error_msg):
            barnard_exact(np.arange(6).reshape(2, 3))
        error_msg = 'All values in `table` must be nonnegative.'
        with assert_raises(ValueError, match=error_msg):
            barnard_exact([[-1, 2], [3, 4]])
        error_msg = "`alternative` should be one of {'two-sided', 'less', 'greater'}, found .*"
        with assert_raises(ValueError, match=error_msg):
            barnard_exact([[1, 2], [3, 4]], 'not-correct')

    @pytest.mark.parametrize('input_sample,expected', [([[0, 0], [4, 3]], (1.0, 0))])
    def test_edge_cases(self, input_sample, expected):
        if False:
            while True:
                i = 10
        res = barnard_exact(input_sample)
        (statistic, pvalue) = (res.statistic, res.pvalue)
        assert_equal(pvalue, expected[0])
        assert_equal(statistic, expected[1])

    @pytest.mark.parametrize('input_sample,expected', [([[0, 5], [0, 10]], (1.0, np.nan)), ([[5, 0], [10, 0]], (1.0, np.nan))])
    def test_row_or_col_zero(self, input_sample, expected):
        if False:
            while True:
                i = 10
        res = barnard_exact(input_sample)
        (statistic, pvalue) = (res.statistic, res.pvalue)
        assert_equal(pvalue, expected[0])
        assert_equal(statistic, expected[1])

    @pytest.mark.parametrize('input_sample,expected', [([[2, 7], [8, 2]], (-2.518474945157, 0.009886140845)), ([[7, 200], [300, 8]], (-21.32003669846, 0.0)), ([[21, 28], [1957, 6]], (-30.489638143953, 0.0))])
    @pytest.mark.parametrize('alternative', ['greater', 'less'])
    def test_less_greater(self, input_sample, expected, alternative):
        if False:
            print('Hello World!')
        '\n        "The expected values have been generated by R, using a resolution\n        for the nuisance parameter of 1e-6 :\n        ```R\n        library(Barnard)\n        options(digits=10)\n        a = barnard.test(2, 7, 8, 2, dp=1e-6, pooled=TRUE)\n        a$p.value[1]\n        ```\n        In this test, we are using the "one-sided" return value `a$p.value[1]`\n        to test our pvalue.\n        '
        (expected_stat, less_pvalue_expect) = expected
        if alternative == 'greater':
            input_sample = np.array(input_sample)[:, ::-1]
            expected_stat = -expected_stat
        res = barnard_exact(input_sample, alternative=alternative)
        (statistic, pvalue) = (res.statistic, res.pvalue)
        assert_allclose([statistic, pvalue], [expected_stat, less_pvalue_expect], atol=1e-07)

class TestBoschlooExact:
    """Some tests to show that boschloo_exact() works correctly."""
    ATOL = 1e-07

    @pytest.mark.parametrize('input_sample,expected', [([[2, 7], [8, 2]], (0.01852173, 0.009886142)), ([[5, 1], [10, 10]], (0.9782609, 0.9450994)), ([[5, 16], [20, 25]], (0.08913823, 0.05827348)), ([[10, 5], [10, 1]], (0.1652174, 0.08565611)), ([[5, 0], [1, 4]], (1, 1)), ([[0, 1], [3, 2]], (0.5, 0.34375)), ([[2, 7], [8, 2]], (0.01852173, 0.009886142)), ([[7, 12], [8, 3]], (0.06406797, 0.03410916)), ([[10, 24], [25, 37]], (0.2009359, 0.1512882))])
    def test_less(self, input_sample, expected):
        if False:
            for i in range(10):
                print('nop')
        'The expected values have been generated by R, using a resolution\n        for the nuisance parameter of 1e-8 :\n        ```R\n        library(Exact)\n        options(digits=10)\n        data <- matrix(c(43, 10, 40, 39), 2, 2, byrow=TRUE)\n        a = exact.test(data, method="Boschloo", alternative="less",\n                       tsmethod="central", np.interval=TRUE, beta=1e-8)\n        ```\n        '
        res = boschloo_exact(input_sample, alternative='less')
        (statistic, pvalue) = (res.statistic, res.pvalue)
        assert_allclose([statistic, pvalue], expected, atol=self.ATOL)

    @pytest.mark.parametrize('input_sample,expected', [([[43, 40], [10, 39]], (0.0002875544, 0.0001615562)), ([[2, 7], [8, 2]], (0.9990149, 0.9918327)), ([[5, 1], [10, 10]], (0.1652174, 0.09008534)), ([[5, 15], [20, 20]], (0.9849087, 0.9706997)), ([[5, 16], [20, 25]], (0.972349, 0.9524124)), ([[5, 0], [1, 4]], (0.02380952, 0.006865367)), ([[0, 1], [3, 2]], (1, 1)), ([[0, 2], [6, 4]], (1, 1)), ([[2, 7], [8, 2]], (0.9990149, 0.9918327)), ([[7, 12], [8, 3]], (0.9895302, 0.9771215)), ([[10, 24], [25, 37]], (0.9012936, 0.8633275))])
    def test_greater(self, input_sample, expected):
        if False:
            print('Hello World!')
        'The expected values have been generated by R, using a resolution\n        for the nuisance parameter of 1e-8 :\n        ```R\n        library(Exact)\n        options(digits=10)\n        data <- matrix(c(43, 10, 40, 39), 2, 2, byrow=TRUE)\n        a = exact.test(data, method="Boschloo", alternative="greater",\n                       tsmethod="central", np.interval=TRUE, beta=1e-8)\n        ```\n        '
        res = boschloo_exact(input_sample, alternative='greater')
        (statistic, pvalue) = (res.statistic, res.pvalue)
        assert_allclose([statistic, pvalue], expected, atol=self.ATOL)

    @pytest.mark.parametrize('input_sample,expected', [([[43, 40], [10, 39]], (0.0002875544, 0.0003231115)), ([[2, 7], [8, 2]], (0.01852173, 0.01977228)), ([[5, 1], [10, 10]], (0.1652174, 0.1801707)), ([[5, 16], [20, 25]], (0.08913823, 0.116547)), ([[5, 0], [1, 4]], (0.02380952, 0.01373073)), ([[0, 1], [3, 2]], (0.5, 0.6875)), ([[2, 7], [8, 2]], (0.01852173, 0.01977228)), ([[7, 12], [8, 3]], (0.06406797, 0.06821831))])
    def test_two_sided(self, input_sample, expected):
        if False:
            i = 10
            return i + 15
        'The expected values have been generated by R, using a resolution\n        for the nuisance parameter of 1e-8 :\n        ```R\n        library(Exact)\n        options(digits=10)\n        data <- matrix(c(43, 10, 40, 39), 2, 2, byrow=TRUE)\n        a = exact.test(data, method="Boschloo", alternative="two.sided",\n                       tsmethod="central", np.interval=TRUE, beta=1e-8)\n        ```\n        '
        res = boschloo_exact(input_sample, alternative='two-sided', n=64)
        (statistic, pvalue) = (res.statistic, res.pvalue)
        assert_allclose([statistic, pvalue], expected, atol=self.ATOL)

    def test_raises(self):
        if False:
            while True:
                i = 10
        error_msg = 'Number of points `n` must be strictly positive, found 0'
        with assert_raises(ValueError, match=error_msg):
            boschloo_exact([[1, 2], [3, 4]], n=0)
        error_msg = 'The input `table` must be of shape \\(2, 2\\).'
        with assert_raises(ValueError, match=error_msg):
            boschloo_exact(np.arange(6).reshape(2, 3))
        error_msg = 'All values in `table` must be nonnegative.'
        with assert_raises(ValueError, match=error_msg):
            boschloo_exact([[-1, 2], [3, 4]])
        error_msg = "`alternative` should be one of \\('two-sided', 'less', 'greater'\\), found .*"
        with assert_raises(ValueError, match=error_msg):
            boschloo_exact([[1, 2], [3, 4]], 'not-correct')

    @pytest.mark.parametrize('input_sample,expected', [([[0, 5], [0, 10]], (np.nan, np.nan)), ([[5, 0], [10, 0]], (np.nan, np.nan))])
    def test_row_or_col_zero(self, input_sample, expected):
        if False:
            print('Hello World!')
        res = boschloo_exact(input_sample)
        (statistic, pvalue) = (res.statistic, res.pvalue)
        assert_equal(pvalue, expected[0])
        assert_equal(statistic, expected[1])

    def test_two_sided_gt_1(self):
        if False:
            print('Hello World!')
        tbl = [[1, 1], [13, 12]]
        pl = boschloo_exact(tbl, alternative='less').pvalue
        pg = boschloo_exact(tbl, alternative='greater').pvalue
        assert 2 * min(pl, pg) > 1
        pt = boschloo_exact(tbl, alternative='two-sided').pvalue
        assert pt == 1.0

    @pytest.mark.parametrize('alternative', ('less', 'greater'))
    def test_against_fisher_exact(self, alternative):
        if False:
            for i in range(10):
                print('nop')
        tbl = [[2, 7], [8, 2]]
        boschloo_stat = boschloo_exact(tbl, alternative=alternative).statistic
        fisher_p = stats.fisher_exact(tbl, alternative=alternative)[1]
        assert_allclose(boschloo_stat, fisher_p)

class TestCvm_2samp:

    def test_invalid_input(self):
        if False:
            return 10
        x = np.arange(10).reshape((2, 5))
        y = np.arange(5)
        msg = 'The samples must be one-dimensional'
        with pytest.raises(ValueError, match=msg):
            cramervonmises_2samp(x, y)
        with pytest.raises(ValueError, match=msg):
            cramervonmises_2samp(y, x)
        msg = 'x and y must contain at least two observations.'
        with pytest.raises(ValueError, match=msg):
            cramervonmises_2samp([], y)
        with pytest.raises(ValueError, match=msg):
            cramervonmises_2samp(y, [1])
        msg = 'method must be either auto, exact or asymptotic'
        with pytest.raises(ValueError, match=msg):
            cramervonmises_2samp(y, y, 'xyz')

    def test_list_input(self):
        if False:
            i = 10
            return i + 15
        x = [2, 3, 4, 7, 6]
        y = [0.2, 0.7, 12, 18]
        r1 = cramervonmises_2samp(x, y)
        r2 = cramervonmises_2samp(np.array(x), np.array(y))
        assert_equal((r1.statistic, r1.pvalue), (r2.statistic, r2.pvalue))

    def test_example_conover(self):
        if False:
            return 10
        x = [7.6, 8.4, 8.6, 8.7, 9.3, 9.9, 10.1, 10.6, 11.2]
        y = [5.2, 5.7, 5.9, 6.5, 6.8, 8.2, 9.1, 9.8, 10.8, 11.3, 11.5, 12.3, 12.5, 13.4, 14.6]
        r = cramervonmises_2samp(x, y)
        assert_allclose(r.statistic, 0.262, atol=0.001)
        assert_allclose(r.pvalue, 0.18, atol=0.01)

    @pytest.mark.parametrize('statistic, m, n, pval', [(710, 5, 6, 48.0 / 462), (1897, 7, 7, 117.0 / 1716), (576, 4, 6, 2.0 / 210), (1764, 6, 7, 2.0 / 1716)])
    def test_exact_pvalue(self, statistic, m, n, pval):
        if False:
            print('Hello World!')
        assert_equal(_pval_cvm_2samp_exact(statistic, m, n), pval)

    def test_large_sample(self):
        if False:
            while True:
                i = 10
        np.random.seed(4367)
        x = distributions.norm.rvs(size=1000000)
        y = distributions.norm.rvs(size=900000)
        r = cramervonmises_2samp(x, y)
        assert_(0 < r.pvalue < 1)
        r = cramervonmises_2samp(x, y + 0.1)
        assert_(0 < r.pvalue < 1)

    def test_exact_vs_asymptotic(self):
        if False:
            while True:
                i = 10
        np.random.seed(0)
        x = np.random.rand(7)
        y = np.random.rand(8)
        r1 = cramervonmises_2samp(x, y, method='exact')
        r2 = cramervonmises_2samp(x, y, method='asymptotic')
        assert_equal(r1.statistic, r2.statistic)
        assert_allclose(r1.pvalue, r2.pvalue, atol=0.01)

    def test_method_auto(self):
        if False:
            return 10
        x = np.arange(20)
        y = [0.5, 4.7, 13.1]
        r1 = cramervonmises_2samp(x, y, method='exact')
        r2 = cramervonmises_2samp(x, y, method='auto')
        assert_equal(r1.pvalue, r2.pvalue)
        x = np.arange(21)
        r1 = cramervonmises_2samp(x, y, method='asymptotic')
        r2 = cramervonmises_2samp(x, y, method='auto')
        assert_equal(r1.pvalue, r2.pvalue)

    def test_same_input(self):
        if False:
            i = 10
            return i + 15
        x = np.arange(15)
        res = cramervonmises_2samp(x, x)
        assert_equal((res.statistic, res.pvalue), (0.0, 1.0))
        res = cramervonmises_2samp(x[:4], x[:4])
        assert_equal((res.statistic, res.pvalue), (0.0, 1.0))

class TestTukeyHSD:
    data_same_size = ([24.5, 23.5, 26.4, 27.1, 29.9], [28.4, 34.2, 29.5, 32.2, 30.1], [26.1, 28.3, 24.3, 26.2, 27.8])
    data_diff_size = ([24.5, 23.5, 26.28, 26.4, 27.1, 29.9, 30.1, 30.1], [28.4, 34.2, 29.5, 32.2, 30.1], [26.1, 28.3, 24.3, 26.2, 27.8])
    extreme_size = ([24.5, 23.5, 26.4], [28.4, 34.2, 29.5, 32.2, 30.1, 28.4, 34.2, 29.5, 32.2, 30.1], [26.1, 28.3, 24.3, 26.2, 27.8])
    sas_same_size = '\n    Comparison LowerCL Difference UpperCL Significance\n    2 - 3\t0.6908830568\t4.34\t7.989116943\t    1\n    2 - 1\t0.9508830568\t4.6 \t8.249116943 \t1\n    3 - 2\t-7.989116943\t-4.34\t-0.6908830568\t1\n    3 - 1\t-3.389116943\t0.26\t3.909116943\t    0\n    1 - 2\t-8.249116943\t-4.6\t-0.9508830568\t1\n    1 - 3\t-3.909116943\t-0.26\t3.389116943\t    0\n    '
    sas_diff_size = '\n    Comparison LowerCL Difference UpperCL Significance\n    2 - 1\t0.2679292645\t3.645\t7.022070736\t    1\n    2 - 3\t0.5934764007\t4.34\t8.086523599\t    1\n    1 - 2\t-7.022070736\t-3.645\t-0.2679292645\t1\n    1 - 3\t-2.682070736\t0.695\t4.072070736\t    0\n    3 - 2\t-8.086523599\t-4.34\t-0.5934764007\t1\n    3 - 1\t-4.072070736\t-0.695\t2.682070736\t    0\n    '
    sas_extreme = '\n    Comparison LowerCL Difference UpperCL Significance\n    2 - 3\t1.561605075\t    4.34\t7.118394925\t    1\n    2 - 1\t2.740784879\t    6.08\t9.419215121\t    1\n    3 - 2\t-7.118394925\t-4.34\t-1.561605075\t1\n    3 - 1\t-1.964526566\t1.74\t5.444526566\t    0\n    1 - 2\t-9.419215121\t-6.08\t-2.740784879\t1\n    1 - 3\t-5.444526566\t-1.74\t1.964526566\t    0\n    '

    @pytest.mark.parametrize('data,res_expect_str,atol', ((data_same_size, sas_same_size, 0.0001), (data_diff_size, sas_diff_size, 0.0001), (extreme_size, sas_extreme, 1e-10)), ids=['equal size sample', 'unequal sample size', 'extreme sample size differences'])
    def test_compare_sas(self, data, res_expect_str, atol):
        if False:
            for i in range(10):
                print('nop')
        '\n        SAS code used to generate results for each sample:\n        DATA ACHE;\n        INPUT BRAND RELIEF;\n        CARDS;\n        1 24.5\n        ...\n        3 27.8\n        ;\n        ods graphics on;   ODS RTF;ODS LISTING CLOSE;\n           PROC ANOVA DATA=ACHE;\n           CLASS BRAND;\n           MODEL RELIEF=BRAND;\n           MEANS BRAND/TUKEY CLDIFF;\n           TITLE \'COMPARE RELIEF ACROSS MEDICINES  - ANOVA EXAMPLE\';\n           ods output  CLDiffs =tc;\n        proc print data=tc;\n            format LowerCL 17.16 UpperCL 17.16 Difference 17.16;\n            title "Output with many digits";\n        RUN;\n        QUIT;\n        ODS RTF close;\n        ODS LISTING;\n        '
        res_expect = np.asarray(res_expect_str.replace(' - ', ' ').split()[5:], dtype=float).reshape((6, 6))
        res_tukey = stats.tukey_hsd(*data)
        conf = res_tukey.confidence_interval()
        for (i, j, l, s, h, sig) in res_expect:
            (i, j) = (int(i) - 1, int(j) - 1)
            assert_allclose(conf.low[i, j], l, atol=atol)
            assert_allclose(res_tukey.statistic[i, j], s, atol=atol)
            assert_allclose(conf.high[i, j], h, atol=atol)
            assert_allclose(res_tukey.pvalue[i, j] <= 0.05, sig == 1)
    matlab_sm_siz = '\n        1\t2\t-8.2491590248597\t-4.6\t-0.9508409751403\t0.0144483269098\n        1\t3\t-3.9091590248597\t-0.26\t3.3891590248597\t0.9803107240900\n        2\t3\t0.6908409751403\t4.34\t7.9891590248597\t0.0203311368795\n        '
    matlab_diff_sz = '\n        1\t2\t-7.02207069748501\t-3.645\t-0.26792930251500 0.03371498443080\n        1\t3\t-2.68207069748500\t0.695\t4.07207069748500 0.85572267328807\n        2\t3\t0.59347644287720\t4.34\t8.08652355712281 0.02259047020620\n        '

    @pytest.mark.parametrize('data,res_expect_str,atol', ((data_same_size, matlab_sm_siz, 1e-12), (data_diff_size, matlab_diff_sz, 1e-07)), ids=['equal size sample', 'unequal size sample'])
    def test_compare_matlab(self, data, res_expect_str, atol):
        if False:
            while True:
                i = 10
        '\n        vals = [24.5, 23.5,  26.4, 27.1, 29.9, 28.4, 34.2, 29.5, 32.2, 30.1,\n         26.1, 28.3, 24.3, 26.2, 27.8]\n        names = {\'zero\', \'zero\', \'zero\', \'zero\', \'zero\', \'one\', \'one\', \'one\',\n         \'one\', \'one\', \'two\', \'two\', \'two\', \'two\', \'two\'}\n        [p,t,stats] = anova1(vals,names,"off");\n        [c,m,h,nms] = multcompare(stats, "CType","hsd");\n        '
        res_expect = np.asarray(res_expect_str.split(), dtype=float).reshape((3, 6))
        res_tukey = stats.tukey_hsd(*data)
        conf = res_tukey.confidence_interval()
        for (i, j, l, s, h, p) in res_expect:
            (i, j) = (int(i) - 1, int(j) - 1)
            assert_allclose(conf.low[i, j], l, atol=atol)
            assert_allclose(res_tukey.statistic[i, j], s, atol=atol)
            assert_allclose(conf.high[i, j], h, atol=atol)
            assert_allclose(res_tukey.pvalue[i, j], p, atol=atol)

    def test_compare_r(self):
        if False:
            while True:
                i = 10
        '\n        Testing against results and p-values from R:\n        from: https://www.rdocumentation.org/packages/stats/versions/3.6.2/\n        topics/TukeyHSD\n        > require(graphics)\n        > summary(fm1 <- aov(breaks ~ tension, data = warpbreaks))\n        > TukeyHSD(fm1, "tension", ordered = TRUE)\n        > plot(TukeyHSD(fm1, "tension"))\n        Tukey multiple comparisons of means\n        95% family-wise confidence level\n        factor levels have been ordered\n        Fit: aov(formula = breaks ~ tension, data = warpbreaks)\n        $tension\n        '
        str_res = '\n                diff        lwr      upr     p adj\n        2 - 3  4.722222 -4.8376022 14.28205 0.4630831\n        1 - 3 14.722222  5.1623978 24.28205 0.0014315\n        1 - 2 10.000000  0.4401756 19.55982 0.0384598\n        '
        res_expect = np.asarray(str_res.replace(' - ', ' ').split()[5:], dtype=float).reshape((3, 6))
        data = ([26, 30, 54, 25, 70, 52, 51, 26, 67, 27, 14, 29, 19, 29, 31, 41, 20, 44], [18, 21, 29, 17, 12, 18, 35, 30, 36, 42, 26, 19, 16, 39, 28, 21, 39, 29], [36, 21, 24, 18, 10, 43, 28, 15, 26, 20, 21, 24, 17, 13, 15, 15, 16, 28])
        res_tukey = stats.tukey_hsd(*data)
        conf = res_tukey.confidence_interval()
        for (i, j, s, l, h, p) in res_expect:
            (i, j) = (int(i) - 1, int(j) - 1)
            assert_allclose(conf.low[i, j], l, atol=1e-07)
            assert_allclose(res_tukey.statistic[i, j], s, atol=1e-06)
            assert_allclose(conf.high[i, j], h, atol=1e-05)
            assert_allclose(res_tukey.pvalue[i, j], p, atol=1e-07)

    def test_engineering_stat_handbook(self):
        if False:
            return 10
        '\n        Example sourced from:\n        https://www.itl.nist.gov/div898/handbook/prc/section4/prc471.htm\n        '
        group1 = [6.9, 5.4, 5.8, 4.6, 4.0]
        group2 = [8.3, 6.8, 7.8, 9.2, 6.5]
        group3 = [8.0, 10.5, 8.1, 6.9, 9.3]
        group4 = [5.8, 3.8, 6.1, 5.6, 6.2]
        res = stats.tukey_hsd(group1, group2, group3, group4)
        conf = res.confidence_interval()
        lower = np.asarray([[0, 0, 0, -2.25], [0.29, 0, -2.93, 0.13], [1.13, 0, 0, 0.97], [0, 0, 0, 0]])
        upper = np.asarray([[0, 0, 0, 1.93], [4.47, 0, 1.25, 4.31], [5.31, 0, 0, 5.15], [0, 0, 0, 0]])
        for (i, j) in [(1, 0), (2, 0), (0, 3), (1, 2), (2, 3)]:
            assert_allclose(conf.low[i, j], lower[i, j], atol=0.01)
            assert_allclose(conf.high[i, j], upper[i, j], atol=0.01)

    def test_rand_symm(self):
        if False:
            print('Hello World!')
        np.random.seed(1234)
        data = np.random.rand(3, 100)
        res = stats.tukey_hsd(*data)
        conf = res.confidence_interval()
        assert_equal(conf.low, -conf.high.T)
        assert_equal(np.diagonal(conf.high), conf.high[0, 0])
        assert_equal(np.diagonal(conf.low), conf.low[0, 0])
        assert_equal(res.statistic, -res.statistic.T)
        assert_equal(np.diagonal(res.statistic), 0)
        assert_equal(res.pvalue, res.pvalue.T)
        assert_equal(np.diagonal(res.pvalue), 1)

    def test_no_inf(self):
        if False:
            while True:
                i = 10
        with assert_raises(ValueError, match='...must be finite.'):
            stats.tukey_hsd([1, 2, 3], [2, np.inf], [6, 7, 3])

    def test_is_1d(self):
        if False:
            print('Hello World!')
        with assert_raises(ValueError, match='...must be one-dimensional'):
            stats.tukey_hsd([[1, 2], [2, 3]], [2, 5], [5, 23, 6])

    def test_no_empty(self):
        if False:
            i = 10
            return i + 15
        with assert_raises(ValueError, match='...must be greater than one'):
            stats.tukey_hsd([], [2, 5], [4, 5, 6])

    @pytest.mark.parametrize('nargs', (0, 1))
    def test_not_enough_treatments(self, nargs):
        if False:
            print('Hello World!')
        with assert_raises(ValueError, match='...more than 1 treatment.'):
            stats.tukey_hsd(*[[23, 7, 3]] * nargs)

    @pytest.mark.parametrize('cl', [-0.5, 0, 1, 2])
    def test_conf_level_invalid(self, cl):
        if False:
            for i in range(10):
                print('nop')
        with assert_raises(ValueError, match='must be between 0 and 1'):
            r = stats.tukey_hsd([23, 7, 3], [3, 4], [9, 4])
            r.confidence_interval(cl)

    def test_2_args_ttest(self):
        if False:
            for i in range(10):
                print('nop')
        res_tukey = stats.tukey_hsd(*self.data_diff_size[:2])
        res_ttest = stats.ttest_ind(*self.data_diff_size[:2])
        assert_allclose(res_ttest.pvalue, res_tukey.pvalue[0, 1])
        assert_allclose(res_ttest.pvalue, res_tukey.pvalue[1, 0])

class TestPoissonMeansTest:

    @pytest.mark.parametrize('c1, n1, c2, n2, p_expect', ([0, 100, 3, 100, 0.0884], [2, 100, 6, 100, 0.1749]))
    def test_paper_examples(self, c1, n1, c2, n2, p_expect):
        if False:
            for i in range(10):
                print('nop')
        res = stats.poisson_means_test(c1, n1, c2, n2)
        assert_allclose(res.pvalue, p_expect, atol=0.0001)

    @pytest.mark.parametrize('c1, n1, c2, n2, p_expect, alt, d', ([20, 10, 20, 10, 0.999999756892963, 'two-sided', 0], [10, 10, 10, 10, 0.9999998403241203, 'two-sided', 0], [50, 15, 1, 1, 0.09920321053409643, 'two-sided', 0.05], [3, 100, 20, 300, 0.12202725450896404, 'two-sided', 0], [3, 12, 4, 20, 0.40416087318539173, 'greater', 0], [4, 20, 3, 100, 0.008053640402974236, 'greater', 0], [4, 20, 3, 10, 0.3083216325432898, 'less', 0], [1, 1, 50, 15, 0.09322998607245102, 'less', 0]))
    def test_fortran_authors(self, c1, n1, c2, n2, p_expect, alt, d):
        if False:
            print('Hello World!')
        res = stats.poisson_means_test(c1, n1, c2, n2, alternative=alt, diff=d)
        assert_allclose(res.pvalue, p_expect, atol=2e-06, rtol=1e-16)

    def test_different_results(self):
        if False:
            while True:
                i = 10
        (count1, count2) = (10000, 10000)
        (nobs1, nobs2) = (10000, 10000)
        res = stats.poisson_means_test(count1, nobs1, count2, nobs2)
        assert_allclose(res.pvalue, 1)

    def test_less_than_zero_lambda_hat2(self):
        if False:
            return 10
        (count1, count2) = (0, 0)
        (nobs1, nobs2) = (1, 1)
        res = stats.poisson_means_test(count1, nobs1, count2, nobs2)
        assert_allclose(res.pvalue, 1)

    def test_input_validation(self):
        if False:
            print('Hello World!')
        (count1, count2) = (0, 0)
        (nobs1, nobs2) = (1, 1)
        message = '`k1` and `k2` must be integers.'
        with assert_raises(TypeError, match=message):
            stats.poisson_means_test(0.7, nobs1, count2, nobs2)
        with assert_raises(TypeError, match=message):
            stats.poisson_means_test(count1, nobs1, 0.7, nobs2)
        message = '`k1` and `k2` must be greater than or equal to 0.'
        with assert_raises(ValueError, match=message):
            stats.poisson_means_test(-1, nobs1, count2, nobs2)
        with assert_raises(ValueError, match=message):
            stats.poisson_means_test(count1, nobs1, -1, nobs2)
        message = '`n1` and `n2` must be greater than 0.'
        with assert_raises(ValueError, match=message):
            stats.poisson_means_test(count1, -1, count2, nobs2)
        with assert_raises(ValueError, match=message):
            stats.poisson_means_test(count1, nobs1, count2, -1)
        message = 'diff must be greater than or equal to 0.'
        with assert_raises(ValueError, match=message):
            stats.poisson_means_test(count1, nobs1, count2, nobs2, diff=-1)
        message = 'Alternative must be one of ...'
        with assert_raises(ValueError, match=message):
            stats.poisson_means_test(1, 2, 1, 2, alternative='error')

class TestBWSTest:

    def test_bws_input_validation(self):
        if False:
            print('Hello World!')
        rng = np.random.default_rng(4571775098104213308)
        (x, y) = rng.random(size=(2, 7))
        message = '`x` and `y` must be exactly one-dimensional.'
        with pytest.raises(ValueError, match=message):
            stats.bws_test([x, x], [y, y])
        message = '`x` and `y` must not contain NaNs.'
        with pytest.raises(ValueError, match=message):
            stats.bws_test([np.nan], y)
        message = '`x` and `y` must be of nonzero size.'
        with pytest.raises(ValueError, match=message):
            stats.bws_test(x, [])
        message = 'alternative` must be one of...'
        with pytest.raises(ValueError, match=message):
            stats.bws_test(x, y, alternative='ekki-ekki')
        message = 'method` must be an instance of...'
        with pytest.raises(ValueError, match=message):
            stats.bws_test(x, y, method=42)

    def test_against_published_reference(self):
        if False:
            print('Hello World!')
        x = [1, 2, 3, 4, 6, 7, 8]
        y = [5, 9, 10, 11, 12, 13, 14]
        res = stats.bws_test(x, y, alternative='two-sided')
        assert_allclose(res.statistic, 5.132, atol=0.001)
        assert_equal(res.pvalue, 10 / 3432)

    @pytest.mark.parametrize(('alternative', 'statistic', 'pvalue'), [('two-sided', 1.7510204081633, 0.1264422777777), ('less', -1.7510204081633, 0.05754662004662), ('greater', -1.7510204081633, 0.9424533799534)])
    def test_against_R(self, alternative, statistic, pvalue):
        if False:
            for i in range(10):
                print('nop')
        rng = np.random.default_rng(4571775098104213308)
        (x, y) = rng.random(size=(2, 7))
        res = stats.bws_test(x, y, alternative=alternative)
        assert_allclose(res.statistic, statistic, rtol=1e-13)
        assert_allclose(res.pvalue, pvalue, atol=0.01, rtol=0.1)

    @pytest.mark.parametrize(('alternative', 'statistic', 'pvalue'), [('two-sided', 1.142629265891, 0.2903950180801), ('less', 0.99629665877411, 0.8545660222131), ('greater', 0.99629665877411, 0.1454339777869)])
    def test_against_R_imbalanced(self, alternative, statistic, pvalue):
        if False:
            return 10
        rng = np.random.default_rng(5429015622386364034)
        x = rng.random(size=9)
        y = rng.random(size=8)
        res = stats.bws_test(x, y, alternative=alternative)
        assert_allclose(res.statistic, statistic, rtol=1e-13)
        assert_allclose(res.pvalue, pvalue, atol=0.01, rtol=0.1)

    def test_method(self):
        if False:
            for i in range(10):
                print('nop')
        rng = np.random.default_rng(1520514347193347862)
        (x, y) = rng.random(size=(2, 10))
        rng = np.random.default_rng(1520514347193347862)
        method = stats.PermutationMethod(n_resamples=10, random_state=rng)
        res1 = stats.bws_test(x, y, method=method)
        assert len(res1.null_distribution) == 10
        rng = np.random.default_rng(1520514347193347862)
        method = stats.PermutationMethod(n_resamples=10, random_state=rng)
        res2 = stats.bws_test(x, y, method=method)
        assert_allclose(res1.null_distribution, res2.null_distribution)
        rng = np.random.default_rng(5205143471933478621)
        method = stats.PermutationMethod(n_resamples=10, random_state=rng)
        res3 = stats.bws_test(x, y, method=method)
        assert not np.allclose(res3.null_distribution, res1.null_distribution)

    def test_directions(self):
        if False:
            i = 10
            return i + 15
        rng = np.random.default_rng(1520514347193347862)
        x = rng.random(size=5)
        y = x - 1
        res = stats.bws_test(x, y, alternative='greater')
        assert res.statistic > 0
        assert_equal(res.pvalue, 1 / len(res.null_distribution))
        res = stats.bws_test(x, y, alternative='less')
        assert res.statistic > 0
        assert_equal(res.pvalue, 1)
        res = stats.bws_test(y, x, alternative='less')
        assert res.statistic < 0
        assert_equal(res.pvalue, 1 / len(res.null_distribution))
        res = stats.bws_test(y, x, alternative='greater')
        assert res.statistic < 0
        assert_equal(res.pvalue, 1)