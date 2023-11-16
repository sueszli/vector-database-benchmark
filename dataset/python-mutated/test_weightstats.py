"""tests for weightstats, compares with replication

no failures but needs cleanup
update 2012-09-09:
   added test after fixing bug in covariance
   TODOs:
     - I do not remember what all the commented out code is doing
     - should be refactored to use generator or inherited tests
     - still gaps in test coverage
       - value/diff in ttest_ind is tested in test_tost.py
     - what about pandas data structures?

Author: Josef Perktold
License: BSD (3-clause)

"""
import numpy as np
from scipy import stats
import pandas as pd
from numpy.testing import assert_, assert_almost_equal, assert_allclose
from statsmodels.stats.weightstats import DescrStatsW, CompareMeans, ttest_ind, ztest, zconfint
from statsmodels.tools.testing import Holder

class CheckExternalMixin:

    @classmethod
    def get_descriptives(cls, ddof=0):
        if False:
            for i in range(10):
                print('nop')
        cls.descriptive = DescrStatsW(cls.data, cls.weights, ddof)

    @classmethod
    def save_data(cls, fname='data.csv'):
        if False:
            return 10
        df = pd.DataFrame(index=np.arange(len(cls.weights)))
        df['weights'] = cls.weights
        if cls.data.ndim == 1:
            df['data1'] = cls.data
        else:
            for k in range(cls.data.shape[1]):
                df['data%d' % (k + 1)] = cls.data[:, k]
        df.to_csv(fname)

    def test_mean(self):
        if False:
            while True:
                i = 10
        mn = self.descriptive.mean
        assert_allclose(mn, self.mean, rtol=0.0001)

    def test_sum(self):
        if False:
            print('Hello World!')
        sm = self.descriptive.sum
        assert_allclose(sm, self.sum, rtol=0.0001)

    def test_var(self):
        if False:
            i = 10
            return i + 15
        var = self.descriptive.var
        assert_allclose(var, self.var, rtol=0.0001)

    def test_std(self):
        if False:
            for i in range(10):
                print('nop')
        std = self.descriptive.std
        assert_allclose(std, self.std, rtol=0.0001)

    def test_sem(self):
        if False:
            i = 10
            return i + 15
        if not hasattr(self, 'sem'):
            return
        sem = self.descriptive.std_mean
        assert_allclose(sem, self.sem, rtol=0.0001)

    def test_quantiles(self):
        if False:
            while True:
                i = 10
        quant = np.asarray(self.quantiles, dtype=np.float64)
        for return_pandas in (False, True):
            qtl = self.descriptive.quantile(self.quantile_probs, return_pandas=return_pandas)
            qtl = np.asarray(qtl, dtype=np.float64)
            assert_allclose(qtl, quant, rtol=0.0001)

class TestSim1(CheckExternalMixin):
    mean = 0.401499
    sum = 12.9553441
    var = 1.08022
    std = 1.03933
    quantiles = np.r_[-1.81098, -0.84052, 0.32859, 0.77808, 2.93431]

    @classmethod
    def setup_class(cls):
        if False:
            i = 10
            return i + 15
        np.random.seed(9876789)
        cls.data = np.random.normal(size=20)
        cls.weights = np.random.uniform(0, 3, size=20)
        cls.quantile_probs = np.r_[0, 0.1, 0.5, 0.75, 1]
        cls.get_descriptives()

class TestSim1t(CheckExternalMixin):
    mean = 5.05103296
    sum = 156.573464
    var = 9.9711934
    std = 3.15771965
    quantiles = np.r_[0, 1, 5, 8, 9]

    @classmethod
    def setup_class(cls):
        if False:
            while True:
                i = 10
        np.random.seed(9876789)
        cls.data = np.random.randint(0, 10, size=20)
        cls.data[15:20] = cls.data[0:5]
        cls.data[18:20] = cls.data[15:17]
        cls.weights = np.random.uniform(0, 3, size=20)
        cls.quantile_probs = np.r_[0, 0.1, 0.5, 0.75, 1]
        cls.get_descriptives()

class TestSim1n(CheckExternalMixin):
    mean = -0.3131058
    sum = -6.2621168
    var = 0.49722696
    std = 0.70514322
    sem = 0.15767482
    quantiles = np.r_[-1.61593, -1.45576, -0.24356, 0.1677, 1.18791]

    @classmethod
    def setup_class(cls):
        if False:
            print('Hello World!')
        np.random.seed(4342)
        cls.data = np.random.normal(size=20)
        cls.weights = np.random.uniform(0, 3, size=20)
        cls.weights *= 20 / cls.weights.sum()
        cls.quantile_probs = np.r_[0, 0.1, 0.5, 0.75, 1]
        cls.get_descriptives(1)

class TestSim2(CheckExternalMixin):
    mean = [-0.2170406, -0.2387543]
    sum = [-6.8383999, -7.5225444]
    var = [1.77426344, 0.61933542]
    std = [1.3320148, 0.78697867]
    quantiles = np.column_stack((np.r_[-2.55277, -1.40479, -0.6104, 0.5274, 2.66246], np.r_[-1.49263, -1.15403, -0.16231, 0.16464, 1.83062]))

    @classmethod
    def setup_class(cls):
        if False:
            for i in range(10):
                print('nop')
        np.random.seed(2249)
        cls.data = np.random.normal(size=(20, 2))
        cls.weights = np.random.uniform(0, 3, size=20)
        cls.quantile_probs = np.r_[0, 0.1, 0.5, 0.75, 1]
        cls.get_descriptives()

class TestWeightstats:

    @classmethod
    def setup_class(cls):
        if False:
            return 10
        np.random.seed(9876789)
        (n1, n2) = (20, 20)
        (m1, m2) = (1, 1.2)
        x1 = m1 + np.random.randn(n1)
        x2 = m2 + np.random.randn(n2)
        x1_2d = m1 + np.random.randn(n1, 3)
        x2_2d = m2 + np.random.randn(n2, 3)
        w1 = np.random.randint(1, 4, n1)
        w2 = np.random.randint(1, 4, n2)
        (cls.x1, cls.x2) = (x1, x2)
        (cls.w1, cls.w2) = (w1, w2)
        (cls.x1_2d, cls.x2_2d) = (x1_2d, x2_2d)

    def test_weightstats_1(self):
        if False:
            while True:
                i = 10
        (x1, x2) = (self.x1, self.x2)
        (w1, w2) = (self.w1, self.w2)
        w1_ = 2.0 * np.ones(len(x1))
        w2_ = 2.0 * np.ones(len(x2))
        d1 = DescrStatsW(x1)
        assert_almost_equal(ttest_ind(x1, x2, weights=(w1_, w2_))[:2], stats.ttest_ind(np.r_[x1, x1], np.r_[x2, x2]))

    def test_weightstats_2(self):
        if False:
            i = 10
            return i + 15
        (x1, x2) = (self.x1, self.x2)
        (w1, w2) = (self.w1, self.w2)
        d1 = DescrStatsW(x1)
        d1w = DescrStatsW(x1, weights=w1)
        d2w = DescrStatsW(x2, weights=w2)
        x1r = d1w.asrepeats()
        x2r = d2w.asrepeats()
        assert_almost_equal(ttest_ind(x1, x2, weights=(w1, w2))[:2], stats.ttest_ind(x1r, x2r), 14)
        assert_almost_equal(x2r.mean(0), d2w.mean, 14)
        assert_almost_equal(x2r.var(), d2w.var, 14)
        assert_almost_equal(x2r.std(), d2w.std, 14)
        assert_almost_equal(np.cov(x2r, bias=1), d2w.cov, 14)
        assert_almost_equal(d1.ttest_mean(3)[:2], stats.ttest_1samp(x1, 3), 11)
        assert_almost_equal(d1w.ttest_mean(3)[:2], stats.ttest_1samp(x1r, 3), 11)

    def test_weightstats_3(self):
        if False:
            while True:
                i = 10
        (x1_2d, x2_2d) = (self.x1_2d, self.x2_2d)
        (w1, w2) = (self.w1, self.w2)
        d1w_2d = DescrStatsW(x1_2d, weights=w1)
        d2w_2d = DescrStatsW(x2_2d, weights=w2)
        x1r_2d = d1w_2d.asrepeats()
        x2r_2d = d2w_2d.asrepeats()
        assert_almost_equal(x2r_2d.mean(0), d2w_2d.mean, 14)
        assert_almost_equal(x2r_2d.var(0), d2w_2d.var, 14)
        assert_almost_equal(x2r_2d.std(0), d2w_2d.std, 14)
        assert_almost_equal(np.cov(x2r_2d.T, bias=1), d2w_2d.cov, 14)
        assert_almost_equal(np.corrcoef(x2r_2d.T), d2w_2d.corrcoef, 14)
        (t, p, d) = d1w_2d.ttest_mean(3)
        assert_almost_equal([t, p], stats.ttest_1samp(x1r_2d, 3), 11)
        cm = CompareMeans(d1w_2d, d2w_2d)
        ressm = cm.ttest_ind()
        resss = stats.ttest_ind(x1r_2d, x2r_2d)
        assert_almost_equal(ressm[:2], resss, 14)

    def test_weightstats_ddof_tests(self):
        if False:
            while True:
                i = 10
        x1_2d = self.x1_2d
        w1 = self.w1
        d1w_d0 = DescrStatsW(x1_2d, weights=w1, ddof=0)
        d1w_d1 = DescrStatsW(x1_2d, weights=w1, ddof=1)
        d1w_d2 = DescrStatsW(x1_2d, weights=w1, ddof=2)
        res0 = d1w_d0.ttest_mean()
        res1 = d1w_d1.ttest_mean()
        res2 = d1w_d2.ttest_mean()
        assert_almost_equal(np.r_[res1], np.r_[res0], 14)
        assert_almost_equal(np.r_[res2], np.r_[res0], 14)
        res0 = d1w_d0.ttest_mean(0.5)
        res1 = d1w_d1.ttest_mean(0.5)
        res2 = d1w_d2.ttest_mean(0.5)
        assert_almost_equal(np.r_[res1], np.r_[res0], 14)
        assert_almost_equal(np.r_[res2], np.r_[res0], 14)
        res0 = d1w_d0.tconfint_mean()
        res1 = d1w_d1.tconfint_mean()
        res2 = d1w_d2.tconfint_mean()
        assert_almost_equal(res1, res0, 14)
        assert_almost_equal(res2, res0, 14)

    def test_comparemeans_convenient_interface(self):
        if False:
            i = 10
            return i + 15
        (x1_2d, x2_2d) = (self.x1_2d, self.x2_2d)
        d1 = DescrStatsW(x1_2d)
        d2 = DescrStatsW(x2_2d)
        cm1 = CompareMeans(d1, d2)
        from statsmodels.iolib.table import SimpleTable
        for use_t in [True, False]:
            for usevar in ['pooled', 'unequal']:
                smry = cm1.summary(use_t=use_t, usevar=usevar)
                assert_(isinstance(smry, SimpleTable))
        cm2 = CompareMeans.from_data(x1_2d, x2_2d)
        assert_(str(cm1.summary()) == str(cm2.summary()))

    def test_comparemeans_convenient_interface_1d(self):
        if False:
            return 10
        (x1_2d, x2_2d) = (self.x1, self.x2)
        d1 = DescrStatsW(x1_2d)
        d2 = DescrStatsW(x2_2d)
        cm1 = CompareMeans(d1, d2)
        from statsmodels.iolib.table import SimpleTable
        for use_t in [True, False]:
            for usevar in ['pooled', 'unequal']:
                smry = cm1.summary(use_t=use_t, usevar=usevar)
                assert_(isinstance(smry, SimpleTable))
        cm2 = CompareMeans.from_data(x1_2d, x2_2d)
        assert_(str(cm1.summary()) == str(cm2.summary()))

class CheckWeightstats1dMixin:

    def test_basic(self):
        if False:
            while True:
                i = 10
        x1r = self.x1r
        d1w = self.d1w
        assert_almost_equal(x1r.mean(0), d1w.mean, 14)
        assert_almost_equal(x1r.var(0, ddof=d1w.ddof), d1w.var, 14)
        assert_almost_equal(x1r.std(0, ddof=d1w.ddof), d1w.std, 14)
        var1 = d1w.var_ddof(ddof=1)
        assert_almost_equal(x1r.var(0, ddof=1), var1, 14)
        std1 = d1w.std_ddof(ddof=1)
        assert_almost_equal(x1r.std(0, ddof=1), std1, 14)
        assert_almost_equal(np.cov(x1r.T, bias=1 - d1w.ddof), d1w.cov, 14)

    def test_ttest(self):
        if False:
            while True:
                i = 10
        x1r = self.x1r
        d1w = self.d1w
        assert_almost_equal(d1w.ttest_mean(3)[:2], stats.ttest_1samp(x1r, 3), 11)

    def test_ttest_2sample(self):
        if False:
            return 10
        (x1, x2) = (self.x1, self.x2)
        (x1r, x2r) = (self.x1r, self.x2r)
        (w1, w2) = (self.w1, self.w2)
        res_sp = stats.ttest_ind(x1r, x2r)
        assert_almost_equal(ttest_ind(x1, x2, weights=(w1, w2))[:2], res_sp, 14)
        cm = CompareMeans(DescrStatsW(x1, weights=w1, ddof=0), DescrStatsW(x2, weights=w2, ddof=1))
        assert_almost_equal(cm.ttest_ind()[:2], res_sp, 14)
        cm = CompareMeans(DescrStatsW(x1, weights=w1, ddof=1), DescrStatsW(x2, weights=w2, ddof=2))
        assert_almost_equal(cm.ttest_ind()[:2], res_sp, 14)
        cm0 = CompareMeans(DescrStatsW(x1, weights=w1, ddof=0), DescrStatsW(x2, weights=w2, ddof=0))
        cm1 = CompareMeans(DescrStatsW(x1, weights=w1, ddof=0), DescrStatsW(x2, weights=w2, ddof=1))
        cm2 = CompareMeans(DescrStatsW(x1, weights=w1, ddof=1), DescrStatsW(x2, weights=w2, ddof=2))
        res0 = cm0.ttest_ind(usevar='unequal')
        res1 = cm1.ttest_ind(usevar='unequal')
        res2 = cm2.ttest_ind(usevar='unequal')
        assert_almost_equal(res1, res0, 14)
        assert_almost_equal(res2, res0, 14)
        res0 = cm0.tconfint_diff(usevar='pooled')
        res1 = cm1.tconfint_diff(usevar='pooled')
        res2 = cm2.tconfint_diff(usevar='pooled')
        assert_almost_equal(res1, res0, 14)
        assert_almost_equal(res2, res0, 14)
        res0 = cm0.tconfint_diff(usevar='unequal')
        res1 = cm1.tconfint_diff(usevar='unequal')
        res2 = cm2.tconfint_diff(usevar='unequal')
        assert_almost_equal(res1, res0, 14)
        assert_almost_equal(res2, res0, 14)

    def test_confint_mean(self):
        if False:
            return 10
        d1w = self.d1w
        alpha = 0.05
        (low, upp) = d1w.tconfint_mean()
        (t, p, d) = d1w.ttest_mean(low)
        assert_almost_equal(p, alpha * np.ones(p.shape), 8)
        (t, p, d) = d1w.ttest_mean(upp)
        assert_almost_equal(p, alpha * np.ones(p.shape), 8)
        (t, p, d) = d1w.ttest_mean(np.vstack((low, upp)))
        assert_almost_equal(p, alpha * np.ones(p.shape), 8)

class CheckWeightstats2dMixin(CheckWeightstats1dMixin):

    def test_corr(self):
        if False:
            i = 10
            return i + 15
        x1r = self.x1r
        d1w = self.d1w
        assert_almost_equal(np.corrcoef(x1r.T), d1w.corrcoef, 14)

class TestWeightstats1d_ddof(CheckWeightstats1dMixin):

    @classmethod
    def setup_class(cls):
        if False:
            for i in range(10):
                print('nop')
        np.random.seed(9876789)
        (n1, n2) = (20, 20)
        (m1, m2) = (1, 1.2)
        x1 = m1 + np.random.randn(n1, 1)
        x2 = m2 + np.random.randn(n2, 1)
        w1 = np.random.randint(1, 4, n1)
        w2 = np.random.randint(1, 4, n2)
        (cls.x1, cls.x2) = (x1, x2)
        (cls.w1, cls.w2) = (w1, w2)
        cls.d1w = DescrStatsW(x1, weights=w1, ddof=1)
        cls.d2w = DescrStatsW(x2, weights=w2, ddof=1)
        cls.x1r = cls.d1w.asrepeats()
        cls.x2r = cls.d2w.asrepeats()

class TestWeightstats2d(CheckWeightstats2dMixin):

    @classmethod
    def setup_class(cls):
        if False:
            print('Hello World!')
        np.random.seed(9876789)
        (n1, n2) = (20, 20)
        (m1, m2) = (1, 1.2)
        x1 = m1 + np.random.randn(n1, 3)
        x2 = m2 + np.random.randn(n2, 3)
        w1 = np.random.randint(1, 4, n1)
        w2 = np.random.randint(1, 4, n2)
        (cls.x1, cls.x2) = (x1, x2)
        (cls.w1, cls.w2) = (w1, w2)
        cls.d1w = DescrStatsW(x1, weights=w1)
        cls.d2w = DescrStatsW(x2, weights=w2)
        cls.x1r = cls.d1w.asrepeats()
        cls.x2r = cls.d2w.asrepeats()

class TestWeightstats2d_ddof(CheckWeightstats2dMixin):

    @classmethod
    def setup_class(cls):
        if False:
            for i in range(10):
                print('nop')
        np.random.seed(9876789)
        (n1, n2) = (20, 20)
        (m1, m2) = (1, 1.2)
        x1 = m1 + np.random.randn(n1, 3)
        x2 = m2 + np.random.randn(n2, 3)
        w1 = np.random.randint(1, 4, n1)
        w2 = np.random.randint(1, 4, n2)
        (cls.x1, cls.x2) = (x1, x2)
        (cls.w1, cls.w2) = (w1, w2)
        cls.d1w = DescrStatsW(x1, weights=w1, ddof=1)
        cls.d2w = DescrStatsW(x2, weights=w2, ddof=1)
        cls.x1r = cls.d1w.asrepeats()
        cls.x2r = cls.d2w.asrepeats()

class TestWeightstats2d_nobs(CheckWeightstats2dMixin):

    @classmethod
    def setup_class(cls):
        if False:
            print('Hello World!')
        np.random.seed(9876789)
        (n1, n2) = (20, 30)
        (m1, m2) = (1, 1.2)
        x1 = m1 + np.random.randn(n1, 3)
        x2 = m2 + np.random.randn(n2, 3)
        w1 = np.random.randint(1, 4, n1)
        w2 = np.random.randint(1, 4, n2)
        (cls.x1, cls.x2) = (x1, x2)
        (cls.w1, cls.w2) = (w1, w2)
        cls.d1w = DescrStatsW(x1, weights=w1, ddof=0)
        cls.d2w = DescrStatsW(x2, weights=w2, ddof=1)
        cls.x1r = cls.d1w.asrepeats()
        cls.x2r = cls.d2w.asrepeats()

def test_ttest_ind_with_uneq_var():
    if False:
        return 10
    a = (1, 2, 3)
    b = (1.1, 2.9, 4.2)
    pr = 0.5361949075312673
    tr = -0.6864951273557258
    (t, p, df) = ttest_ind(a, b, usevar='unequal')
    assert_almost_equal([t, p], [tr, pr], 13)
    a = (1, 2, 3, 4)
    pr = 0.8435413913160829
    tr = -0.2108663315950719
    (t, p, df) = ttest_ind(a, b, usevar='unequal')
    assert_almost_equal([t, p], [tr, pr], 13)

def test_ztest_ztost():
    if False:
        i = 10
        return i + 15
    import statsmodels.stats.proportion as smprop
    x1 = [0, 1]
    w1 = [5, 15]
    res2 = smprop.proportions_ztest(15, 20.0, value=0.5)
    d1 = DescrStatsW(x1, w1)
    res1 = d1.ztest_mean(0.5)
    assert_allclose(res1, res2, rtol=0.03, atol=0.003)
    d2 = DescrStatsW(x1, np.array(w1) * 21.0 / 20)
    res1 = d2.ztest_mean(0.5)
    assert_almost_equal(res1, res2, decimal=12)
    res1 = d2.ztost_mean(0.4, 0.6)
    res2 = smprop.proportions_ztost(15, 20.0, 0.4, 0.6)
    assert_almost_equal(res1[0], res2[0], decimal=12)
    x2 = [0, 1]
    w2 = [10, 10]
    d2 = DescrStatsW(x2, w2)
    res1 = ztest(d1.asrepeats(), d2.asrepeats())
    res2 = smprop.proportions_chisquare(np.asarray([15, 10]), np.asarray([20.0, 20]))
    assert_allclose(res1[1], res2[1], rtol=0.03)
    res1a = CompareMeans(d1, d2).ztest_ind()
    assert_allclose(res1a[1], res2[1], rtol=0.03)
    assert_almost_equal(res1a, res1, decimal=12)
ztest_ = Holder()
ztest_.statistic = 6.55109865675183
ztest_.p_value = 5.711530850508982e-11
ztest_.conf_int = np.array([1.230415246535603, 2.280948389828034])
ztest_.estimate = np.array([7.01818181818182, 5.2625])
ztest_.null_value = 0
ztest_.alternative = 'two.sided'
ztest_.method = 'Two-sample z-Test'
ztest_.data_name = 'x and y'
ztest_smaller = Holder()
ztest_smaller.statistic = 6.55109865675183
ztest_smaller.p_value = 0.999999999971442
ztest_smaller.conf_int = np.array([np.nan, 2.196499421109045])
ztest_smaller.estimate = np.array([7.01818181818182, 5.2625])
ztest_smaller.null_value = 0
ztest_smaller.alternative = 'less'
ztest_smaller.method = 'Two-sample z-Test'
ztest_smaller.data_name = 'x and y'
ztest_larger = Holder()
ztest_larger.statistic = 6.55109865675183
ztest_larger.p_value = 2.855760072861813e-11
ztest_larger.conf_int = np.array([1.314864215254592, np.nan])
ztest_larger.estimate = np.array([7.01818181818182, 5.2625])
ztest_larger.null_value = 0
ztest_larger.alternative = 'greater'
ztest_larger.method = 'Two-sample z-Test'
ztest_larger.data_name = 'x and y'
ztest_mu = Holder()
ztest_mu.statistic = 2.81972854805176
ztest_mu.p_value = 0.00480642898427981
ztest_mu.conf_int = np.array([1.230415246535603, 2.280948389828034])
ztest_mu.estimate = np.array([7.01818181818182, 5.2625])
ztest_mu.null_value = 1
ztest_mu.alternative = 'two.sided'
ztest_mu.method = 'Two-sample z-Test'
ztest_mu.data_name = 'x and y'
ztest_larger_mu = Holder()
ztest_larger_mu.statistic = 2.81972854805176
ztest_larger_mu.p_value = 0.002403214492139871
ztest_larger_mu.conf_int = np.array([1.314864215254592, np.nan])
ztest_larger_mu.estimate = np.array([7.01818181818182, 5.2625])
ztest_larger_mu.null_value = 1
ztest_larger_mu.alternative = 'greater'
ztest_larger_mu.method = 'Two-sample z-Test'
ztest_larger_mu.data_name = 'x and y'
ztest_smaller_mu = Holder()
ztest_smaller_mu.statistic = -0.911641560648313
ztest_smaller_mu.p_value = 0.1809787183191324
ztest_smaller_mu.conf_int = np.array([np.nan, 2.196499421109045])
ztest_smaller_mu.estimate = np.array([7.01818181818182, 5.2625])
ztest_smaller_mu.null_value = 2
ztest_smaller_mu.alternative = 'less'
ztest_smaller_mu.method = 'Two-sample z-Test'
ztest_smaller_mu.data_name = 'x and y'
ztest_mu_1s = Holder()
ztest_mu_1s.statistic = 4.415212090914452
ztest_mu_1s.p_value = 1.009110038015147e-05
ztest_mu_1s.conf_int = np.array([6.74376372125119, 7.29259991511245])
ztest_mu_1s.estimate = 7.01818181818182
ztest_mu_1s.null_value = 6.4
ztest_mu_1s.alternative = 'two.sided'
ztest_mu_1s.method = 'One-sample z-Test'
ztest_mu_1s.data_name = 'x'
ztest_smaller_mu_1s = Holder()
ztest_smaller_mu_1s.statistic = -2.727042762035397
ztest_smaller_mu_1s.p_value = 0.00319523783881176
ztest_smaller_mu_1s.conf_int = np.array([np.nan, 7.248480744895716])
ztest_smaller_mu_1s.estimate = 7.01818181818182
ztest_smaller_mu_1s.null_value = 7.4
ztest_smaller_mu_1s.alternative = 'less'
ztest_smaller_mu_1s.method = 'One-sample z-Test'
ztest_smaller_mu_1s.data_name = 'x'
ztest_larger_mu_1s = Holder()
ztest_larger_mu_1s.statistic = 4.415212090914452
ztest_larger_mu_1s.p_value = 5.045550190097003e-06
ztest_larger_mu_1s.conf_int = np.array([6.78788289146792, np.nan])
ztest_larger_mu_1s.estimate = 7.01818181818182
ztest_larger_mu_1s.null_value = 6.4
ztest_larger_mu_1s.alternative = 'greater'
ztest_larger_mu_1s.method = 'One-sample z-Test'
ztest_larger_mu_1s.data_name = 'x'
ztest_unequal = Holder()
ztest_unequal.statistic = 6.12808151466544
ztest_unequal.p_value = 8.89450168270109e-10
ztest_unequal.conf_int = np.array([1.19415646579981, 2.31720717056382])
ztest_unequal.estimate = np.array([7.01818181818182, 5.2625])
ztest_unequal.null_value = 0
ztest_unequal.alternative = 'two.sided'
ztest_unequal.usevar = 'unequal'
ztest_unequal.method = 'Two-sample z-Test'
ztest_unequal.data_name = 'x and y'
ztest_smaller_unequal = Holder()
ztest_smaller_unequal.statistic = 6.12808151466544
ztest_smaller_unequal.p_value = 0.999999999555275
ztest_smaller_unequal.conf_int = np.array([np.nan, 2.22692874913371])
ztest_smaller_unequal.estimate = np.array([7.01818181818182, 5.2625])
ztest_smaller_unequal.null_value = 0
ztest_smaller_unequal.alternative = 'less'
ztest_smaller_unequal.usevar = 'unequal'
ztest_smaller_unequal.method = 'Two-sample z-Test'
ztest_smaller_unequal.data_name = 'x and y'
ztest_larger_unequal = Holder()
ztest_larger_unequal.statistic = 6.12808151466544
ztest_larger_unequal.p_value = 4.44725034576265e-10
ztest_larger_unequal.conf_int = np.array([1.28443488722992, np.nan])
ztest_larger_unequal.estimate = np.array([7.01818181818182, 5.2625])
ztest_larger_unequal.null_value = 0
ztest_larger_unequal.alternative = 'greater'
ztest_larger_unequal.usevar = 'unequal'
ztest_larger_unequal.method = 'Two-sample z-Test'
ztest_larger_unequal.data_name = 'x and y'
alternatives = {'less': 'smaller', 'greater': 'larger', 'two.sided': 'two-sided'}

class TestZTest:

    @classmethod
    def setup_class(cls):
        if False:
            i = 10
            return i + 15
        cls.x1 = np.array([7.8, 6.6, 6.5, 7.4, 7.3, 7.0, 6.4, 7.1, 6.7, 7.6, 6.8])
        cls.x2 = np.array([4.5, 5.4, 6.1, 6.1, 5.4, 5.0, 4.1, 5.5])
        cls.d1 = DescrStatsW(cls.x1)
        cls.d2 = DescrStatsW(cls.x2)
        cls.cm = CompareMeans(cls.d1, cls.d2)

    def test(self):
        if False:
            i = 10
            return i + 15
        (x1, x2) = (self.x1, self.x2)
        cm = self.cm
        for tc in [ztest_, ztest_smaller, ztest_larger, ztest_mu, ztest_smaller_mu, ztest_larger_mu]:
            (zstat, pval) = ztest(x1, x2, value=tc.null_value, alternative=alternatives[tc.alternative])
            assert_allclose(zstat, tc.statistic, rtol=1e-10)
            assert_allclose(pval, tc.p_value, rtol=1e-10, atol=1e-16)
            (zstat, pval) = cm.ztest_ind(value=tc.null_value, alternative=alternatives[tc.alternative])
            assert_allclose(zstat, tc.statistic, rtol=1e-10)
            assert_allclose(pval, tc.p_value, rtol=1e-10, atol=1e-16)
            tc_conf_int = tc.conf_int.copy()
            if np.isnan(tc_conf_int[0]):
                tc_conf_int[0] = -np.inf
            if np.isnan(tc_conf_int[1]):
                tc_conf_int[1] = np.inf
            ci = zconfint(x1, x2, value=0, alternative=alternatives[tc.alternative])
            assert_allclose(ci, tc_conf_int, rtol=1e-10)
            ci = cm.zconfint_diff(alternative=alternatives[tc.alternative])
            assert_allclose(ci, tc_conf_int, rtol=1e-10)
            ci = zconfint(x1, x2, value=tc.null_value, alternative=alternatives[tc.alternative])
            assert_allclose(ci, tc_conf_int - tc.null_value, rtol=1e-10)
        for tc in [ztest_unequal, ztest_smaller_unequal, ztest_larger_unequal]:
            (zstat, pval) = ztest(x1, x2, value=tc.null_value, alternative=alternatives[tc.alternative], usevar='unequal')
            assert_allclose(zstat, tc.statistic, rtol=1e-10)
            assert_allclose(pval, tc.p_value, rtol=1e-10, atol=1e-16)
        d1 = self.d1
        for tc in [ztest_mu_1s, ztest_smaller_mu_1s, ztest_larger_mu_1s]:
            (zstat, pval) = ztest(x1, value=tc.null_value, alternative=alternatives[tc.alternative])
            assert_allclose(zstat, tc.statistic, rtol=1e-10)
            assert_allclose(pval, tc.p_value, rtol=1e-10, atol=1e-16)
            (zstat, pval) = d1.ztest_mean(value=tc.null_value, alternative=alternatives[tc.alternative])
            assert_allclose(zstat, tc.statistic, rtol=1e-10)
            assert_allclose(pval, tc.p_value, rtol=1e-10, atol=1e-16)
            tc_conf_int = tc.conf_int.copy()
            if np.isnan(tc_conf_int[0]):
                tc_conf_int[0] = -np.inf
            if np.isnan(tc_conf_int[1]):
                tc_conf_int[1] = np.inf
            ci = zconfint(x1, value=0, alternative=alternatives[tc.alternative])
            assert_allclose(ci, tc_conf_int, rtol=1e-10)
            ci = d1.zconfint_mean(alternative=alternatives[tc.alternative])
            assert_allclose(ci, tc_conf_int, rtol=1e-10)

def test_weightstats_len_1():
    if False:
        i = 10
        return i + 15
    x1 = [1]
    w1 = [1]
    d1 = DescrStatsW(x1, w1)
    assert (d1.quantile([0.0, 0.5, 1.0]) == 1).all()

def test_weightstats_2d_w1():
    if False:
        print('Hello World!')
    x1 = [[1], [2]]
    w1 = [[1], [2]]
    d1 = DescrStatsW(x1, w1)
    print(len(np.array(w1).shape))
    assert (d1.quantile([0.5, 1.0]) == 2).all().all()

def test_weightstats_2d_w2():
    if False:
        while True:
            i = 10
    x1 = [[1]]
    w1 = [[1]]
    d1 = DescrStatsW(x1, w1)
    assert (d1.quantile([0, 0.5, 1.0]) == 1).all().all()