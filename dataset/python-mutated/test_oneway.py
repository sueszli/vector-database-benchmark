"""
Created on Wed Mar 18 17:45:51 2020

Author: Josef Perktold
License: BSD-3

"""
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest
from statsmodels.regression.linear_model import OLS
import statsmodels.stats.power as smpwr
import statsmodels.stats.oneway as smo
from statsmodels.stats.oneway import confint_effectsize_oneway, confint_noncentrality, effectsize_oneway, anova_oneway, anova_generic, equivalence_oneway, equivalence_oneway_generic, power_equivalence_oneway, _power_equivalence_oneway_emp, f2_to_wellek, fstat_to_wellek, wellek_to_f2
from statsmodels.stats.robust_compare import scale_transform
from statsmodels.stats.contrast import wald_test_noncent_generic, wald_test_noncent, _offset_constraint

def test_oneway_effectsize():
    if False:
        i = 10
        return i + 15
    F = 5
    df1 = 3
    df2 = 76
    nobs = 80
    ci = confint_noncentrality(F, (df1, df2), alpha=0.05, alternative='two-sided')
    ci_es = confint_effectsize_oneway(F, (df1, df2), alpha=0.05)
    ci_steiger = ci_es.ci_f * np.sqrt(4 / 3)
    res_ci_steiger = [0.1764, 0.7367]
    res_ci_nc = np.asarray([1.8666, 32.563])
    assert_allclose(ci, res_ci_nc, atol=0.0001)
    assert_allclose(ci_es.ci_f_corrected, res_ci_steiger, atol=6e-05)
    assert_allclose(ci_steiger, res_ci_steiger, atol=6e-05)
    assert_allclose(ci_es.ci_f ** 2, res_ci_nc / nobs, atol=6e-05)
    assert_allclose(ci_es.ci_nc, res_ci_nc, atol=0.0001)

def test_effectsize_power():
    if False:
        i = 10
        return i + 15
    n_groups = 3
    means = [527.86, 660.43, 649.14]
    vars_ = 107.4304 ** 2
    nobs = 12
    es = effectsize_oneway(means, vars_, nobs, use_var='equal', ddof_between=0)
    es = np.sqrt(es)
    alpha = 0.05
    power = 0.8
    nobs_t = nobs * n_groups
    kwds = {'effect_size': es, 'nobs': nobs_t, 'alpha': alpha, 'power': power, 'k_groups': n_groups}
    from statsmodels.stats.power import FTestAnovaPower
    res_pow = 0.8251
    res_es = 0.559
    kwds_ = kwds.copy()
    del kwds_['power']
    p = FTestAnovaPower().power(**kwds_)
    assert_allclose(p, res_pow, atol=0.0001)
    assert_allclose(es, res_es, atol=0.0006)
    nobs = np.array([15, 9, 9])
    kwds['nobs'] = nobs
    es = effectsize_oneway(means, vars_, nobs, use_var='equal', ddof_between=0)
    es = np.sqrt(es)
    kwds['effect_size'] = es
    p = FTestAnovaPower().power(**kwds_)
    res_pow = 0.8297
    res_es = 0.59
    assert_allclose(p, res_pow, atol=0.005)
    assert_allclose(es, res_es, atol=0.0006)

def test_effectsize_fstat():
    if False:
        print('Hello World!')
    Eta_Sq_partial = 0.796983758700696
    CI_eta2 = (0.685670133284926, 0.855981325777856)
    Epsilon_Sq_partial = 0.779582366589327
    CI_eps2 = (0.658727573280777, 0.843636867987386)
    Omega_Sq_partial = 0.775086505190311
    CI_omega2 = (0.65286429480169, 0.840179680453464)
    Cohens_f_partial = 1.98134153686695
    CI_f = (1.47694659580859, 2.43793847155554)
    (f_stat, df1, df2) = (45.8, 3, 35)
    fes = smo._fstat2effectsize(f_stat, (df1, df2))
    assert_allclose(np.sqrt(fes.f2), Cohens_f_partial, rtol=1e-13)
    assert_allclose(fes.eta2, Eta_Sq_partial, rtol=1e-13)
    assert_allclose(fes.eps2, Epsilon_Sq_partial, rtol=1e-13)
    assert_allclose(fes.omega2, Omega_Sq_partial, rtol=1e-13)
    ci_nc = confint_noncentrality(f_stat, (df1, df2), alpha=0.1)
    ci_es = smo._fstat2effectsize(ci_nc / df1, (df1, df2))
    assert_allclose(ci_es.eta2, CI_eta2, rtol=0.0002)
    assert_allclose(ci_es.eps2, CI_eps2, rtol=0.0002)
    assert_allclose(ci_es.omega2, CI_omega2, rtol=0.0002)
    assert_allclose(np.sqrt(ci_es.f2), CI_f, rtol=0.0002)

def test_effectsize_fstat_stata():
    if False:
        while True:
            i = 10
    eta2 = 0.2720398648288652
    lb_eta2 = 0.0742092468714613
    ub_eta2 = 0.4156116886974804
    omega2 = 0.2356418580703085
    lb_omega2 = 0.0279197092150344
    ub_omega2 = 0.3863922731323545
    (f_stat, df1, df2) = (7.47403193349075, 2, 40)
    fes = smo._fstat2effectsize(f_stat, (df1, df2))
    assert_allclose(fes.eta2, eta2, rtol=1e-13)
    assert_allclose(fes.omega2, omega2, rtol=0.02)
    ci_es = smo.confint_effectsize_oneway(f_stat, (df1, df2), alpha=0.1)
    assert_allclose(ci_es.eta2, (lb_eta2, ub_eta2), rtol=0.0001)
    assert_allclose(ci_es.ci_omega2, (lb_omega2, ub_omega2), rtol=0.025)

@pytest.mark.parametrize('center', ['median', 'mean', 'trimmed'])
def test_scale_transform(center):
    if False:
        for i in range(10):
            print('nop')
    x = np.random.randn(5, 3)
    xt = scale_transform(x, center=center, transform='abs', trim_frac=0.2, axis=0)
    xtt = scale_transform(x.T, center=center, transform='abs', trim_frac=0.2, axis=1)
    assert_allclose(xt.T, xtt, rtol=1e-13)
    xt0 = scale_transform(x[:, 0], center=center, transform='abs', trim_frac=0.2)
    assert_allclose(xt0, xt[:, 0], rtol=1e-13)
    assert_allclose(xt0, xtt[0, :], rtol=1e-13)

class TestOnewayEquivalenc:

    @classmethod
    def setup_class(cls):
        if False:
            while True:
                i = 10
        y0 = [112.488, 103.738, 86.344, 101.708, 95.108, 105.931, 95.815, 91.864, 102.479, 102.644]
        y1 = [100.421, 101.966, 99.636, 105.983, 88.377, 102.618, 105.486, 98.662, 94.137, 98.626, 89.367, 106.204]
        y2 = [84.846, 100.488, 119.763, 103.736, 93.141, 108.254, 99.51, 89.005, 108.2, 82.209, 100.104, 103.706, 107.067]
        y3 = [100.825, 100.255, 103.363, 93.23, 95.325, 100.288, 94.75, 107.129, 98.246, 96.365, 99.74, 106.049, 92.691, 93.111, 98.243]
        n_groups = 4
        arrs_w = [np.asarray(yi) for yi in [y0, y1, y2, y3]]
        nobs = np.asarray([len(yi) for yi in arrs_w])
        nobs_mean = np.mean(nobs)
        means = np.asarray([yi.mean() for yi in arrs_w])
        stds = np.asarray([yi.std(ddof=1) for yi in arrs_w])
        cls.data = arrs_w
        cls.means = means
        cls.nobs = nobs
        cls.stds = stds
        cls.n_groups = n_groups
        cls.nobs_mean = nobs_mean

    def test_equivalence_equal(self):
        if False:
            print('Hello World!')
        means = self.means
        nobs = self.nobs
        stds = self.stds
        n_groups = self.n_groups
        eps = 0.5
        res0 = anova_generic(means, stds ** 2, nobs, use_var='equal')
        f = res0.statistic
        res = equivalence_oneway_generic(f, n_groups, nobs.sum(), eps, res0.df, alpha=0.05, margin_type='wellek')
        assert_allclose(res.pvalue, 0.0083, atol=0.001)
        assert_equal(res.df, [3, 46])
        assert_allclose(f, 0.0926, atol=0.0006)
        res = equivalence_oneway(self.data, eps, use_var='equal', margin_type='wellek')
        assert_allclose(res.pvalue, 0.0083, atol=0.001)
        assert_equal(res.df, [3, 46])

    def test_equivalence_welch(self):
        if False:
            for i in range(10):
                print('nop')
        means = self.means
        nobs = self.nobs
        stds = self.stds
        n_groups = self.n_groups
        vars_ = stds ** 2
        eps = 0.5
        res0 = anova_generic(means, vars_, nobs, use_var='unequal', welch_correction=False)
        f_stat = res0.statistic
        res = equivalence_oneway_generic(f_stat, n_groups, nobs.sum(), eps, res0.df, alpha=0.05, margin_type='wellek')
        assert_allclose(res.pvalue, 0.011, atol=0.001)
        assert_allclose(res.df, [3.0, 22.6536], atol=0.0006)
        assert_allclose(f_stat, 0.1102, atol=0.007)
        res = equivalence_oneway(self.data, eps, use_var='unequal', margin_type='wellek')
        assert_allclose(res.pvalue, 0.011, atol=0.0001)
        assert_allclose(res.df, [3.0, 22.6536], atol=0.0006)
        assert_allclose(res.f_stat, 0.1102, atol=0.0001)
        pow_ = _power_equivalence_oneway_emp(f_stat, n_groups, nobs, eps, res0.df)
        assert_allclose(pow_, 0.1552, atol=0.007)
        pow_ = power_equivalence_oneway(eps, eps, nobs.sum(), n_groups=n_groups, df=None, alpha=0.05, margin_type='wellek')
        assert_allclose(pow_, 0.05, atol=1e-13)
        nobs_t = nobs.sum()
        es = effectsize_oneway(means, vars_, nobs, use_var='unequal')
        es = np.sqrt(es)
        es_w0 = f2_to_wellek(es ** 2, n_groups)
        es_w = np.sqrt(fstat_to_wellek(f_stat, n_groups, nobs_t / n_groups))
        pow_ = power_equivalence_oneway(es_w, eps, nobs_t, n_groups=n_groups, df=None, alpha=0.05, margin_type='wellek')
        assert_allclose(pow_, 0.1552, atol=0.007)
        assert_allclose(es_w0, es_w, atol=0.007)
        margin = wellek_to_f2(eps, n_groups)
        pow_ = power_equivalence_oneway(es ** 2, margin, nobs_t, n_groups=n_groups, df=None, alpha=0.05, margin_type='f2')
        assert_allclose(pow_, 0.1552, atol=0.007)

class TestOnewayScale:

    @classmethod
    def setup_class(cls):
        if False:
            for i in range(10):
                print('nop')
        yt0 = np.array([102.0, 320.0, 0.0, 107.0, 198.0, 200.0, 4.0, 20.0, 110.0, 128.0, 7.0, 119.0, 309.0])
        yt1 = np.array([0.0, 1.0, 228.0, 81.0, 87.0, 119.0, 79.0, 181.0, 43.0, 12.0, 90.0, 105.0, 108.0, 119.0, 0.0, 9.0])
        yt2 = np.array([33.0, 294.0, 134.0, 216.0, 83.0, 105.0, 69.0, 20.0, 20.0, 63.0, 98.0, 155.0, 78.0, 75.0])
        y0 = np.array([452.0, 874.0, 554.0, 447.0, 356.0, 754.0, 558.0, 574.0, 664.0, 682.0, 547.0, 435.0, 245.0])
        y1 = np.array([546.0, 547.0, 774.0, 465.0, 459.0, 665.0, 467.0, 365.0, 589.0, 534.0, 456.0, 651.0, 654.0, 665.0, 546.0, 537.0])
        y2 = np.array([785.0, 458.0, 886.0, 536.0, 669.0, 857.0, 821.0, 772.0, 732.0, 689.0, 654.0, 597.0, 830.0, 827.0])
        n_groups = 3
        data = [y0, y1, y2]
        nobs = np.asarray([len(yi) for yi in data])
        nobs_mean = np.mean(nobs)
        means = np.asarray([yi.mean() for yi in data])
        stds = np.asarray([yi.std(ddof=1) for yi in data])
        cls.data = data
        cls.data_transformed = [yt0, yt1, yt2]
        cls.means = means
        cls.nobs = nobs
        cls.stds = stds
        cls.n_groups = n_groups
        cls.nobs_mean = nobs_mean

    def test_means(self):
        if False:
            print('Hello World!')
        statistic = 7.10900606421182
        parameter = [2, 31.4207256105052]
        p_value = 0.00283841965791224
        res = anova_oneway(self.data, use_var='bf')
        assert_allclose(res.pvalue2, p_value, rtol=1e-13)
        assert_allclose(res.statistic, statistic, rtol=1e-13)
        assert_allclose([res.df_num2, res.df_denom], parameter)

    def test_levene(self):
        if False:
            return 10
        data = self.data
        statistic = 1.0866123063642
        p_value = 0.3471072204516
        res0 = smo.test_scale_oneway(data, method='equal', center='median', transform='abs', trim_frac_mean=0.2)
        assert_allclose(res0.pvalue, p_value, rtol=1e-13)
        assert_allclose(res0.statistic, statistic, rtol=1e-13)
        statistic = 1.10732113109744
        p_value = 0.340359251994645
        df = [2, 40]
        res0 = smo.test_scale_oneway(data, method='equal', center='trimmed', transform='abs', trim_frac_mean=0.2)
        assert_allclose(res0.pvalue, p_value, rtol=1e-13)
        assert_allclose(res0.statistic, statistic, rtol=1e-13)
        assert_allclose(res0.df, df)
        statistic = 1.07894485177512
        parameter = [2, 40]
        p_value = 0.349641166869223
        res0 = smo.test_scale_oneway(data, method='equal', center='mean', transform='abs', trim_frac_mean=0.2)
        assert_allclose(res0.pvalue, p_value, rtol=1e-13)
        assert_allclose(res0.statistic, statistic, rtol=1e-13)
        assert_allclose(res0.df, parameter)
        statistic = 3.01982414477323
        p_value = 0.220929402900495
        from scipy import stats
        (stat, pv) = stats.bartlett(*data)
        assert_allclose(pv, p_value, rtol=1e-13)
        assert_allclose(stat, statistic, rtol=1e-13)

    def test_options(self):
        if False:
            while True:
                i = 10
        data = self.data
        (statistic, p_value) = (1.0173464626246675, 0.3763806150460239)
        df = (2.0, 24.40374758005409)
        res = smo.test_scale_oneway(data, method='unequal', center='median', transform='abs', trim_frac_mean=0.2)
        assert_allclose(res.pvalue, p_value, rtol=1e-13)
        assert_allclose(res.statistic, statistic, rtol=1e-13)
        assert_allclose(res.df, df)
        (statistic, p_value) = (1.0329722145270606, 0.3622778213868562)
        df = (1.83153791573948, 30.6733640949525)
        p_value2 = 0.3679999679787619
        df2 = (2, 30.6733640949525)
        res = smo.test_scale_oneway(data, method='bf', center='median', transform='abs', trim_frac_mean=0.2)
        assert_allclose(res.pvalue, p_value, rtol=1e-13)
        assert_allclose(res.statistic, statistic, rtol=1e-13)
        assert_allclose(res.df, df)
        assert_allclose(res.pvalue2, p_value2, rtol=1e-13)
        assert_allclose(res.df2, df2)
        (statistic, p_value) = (1.7252431333701745, 0.19112038168209514)
        df = (2.0, 40.0)
        res = smo.test_scale_oneway(data, method='equal', center='mean', transform='square', trim_frac_mean=0.2)
        assert_allclose(res.pvalue, p_value, rtol=1e-13)
        assert_allclose(res.statistic, statistic, rtol=1e-13)
        assert_equal(res.df, df)
        (statistic, p_value) = (0.4129696057329463, 0.6644711582864451)
        df = (2.0, 40.0)
        res = smo.test_scale_oneway(data, method='equal', center='mean', transform=lambda x: np.log(x * x), trim_frac_mean=0.2)
        assert_allclose(res.pvalue, p_value, rtol=1e-13)
        assert_allclose(res.statistic, statistic, rtol=1e-13)
        assert_allclose(res.df, df)
        res = smo.test_scale_oneway(data, method='unequal', center=0, transform='identity', trim_frac_mean=0.2)
        res2 = anova_oneway(self.data, use_var='unequal')
        assert_allclose(res.pvalue, res2.pvalue, rtol=1e-13)
        assert_allclose(res.statistic, res2.statistic, rtol=1e-13)
        assert_allclose(res.df, res2.df)

    def test_equivalence(self):
        if False:
            for i in range(10):
                print('nop')
        data = self.data
        res = smo.equivalence_scale_oneway(data, 0.5, method='unequal', center=0, transform='identity')
        res2 = equivalence_oneway(self.data, 0.5, use_var='unequal')
        assert_allclose(res.pvalue, res2.pvalue, rtol=1e-13)
        assert_allclose(res.statistic, res2.statistic, rtol=1e-13)
        assert_allclose(res.df, res2.df)
        res = smo.equivalence_scale_oneway(data, 0.5, method='bf', center=0, transform='identity')
        res2 = equivalence_oneway(self.data, 0.5, use_var='bf')
        assert_allclose(res.pvalue, res2.pvalue, rtol=1e-13)
        assert_allclose(res.statistic, res2.statistic, rtol=1e-13)
        assert_allclose(res.df, res2.df)

class TestOnewayOLS:

    @classmethod
    def setup_class(cls):
        if False:
            while True:
                i = 10
        y0 = [112.488, 103.738, 86.344, 101.708, 95.108, 105.931, 95.815, 91.864, 102.479, 102.644]
        y1 = [100.421, 101.966, 99.636, 105.983, 88.377, 102.618, 105.486, 98.662, 94.137, 98.626, 89.367, 106.204]
        y2 = [84.846, 100.488, 119.763, 103.736, 93.141, 108.254, 99.51, 89.005, 108.2, 82.209, 100.104, 103.706, 107.067]
        y3 = [100.825, 100.255, 103.363, 93.23, 95.325, 100.288, 94.75, 107.129, 98.246, 96.365, 99.74, 106.049, 92.691, 93.111, 98.243]
        cls.k_groups = k = 4
        cls.data = data = [y0, y1, y2, y3]
        cls.nobs = nobs = np.asarray([len(yi) for yi in data])
        groups = np.repeat(np.arange(k), nobs)
        cls.ex = (groups[:, None] == np.arange(k)).astype(np.int64)
        cls.y = np.concatenate(data)

    def test_ols_noncentrality(self):
        if False:
            print('Hello World!')
        k = self.k_groups
        res_ols = OLS(self.y, self.ex).fit()
        nobs_t = res_ols.model.nobs
        c_equal = -np.eye(k)[1:]
        c_equal[:, 0] = 1
        v = np.zeros(c_equal.shape[0])
        wt = res_ols.wald_test(c_equal, scalar=True)
        (df_num, df_denom) = (wt.df_num, wt.df_denom)
        cov_p = res_ols.cov_params()
        nc_wt = wald_test_noncent_generic(res_ols.params, c_equal, v, cov_p, diff=None, joint=True)
        assert_allclose(nc_wt, wt.statistic * wt.df_num, rtol=1e-13)
        nc_wt2 = wald_test_noncent(res_ols.params, c_equal, v, res_ols, diff=None, joint=True)
        assert_allclose(nc_wt2, nc_wt, rtol=1e-13)
        es_ols = nc_wt / nobs_t
        es_oneway = smo.effectsize_oneway(res_ols.params, res_ols.scale, self.nobs, use_var='equal')
        assert_allclose(es_ols, es_oneway, rtol=1e-13)
        alpha = 0.05
        pow_ols = smpwr.ftest_power(np.sqrt(es_ols), df_denom, df_num, alpha, ncc=1)
        pow_oneway = smpwr.ftest_anova_power(np.sqrt(es_oneway), nobs_t, alpha, k_groups=k, df=None)
        assert_allclose(pow_ols, pow_oneway, rtol=1e-13)
        params_alt = res_ols.params * 0.75
        v_off = _offset_constraint(c_equal, res_ols.params, params_alt)
        wt_off = res_ols.wald_test((c_equal, v + v_off), scalar=True)
        nc_wt_off = wald_test_noncent_generic(params_alt, c_equal, v, cov_p, diff=None, joint=True)
        assert_allclose(nc_wt_off, wt_off.statistic * wt_off.df_num, rtol=1e-13)
        nc_wt_vec = wald_test_noncent_generic(params_alt, c_equal, v, cov_p, diff=None, joint=False)
        for i in range(c_equal.shape[0]):
            nc_wt_i = wald_test_noncent_generic(params_alt, c_equal[i:i + 1], v[i:i + 1], cov_p, diff=None, joint=False)
            assert_allclose(nc_wt_vec[i], nc_wt_i, rtol=1e-13)

def test_simulate_equivalence():
    if False:
        i = 10
        return i + 15
    k_groups = 4
    k_repl = 10
    nobs = np.array([10, 12, 13, 15]) * k_repl
    means = np.array([-1, 0, 0, 1]) * 0.12
    vars_ = np.array([1, 2, 3, 4])
    nobs_t = nobs.sum()
    eps = 0.0191 * 10
    opt_var = ['unequal', 'equal', 'bf']
    k_mc = 100
    np.random.seed(987126)
    res_mc = smo.simulate_power_equivalence_oneway(means, nobs, eps, vars_=vars_, k_mc=k_mc, trim_frac=0.1, options_var=opt_var, margin_type='wellek')
    frac_reject = (res_mc.pvalue <= 0.05).sum(0) / k_mc
    assert_allclose(frac_reject, [0.17, 0.18, 0.14], atol=0.001)
    es_alt_li = []
    for uv in opt_var:
        es = effectsize_oneway(means, vars_, nobs, use_var=uv)
        es_alt_li.append(es)
    margin = wellek_to_f2(eps, k_groups)
    pow_ = [power_equivalence_oneway(es_, margin, nobs_t, n_groups=k_groups, df=None, alpha=0.05, margin_type='f2') for es_ in es_alt_li]
    assert_allclose(pow_, [0.147749, 0.173358, 0.177412], atol=0.007)