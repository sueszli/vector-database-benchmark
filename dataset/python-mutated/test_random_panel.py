"""Test for short_panel and panel sandwich

Created on Fri May 18 13:05:47 2012

Author: Josef Perktold

moved example from main of random_panel
"""
import numpy as np
from numpy.testing import assert_almost_equal
import numpy.testing as npt
import statsmodels.tools.eval_measures as em
from statsmodels.stats.moment_helpers import cov2corr, se_cov
from statsmodels.regression.linear_model import OLS
from statsmodels.sandbox.panel.panel_short import ShortPanelGLS, ShortPanelGLS2
from statsmodels.sandbox.panel.random_panel import PanelSample
import statsmodels.sandbox.panel.correlation_structures as cs
import statsmodels.stats.sandwich_covariance as sw

def assert_maxabs(actual, expected, value):
    if False:
        while True:
            i = 10
    npt.assert_array_less(em.maxabs(actual, expected, None), value)

def test_short_panel():
    if False:
        i = 10
        return i + 15
    nobs = 100
    nobs_i = 5
    n_groups = nobs // nobs_i
    k_vars = 3
    dgp = PanelSample(nobs, k_vars, n_groups, corr_structure=cs.corr_arma, corr_args=([1], [1.0, -0.9]), seed=377769, within=False)
    y = dgp.generate_panel()
    noise = y - dgp.y_true
    dgp_cov_e = np.array([[1.0, 0.9, 0.81, 0.729, 0.6561], [0.9, 1.0, 0.9, 0.81, 0.729], [0.81, 0.9, 1.0, 0.9, 0.81], [0.729, 0.81, 0.9, 1.0, 0.9], [0.6561, 0.729, 0.81, 0.9, 1.0]])
    npt.assert_almost_equal(dgp.cov, dgp_cov_e, 13)
    cov_noise = np.cov(noise.reshape(-1, n_groups, order='F'))
    corr_noise = cov2corr(cov_noise)
    npt.assert_almost_equal(corr_noise, dgp.cov, 1)
    mod2 = ShortPanelGLS(y, dgp.exog, dgp.groups)
    res2 = mod2.fit_iterative(2)
    corr_wresid = np.corrcoef(res2.wresid.reshape(-1, n_groups, order='F'))
    assert_maxabs(corr_wresid, np.eye(5), 0.1)
    corr_resid = np.corrcoef(res2.resid.reshape(-1, n_groups, order='F'))
    assert_maxabs(corr_resid, dgp.cov, 0.1)
    assert_almost_equal(res2.resid.std(), 1, decimal=0)
    y_pred = np.dot(mod2.exog, res2.params)
    assert_almost_equal(res2.fittedvalues, y_pred, 13)
    res2_ols = mod2._fit_ols()
    npt.assert_(mod2.res_pooled is res2_ols)
    res2_ols = mod2.res_pooled
    mod1 = ShortPanelGLS(y, dgp.exog, dgp.groups)
    res1 = mod1.fit_iterative(1)
    assert_almost_equal(res1.params, res2_ols.params, decimal=13)
    assert_almost_equal(res1.bse, res2_ols.bse, decimal=13)
    res_ols = OLS(y, dgp.exog).fit()
    assert_almost_equal(res1.params, res_ols.params, decimal=13)
    assert_almost_equal(res1.bse, res_ols.bse, decimal=13)
    mod_old = ShortPanelGLS2(y, dgp.exog, dgp.groups)
    res_old = mod_old.fit()
    assert_almost_equal(res2.params, res_old.params, decimal=13)
    assert_almost_equal(res2.bse, res_old.bse, decimal=13)
    mod5 = ShortPanelGLS(y, dgp.exog, dgp.groups)
    res5 = mod5.fit_iterative(5)
    cov_clu = sw.cov_cluster(mod2.res_pooled, dgp.groups.astype(int))
    clubse = se_cov(cov_clu)
    pnwbse = se_cov(sw.cov_nw_panel(mod2.res_pooled, 4, mod2.group.groupidx))
    bser = np.vstack((res2.bse, res5.bse, clubse, pnwbse))
    bser_mean = np.mean(bser, axis=0)
    npt.assert_array_less(clubse / bser_mean - 1, 0.25)
    npt.assert_array_less(pnwbse / bser_mean - 1, 0.1)
    npt.assert_array_less(0.6, bser_mean / res_ols.bse - 1)
    cov_uni = sw.cov_nw_panel(mod2.res_pooled, 4, mod2.group.groupidx, weights_func=sw.weights_uniform, use_correction='c')
    assert_almost_equal(cov_uni, cov_clu, decimal=13)
    cov_clu2 = sw.cov_cluster(mod2.res_pooled, dgp.groups.astype(int), use_correction=False)
    cov_uni2 = sw.cov_nw_panel(mod2.res_pooled, 4, mod2.group.groupidx, weights_func=sw.weights_uniform, use_correction=False)
    assert_almost_equal(cov_uni2, cov_clu2, decimal=13)
    cov_white = sw.cov_white_simple(mod2.res_pooled)
    cov_pnw0 = sw.cov_nw_panel(mod2.res_pooled, 0, mod2.group.groupidx, use_correction='hac')
    assert_almost_equal(cov_pnw0, cov_white, decimal=13)