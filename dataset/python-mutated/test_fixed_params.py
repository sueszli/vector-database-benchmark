"""
Tests for fixing the values of some parameters and estimating others

Author: Chad Fulton
License: Simplified-BSD
"""
from __future__ import division, absolute_import, print_function
import numpy as np
import pytest
from statsmodels import datasets
from statsmodels.tsa.statespace import initialization, mlemodel, sarimax, structural, dynamic_factor, varmax
from numpy.testing import assert_, assert_raises, assert_equal, assert_allclose
macrodata = datasets.macrodata.load_pandas().data

def test_fix_params():
    if False:
        print('Hello World!')
    mod = mlemodel.MLEModel([], 1)
    mod._param_names = ['a', 'b', 'c']
    with mod.fix_params({'b': 1.0}):
        assert_(mod._has_fixed_params)
        assert_equal(mod._fixed_params, {'b': 1.0})
        assert_equal(mod._fixed_params_index, [1])
        assert_equal(mod._free_params_index, [0, 2])
    assert_(not mod._has_fixed_params)
    assert_equal(mod._fixed_params, {})
    assert_equal(mod._fixed_params_index, None)
    assert_equal(mod._free_params_index, None)

def test_nested_fix_params():
    if False:
        i = 10
        return i + 15
    mod = mlemodel.MLEModel([], 1)
    mod._param_names = ['a', 'b', 'c']
    with mod.fix_params({'a': 2, 'b': 0}):
        with mod.fix_params({'b': 1.0}):
            assert_(mod._has_fixed_params)
            assert_equal(mod._fixed_params, {'a': 2, 'b': 1.0})
            assert_equal(mod._fixed_params_index, [0, 1])
            assert_equal(mod._free_params_index, [2])
    assert_(not mod._has_fixed_params)
    assert_equal(mod._fixed_params, {})
    assert_equal(mod._fixed_params_index, None)
    assert_equal(mod._free_params_index, None)

def test_results_append():
    if False:
        return 10
    endog = macrodata['infl']
    endog1 = endog.iloc[:100]
    endog2 = endog.iloc[100:]
    mod_full = sarimax.SARIMAX(endog)
    with mod_full.fix_params({'ar.L1': 0.5}):
        res_full = mod_full.smooth([1.0], includes_fixed=False)
        start_params = [10.3]
        res_full_fit = mod_full.fit(start_params, disp=False)
    mod = sarimax.SARIMAX(endog1)
    with mod.fix_params({'ar.L1': 0.5}):
        res1 = mod.smooth([1.0], includes_fixed=False)
    res2 = res1.append(endog2)
    res2_fit = res1.append(endog2, refit=True, fit_kwargs={'disp': False, 'start_params': res_full_fit.params})
    assert_allclose(res2.params, res_full.params)
    assert_equal(res2._fixed_params, res_full._fixed_params)
    assert_allclose(res2.llf_obs, res_full.llf_obs)
    assert_allclose(res2_fit.params, res_full_fit.params)
    assert_equal(res2_fit._fixed_params, res_full_fit._fixed_params)
    assert_allclose(res2_fit.llf_obs, res_full_fit.llf_obs)

def test_results_extend():
    if False:
        i = 10
        return i + 15
    endog = macrodata['infl']
    endog1 = endog.iloc[:100]
    endog2 = endog.iloc[100:]
    mod_full = sarimax.SARIMAX(endog)
    with mod_full.fix_params({'ar.L1': 0.5}):
        res_full = mod_full.smooth([1.0], includes_fixed=False)
    mod = sarimax.SARIMAX(endog1)
    with mod.fix_params({'ar.L1': 0.5}):
        res1 = mod.smooth([1.0], includes_fixed=False)
    res2 = res1.append(endog2)
    assert_allclose(res2.params, res_full.params)
    assert_equal(res2._fixed_params, res_full._fixed_params)
    assert_allclose(res2.llf_obs, res_full.llf_obs)

def test_results_apply():
    if False:
        print('Hello World!')
    endog = macrodata['infl']
    mod = sarimax.SARIMAX(endog)
    with mod.fix_params({'ar.L1': 0.5}):
        res = mod.smooth([1.0], includes_fixed=False)
        start_params = [10.3]
        res_fit = mod.fit(start_params, disp=False)
    res2 = res.apply(endog)
    res2_fit = res.apply(endog, refit=True, fit_kwargs={'disp': False, 'start_params': res_fit.params})
    assert_allclose(res2.params, res.params)
    assert_equal(res2._fixed_params, res._fixed_params)
    assert_allclose(res2.llf_obs, res.llf_obs)
    assert_allclose(res2_fit.params, res_fit.params)
    assert_equal(res2_fit._fixed_params, res_fit._fixed_params)
    assert_allclose(res2_fit.llf_obs, res_fit.llf_obs)

def test_mle_validate():
    if False:
        print('Hello World!')
    mod = mlemodel.MLEModel([], 1)
    mod._param_names = ['a', 'b', 'c']
    msg = 'Invalid parameter name passed: "d"'
    with pytest.raises(ValueError, match=msg):
        with mod.fix_params({'d': 1}):
            pass

def test_sarimax_validate():
    if False:
        i = 10
        return i + 15
    endog = macrodata['infl']
    mod1 = sarimax.SARIMAX(endog, order=(2, 0, 0))
    assert_raises(ValueError, mod1.fit_constrained, {'AR.L1': 0.5})
    with pytest.raises(ValueError):
        with mod1.fix_params({'ar.L1': 0.5}):
            pass
    assert_raises(ValueError, mod1.fit_constrained, {'ar.L1': 0.5})
    with mod1.fix_params({'ar.L1': 0.5, 'ar.L2': 0.2}):
        assert_(mod1._has_fixed_params)
        assert_equal(mod1._fixed_params, {'ar.L1': 0.5, 'ar.L2': 0.2})
        assert_equal(mod1._fixed_params_index, [0, 1])
        assert_equal(mod1._free_params_index, [2])
    res = mod1.fit_constrained({'ar.L1': 0.5, 'ar.L2': 0.2}, start_params=[7.0], disp=False)
    assert_(res._has_fixed_params)
    assert_equal(res._fixed_params, {'ar.L1': 0.5, 'ar.L2': 0.2})
    assert_equal(res._fixed_params_index, [0, 1])
    assert_equal(res._free_params_index, [2])
    with mod1.fix_params({'ar.L1': 0.5, 'ar.L2': 0.0}):
        with mod1.fix_params({'ar.L2': 0.2}):
            assert_(mod1._has_fixed_params)
            assert_equal(mod1._fixed_params, {'ar.L1': 0.5, 'ar.L2': 0.2})
            assert_equal(mod1._fixed_params_index, [0, 1])
            assert_equal(mod1._free_params_index, [2])

def test_structural_validate():
    if False:
        while True:
            i = 10
    endog = macrodata['infl']
    mod1 = structural.UnobservedComponents(endog, 'rwalk', autoregressive=2)
    assert_raises(ValueError, mod1.fit_constrained, {'AR.L1': 0.5})
    with pytest.raises(ValueError):
        with mod1.fix_params({'ar.L1': 0.5}):
            pass
    assert_raises(ValueError, mod1.fit_constrained, {'ar.L1': 0.5})
    with mod1.fix_params({'ar.L1': 0.5, 'ar.L2': 0.2}):
        assert_(mod1._has_fixed_params)
        assert_equal(mod1._fixed_params, {'ar.L1': 0.5, 'ar.L2': 0.2})
        assert_equal(mod1._fixed_params_index, [2, 3])
        assert_equal(mod1._free_params_index, [0, 1])
    res = mod1.fit_constrained({'ar.L1': 0.5, 'ar.L2': 0.2}, start_params=[7.0], disp=False)
    assert_(res._has_fixed_params)
    assert_equal(res._fixed_params, {'ar.L1': 0.5, 'ar.L2': 0.2})
    assert_equal(res._fixed_params_index, [2, 3])
    assert_equal(res._free_params_index, [0, 1])
    with mod1.fix_params({'ar.L1': 0.5, 'ar.L2': 0.0}):
        with mod1.fix_params({'ar.L2': 0.2}):
            assert_(mod1._has_fixed_params)
            assert_equal(mod1._fixed_params, {'ar.L1': 0.5, 'ar.L2': 0.2})
            assert_equal(mod1._fixed_params_index, [2, 3])
            assert_equal(mod1._free_params_index, [0, 1])

def test_dynamic_factor_validate():
    if False:
        for i in range(10):
            print('nop')
    endog = np.log(macrodata[['cpi', 'realgdp', 'realinv']]).diff().iloc[1:]
    endog = (endog - endog.mean()) / endog.std()
    mod1 = dynamic_factor.DynamicFactor(endog, k_factors=1, factor_order=1, error_cov_type='diagonal')
    constraints = {'loading.f1.cpi': 0.5}
    with mod1.fix_params(constraints):
        assert_(mod1._has_fixed_params)
        assert_equal(mod1._fixed_params, constraints)
        assert_equal(mod1._fixed_params_index, [0])
        assert_equal(mod1._free_params_index, [1, 2, 3, 4, 5, 6])
    res1 = mod1.fit_constrained(constraints, disp=False)
    assert_(res1._has_fixed_params)
    assert_equal(res1._fixed_params, constraints)
    assert_equal(res1._fixed_params_index, [0])
    assert_equal(res1._free_params_index, [1, 2, 3, 4, 5, 6])
    with mod1.fix_params({'L1.f1.f1': 0.5}):
        assert_(mod1._has_fixed_params)
        assert_equal(mod1._fixed_params, {'L1.f1.f1': 0.5})
        assert_equal(mod1._fixed_params_index, [6])
        assert_equal(mod1._free_params_index, [0, 1, 2, 3, 4, 5])
    mod2 = dynamic_factor.DynamicFactor(endog, k_factors=1, factor_order=2, error_cov_type='diagonal')
    with pytest.raises(ValueError):
        with mod2.fix_params({'L1.f1.f1': 0.5}):
            pass
    constraints = {'L1.f1.f1': 0.3, 'L2.f1.f1': 0.1}
    with mod2.fix_params(constraints):
        assert_(mod2._has_fixed_params)
        assert_equal(mod2._fixed_params, constraints)
        assert_equal(mod2._fixed_params_index, [6, 7])
        assert_equal(mod2._free_params_index, [0, 1, 2, 3, 4, 5])
    res2 = mod2.fit_constrained(constraints, disp=False)
    assert_(res2._has_fixed_params)
    assert_equal(res2._fixed_params, constraints)
    assert_equal(res2._fixed_params_index, [6, 7])
    assert_equal(res2._free_params_index, [0, 1, 2, 3, 4, 5])
    with mod2.fix_params(constraints):
        with mod2.fix_params({'L1.f1.f1': -0.3}):
            assert_(mod2._has_fixed_params)
            assert_equal(mod2._fixed_params, {'L1.f1.f1': -0.3, 'L2.f1.f1': 0.1})
            assert_equal(mod2._fixed_params_index, [6, 7])
            assert_equal(mod2._free_params_index, [0, 1, 2, 3, 4, 5])
    mod3 = dynamic_factor.DynamicFactor(endog, k_factors=2, factor_order=1, error_cov_type='diagonal')
    with pytest.raises(ValueError):
        with mod3.fix_params({'L1.f1.f1': 0.3}):
            pass
    constraints = dict([('L1.f1.f1', 0.3), ('L1.f2.f1', 0.1), ('L1.f1.f2', -0.05), ('L1.f2.f2', 0.1)])
    with mod3.fix_params(constraints):
        assert_(mod3._has_fixed_params)
        assert_equal(mod3._fixed_params, constraints)
        assert_equal(mod3._fixed_params_index, [9, 10, 11, 12])
        assert_equal(mod3._free_params_index, [0, 1, 2, 3, 4, 5, 6, 7, 8])
    res3 = mod3.fit_constrained(constraints, disp=False)
    assert_(res3._has_fixed_params)
    assert_equal(res3._fixed_params, constraints)
    assert_equal(res3._fixed_params_index, [9, 10, 11, 12])
    assert_equal(res3._free_params_index, [0, 1, 2, 3, 4, 5, 6, 7, 8])
    with mod3.fix_params(constraints):
        with mod3.fix_params({'L1.f1.f1': -0.3, 'L1.f2.f2': -0.1}):
            assert_(mod3._has_fixed_params)
            assert_equal(mod3._fixed_params, dict([('L1.f1.f1', -0.3), ('L1.f2.f1', 0.1), ('L1.f1.f2', -0.05), ('L1.f2.f2', -0.1)]))
            assert_equal(mod3._fixed_params_index, [9, 10, 11, 12])
            assert_equal(mod3._free_params_index, [0, 1, 2, 3, 4, 5, 6, 7, 8])
    mod4 = dynamic_factor.DynamicFactor(endog, k_factors=1, factor_order=2, error_cov_type='diagonal', enforce_stationarity=False)
    with mod4.fix_params({'L1.f1.f1': 0.6}):
        assert_(mod4._has_fixed_params)
        assert_equal(mod4._fixed_params, {'L1.f1.f1': 0.6})
        assert_equal(mod4._fixed_params_index, [6])
        assert_equal(mod4._free_params_index, [0, 1, 2, 3, 4, 5, 7])
    mod5 = dynamic_factor.DynamicFactor(endog, k_factors=2, factor_order=1, error_cov_type='diagonal', enforce_stationarity=False)
    with mod5.fix_params({'L1.f1.f1': 0.6}):
        assert_(mod5._has_fixed_params)
        assert_equal(mod5._fixed_params, {'L1.f1.f1': 0.6})
        assert_equal(mod5._fixed_params_index, [9])
        assert_equal(mod5._free_params_index, [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12])
    constraints = {'sigma2.cpi': 0.9, 'sigma2.realinv': 3}
    with mod1.fix_params(constraints):
        assert_(mod1._has_fixed_params)
        assert_equal(mod1._fixed_params, constraints)
        assert_equal(mod1._fixed_params_index, [3, 5])
        assert_equal(mod1._free_params_index, [0, 1, 2, 4, 6])
    res1 = mod1.fit_constrained(constraints, disp=False)
    assert_(res1._has_fixed_params)
    assert_equal(res1._fixed_params, constraints)
    assert_equal(res1._fixed_params_index, [3, 5])
    assert_equal(res1._free_params_index, [0, 1, 2, 4, 6])
    mod6 = dynamic_factor.DynamicFactor(endog[['cpi', 'realgdp']], k_factors=1, factor_order=1, error_cov_type='unstructured')
    constraints = {'loading.f1.cpi': 1.0, 'loading.f1.realgdp': 1.0, 'cov.chol[1,1]': 0.5, 'cov.chol[2,1]': 0.1}
    with mod6.fix_params(constraints):
        assert_(mod6._has_fixed_params)
        assert_equal(mod6._fixed_params, constraints)
        assert_equal(mod6._fixed_params_index, [0, 1, 2, 3])
        assert_equal(mod6._free_params_index, [4, 5])
    res6 = mod6.fit_constrained(constraints, disp=False)
    assert_(res6._has_fixed_params)
    assert_equal(res6._fixed_params, constraints)
    assert_equal(res6._fixed_params_index, [0, 1, 2, 3])
    assert_equal(res6._free_params_index, [4, 5])

def test_varmax_validate():
    if False:
        while True:
            i = 10
    endog = np.log(macrodata[['cpi', 'realgdp']]).diff().iloc[1:]
    exog = np.log(macrodata[['realinv']]).diff().iloc[1:]
    mod1 = varmax.VARMAX(endog, order=(1, 0), exog=exog, measurement_error=True)
    constraints = {'intercept.cpi': 0.5, 'intercept.realgdp': 1.1, 'beta.realinv.cpi': 0.2, 'beta.realinv.realgdp': 0.1, 'sqrt.var.cpi': 1.2, 'sqrt.cov.cpi.realgdp': -0.1, 'sqrt.var.realgdp': 2.3, 'measurement_variance.cpi': 0.4, 'measurement_variance.realgdp': 0.4}
    with mod1.fix_params(constraints):
        assert_(mod1._has_fixed_params)
        assert_equal(mod1._fixed_params, constraints)
        assert_equal(mod1._fixed_params_index, [0, 1, 6, 7, 8, 9, 10, 11, 12])
        assert_equal(mod1._free_params_index, [2, 3, 4, 5])
    res1 = mod1.fit_constrained(constraints, disp=False)
    assert_(res1._has_fixed_params)
    assert_equal(res1._fixed_params, constraints)
    assert_equal(res1._fixed_params_index, [0, 1, 6, 7, 8, 9, 10, 11, 12])
    assert_equal(res1._free_params_index, [2, 3, 4, 5])
    mod2 = varmax.VARMAX(endog[['cpi']], order=(1, 0), exog=exog, measurement_error=True)
    constraints = {'L1.cpi.cpi': 0.5}
    with mod2.fix_params(constraints):
        assert_(mod2._has_fixed_params)
        assert_equal(mod2._fixed_params, constraints)
        assert_equal(mod2._fixed_params_index, [1])
        assert_equal(mod2._free_params_index, [0, 2, 3, 4])
    mod3 = varmax.VARMAX(endog[['cpi']], order=(2, 0))
    with pytest.raises(ValueError):
        with mod3.fix_params({'L1.cpi.cpi': 0.5}):
            pass
    constraints = {'L1.cpi.cpi': 0.3, 'L2.cpi.cpi': 0.1}
    with mod3.fix_params(constraints):
        assert_(mod3._has_fixed_params)
        assert_equal(mod3._fixed_params, constraints)
        assert_equal(mod3._fixed_params_index, [1, 2])
        assert_equal(mod3._free_params_index, [0, 3])
    res3 = mod3.fit_constrained(constraints, start_params=[0, 1.0], disp=False)
    assert_(res3._has_fixed_params)
    assert_equal(res3._fixed_params, constraints)
    assert_equal(res3._fixed_params_index, [1, 2])
    assert_equal(res3._free_params_index, [0, 3])
    with mod3.fix_params(constraints):
        with mod3.fix_params({'L1.cpi.cpi': -0.3}):
            assert_(mod3._has_fixed_params)
            assert_equal(mod3._fixed_params, {'L1.cpi.cpi': -0.3, 'L2.cpi.cpi': 0.1})
            assert_equal(mod3._fixed_params_index, [1, 2])
            assert_equal(mod3._free_params_index, [0, 3])
    mod4 = varmax.VARMAX(endog, order=(1, 0))
    with pytest.raises(ValueError):
        with mod4.fix_params({'L1.cpi.cpi': 0.3}):
            pass
    constraints = dict([('L1.cpi.cpi', 0.3), ('L1.realgdp.cpi', 0.1), ('L1.cpi.realgdp', -0.05), ('L1.realgdp.realgdp', 0.1)])
    with mod4.fix_params(constraints):
        assert_(mod4._has_fixed_params)
        assert_equal(mod4._fixed_params, constraints)
        assert_equal(mod4._fixed_params_index, [2, 3, 4, 5])
        assert_equal(mod4._free_params_index, [0, 1, 6, 7, 8])
    res4 = mod4.fit_constrained(constraints, disp=False)
    assert_(res4._has_fixed_params)
    assert_equal(res4._fixed_params, constraints)
    assert_equal(res4._fixed_params_index, [2, 3, 4, 5])
    assert_equal(res4._free_params_index, [0, 1, 6, 7, 8])
    mod5 = varmax.VARMAX(endog[['cpi']], order=(1, 0), enforce_stationarity=False)
    with mod5.fix_params({'L1.cpi.cpi': 0.6}):
        assert_(mod5._has_fixed_params)
        assert_equal(mod5._fixed_params, {'L1.cpi.cpi': 0.6})
        assert_equal(mod5._fixed_params_index, [1])
        assert_equal(mod5._free_params_index, [0, 2])
    mod6 = varmax.VARMAX(endog, order=(1, 0), enforce_stationarity=False)
    with mod6.fix_params({'L1.cpi.cpi': 0.6}):
        assert_(mod6._has_fixed_params)
        assert_equal(mod6._fixed_params, {'L1.cpi.cpi': 0.6})
        assert_equal(mod6._fixed_params_index, [2])
        assert_equal(mod6._free_params_index, [0, 1, 3, 4, 5, 6, 7, 8])

def check_results(res1, res2, check_lutkepohl=False, check_params=True):
    if False:
        while True:
            i = 10
    assert_allclose(res2.nobs, res1.nobs)
    assert_allclose(res2.nobs_diffuse, res1.nobs_diffuse)
    assert_allclose(res2.nobs_effective, res1.nobs_effective)
    assert_allclose(res2.k_diffuse_states, res1.k_diffuse_states)
    assert_allclose(res2.df_model, res1.df_model)
    assert_allclose(res2.df_resid, res1.df_resid)
    assert_allclose(res2.llf, res1.llf)
    assert_allclose(res2.aic, res1.aic)
    assert_allclose(res2.bic, res1.bic)
    assert_allclose(res2.hqic, res1.hqic)
    if check_lutkepohl:
        assert_allclose(res2.info_criteria('aic', 'lutkepohl'), res1.info_criteria('aic', 'lutkepohl'))
        assert_allclose(res2.info_criteria('bic', 'lutkepohl'), res1.info_criteria('bic', 'lutkepohl'))
        assert_allclose(res2.info_criteria('hqic', 'lutkepohl'), res1.info_criteria('hqic', 'lutkepohl'))
    assert_allclose(res2.llf_obs, res1.llf_obs)
    assert_allclose(res2.fittedvalues, res1.fittedvalues)
    assert_allclose(res2.fittedvalues, res1.fittedvalues)
    if check_params:
        mask_free = res2._free_params_index
        mask_fixed = res2._fixed_params_index
        pvalues = np.asarray(res2.pvalues)
        assert_allclose(pvalues[mask_free], res1.pvalues)
        assert_allclose(pvalues[mask_fixed], np.nan)
        bse = np.asarray(res2.bse)
        assert_allclose(bse[mask_free], res1.bse)
        assert_allclose(bse[mask_fixed], np.nan)
        zvalues = np.asarray(res2.zvalues)
        assert_allclose(zvalues[mask_free], res1.zvalues)
        assert_allclose(zvalues[mask_fixed], np.nan)
        mask_free = np.ix_(res2._free_params_index, res2._free_params_index)
        mask_fixed = np.ix_(res2._fixed_params_index, res2._fixed_params_index)
        assert_allclose(res2.cov_params_default.values[mask_free], res1.cov_params_default)
        assert_allclose(res2.cov_params_default.values[mask_fixed], np.nan)
        assert_allclose(res2.cov_params_approx.values[mask_free], res1.cov_params_approx)
        assert_allclose(res2.cov_params_approx.values[mask_fixed], np.nan)
        assert_allclose(res2.cov_params_oim.values[mask_free], res1.cov_params_oim)
        assert_allclose(res2.cov_params_oim.values[mask_fixed], np.nan)
        assert_allclose(res2.cov_params_opg.values[mask_free], res1.cov_params_opg)
        assert_allclose(res2.cov_params_opg.values[mask_fixed], np.nan)
        assert_allclose(res2.cov_params_robust.values[mask_free], res1.cov_params_robust)
        assert_allclose(res2.cov_params_robust.values[mask_fixed], np.nan)
        assert_allclose(res2.cov_params_robust_oim.values[mask_free], res1.cov_params_robust_oim)
        assert_allclose(res2.cov_params_robust_oim.values[mask_fixed], np.nan)
        assert_allclose(res2.cov_params_robust_approx.values[mask_free], res1.cov_params_robust_approx)
        assert_allclose(res2.cov_params_robust_approx.values[mask_fixed], np.nan)
    assert_allclose(res2.test_normality('jarquebera'), res1.test_normality('jarquebera'))
    assert_allclose(res2.test_heteroskedasticity('breakvar'), res1.test_heteroskedasticity('breakvar'))
    actual = res2.test_serial_correlation('ljungbox')
    desired = res1.test_serial_correlation('ljungbox')
    assert_allclose(actual, desired)

def test_sarimax_nonconsecutive():
    if False:
        while True:
            i = 10
    endog = macrodata['infl']
    mod1 = sarimax.SARIMAX(endog, order=([1, 0, 0, 1], 0, 0), enforce_stationarity=False)
    mod2 = sarimax.SARIMAX(endog, order=(4, 0, 0), enforce_stationarity=False)
    start_params = [0.6, 0.2, 6.4]
    res1 = mod1.fit(start_params, disp=False)
    res2 = mod2.fit_constrained({'ar.L2': 0, 'ar.L3': 0}, res1.params, includes_fixed=False, disp=False)
    assert_equal(res1.fixed_params, [])
    assert_equal(res2.fixed_params, ['ar.L2', 'ar.L3'])
    params = np.asarray(res1.params)
    desired = np.r_[params[0], 0, 0, params[1:]]
    assert_allclose(res2.params, desired)
    with mod2.fix_params({'ar.L2': 0, 'ar.L3': 0}):
        res2 = mod2.smooth(res1.params)
    check_results(res1, res2, check_lutkepohl=True)
    with mod2.fix_params({'ar.L2': 0, 'ar.L3': 0}):
        res3 = mod2.filter(res2.params, includes_fixed=True)
        check_results(res1, res3, check_lutkepohl=True)

def test_structural():
    if False:
        for i in range(10):
            print('nop')
    endog = macrodata['infl']
    mod1 = structural.UnobservedComponents(endog, 'llevel')
    mod2 = structural.UnobservedComponents(endog, 'lltrend')
    init = initialization.Initialization(mod2.k_states)
    init[0] = 'approximate_diffuse'
    init.set(1, 'known', constant=[0])
    mod2.ssm.initialization = init
    mod2.ssm.loglikelihood_burn = 1
    constraints = {'sigma2.trend': 0}
    start_params = [3.37, 0.74]
    res1 = mod1.fit(start_params, disp=False)
    res2 = mod2.fit_constrained(constraints, start_params=res1.params, includes_fixed=False, disp=False)
    assert_equal(res1.fixed_params, [])
    assert_equal(res2.fixed_params, ['sigma2.trend'])
    desired = np.r_[res1.params, 0]
    assert_allclose(res2.params, desired)
    with mod2.fix_params(constraints):
        res2 = mod2.smooth(res1.params)
    check_results(res1, res2)

def test_dynamic_factor_diag_error_cov():
    if False:
        print('Hello World!')
    endog = np.log(macrodata[['cpi', 'realgdp']]).diff().iloc[1:]
    endog = (endog - endog.mean()) / endog.std()
    mod1 = dynamic_factor.DynamicFactor(endog, k_factors=1, factor_order=1, error_cov_type='diagonal')
    mod2 = dynamic_factor.DynamicFactor(endog, k_factors=1, factor_order=1, error_cov_type='unstructured')
    constraints = {'cov.chol[2,1]': 0}
    start_params = [-4.5e-06, -1e-05, 0.99, 0.99, -0.14]
    res1 = mod1.fit(start_params=start_params, disp=False)
    res2 = mod2.fit_constrained(constraints, start_params=res1.params, includes_fixed=False, disp=False)
    assert_equal(res1.fixed_params, [])
    assert_equal(res2.fixed_params, ['cov.chol[2,1]'])
    param_vals = np.asarray(res1.params)
    params = np.r_[param_vals[:2], param_vals[2:4] ** 0.5, param_vals[4]]
    desired = np.r_[params[:3], 0, params[3:]]
    assert_allclose(res2.params, desired, atol=1e-05)
    with mod2.fix_params(constraints):
        res2 = mod2.smooth(params)
    check_results(res1, res2, check_params=False)

def test_score_shape():
    if False:
        for i in range(10):
            print('nop')
    endog = macrodata['infl']
    mod = sarimax.SARIMAX(endog, order=(1, 0, 0))
    with mod.fix_params({'ar.L1': 0.5}):
        score = mod.score([1.0])
    assert_equal(score.shape, (1,))