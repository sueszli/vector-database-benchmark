"""
Tests for impulse responses of time series

Author: Chad Fulton
License: Simplified-BSD
"""
import warnings
import numpy as np
import pandas as pd
from scipy.stats import ortho_group
import pytest
from numpy.testing import assert_, assert_allclose
from statsmodels.tools.sm_exceptions import EstimationWarning
from statsmodels.tsa.statespace import mlemodel, sarimax, structural, varmax, dynamic_factor
from statsmodels.tsa.vector_ar.tests.test_var import get_macrodata

def test_sarimax():
    if False:
        i = 10
        return i + 15
    mod = sarimax.SARIMAX([0], order=(1, 0, 0))
    phi = 0.5
    actual = mod.impulse_responses([phi, 1], steps=10)
    desired = np.r_[[phi ** i for i in range(11)]]
    assert_allclose(actual, desired)
    mod = sarimax.SARIMAX([0], order=(0, 0, 1))
    theta = 0.5
    actual = mod.impulse_responses([theta, 1], steps=10)
    desired = np.r_[1, theta, [0] * 9]
    assert_allclose(actual, desired)
    params = [0.01928228, -0.03656216, 0.7588994, 0.27070341, -0.72928328, 0.01122177 ** 0.5]
    mod = sarimax.SARIMAX([0], order=(2, 0, 2), trend='c')
    actual = mod.impulse_responses(params, steps=10)
    desired = [1, 0.234141, 0.021055, 0.17692, 0.00951, 0.133917, 0.002321, 0.101544, -0.001951, 0.077133, -0.004301]
    assert_allclose(actual, desired, atol=1e-06)
    params = [0.12853289, 12.207156, 0.86384742, -0.71463236, 0.81878967, -0.9533955, 14.043884 ** 0.5]
    exog = np.arange(1, 92) ** 2
    mod = sarimax.SARIMAX(np.zeros(91), order=(1, 1, 1), seasonal_order=(1, 0, 1, 4), trend='c', exog=exog, simple_differencing=True)
    actual = mod.impulse_responses(params, steps=10)
    desired = [1, 0.149215, 0.128899, 0.111349, -0.038417, 0.063007, 0.054429, 0.047018, -0.069598, 0.018641, 0.016103]
    assert_allclose(actual, desired, atol=1e-06)

def test_structural():
    if False:
        for i in range(10):
            print('nop')
    steps = 10
    mod = structural.UnobservedComponents([0], autoregressive=1)
    phi = 0.5
    actual = mod.impulse_responses([1, phi], steps)
    desired = np.r_[[phi ** i for i in range(steps + 1)]]
    assert_allclose(actual, desired)
    mod = structural.UnobservedComponents([0], 'irregular')
    actual = mod.impulse_responses([1.0], steps)
    assert_allclose(actual, 0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mod = structural.UnobservedComponents([0], 'fixed intercept')
    actual = mod.impulse_responses([1.0], steps)
    assert_allclose(actual, 0)
    mod = structural.UnobservedComponents([0], 'deterministic constant')
    actual = mod.impulse_responses([1.0], steps)
    assert_allclose(actual, 0)
    mod = structural.UnobservedComponents([0], 'local level')
    actual = mod.impulse_responses([1.0, 1.0], steps)
    assert_allclose(actual, 1)
    mod = structural.UnobservedComponents([0], 'random walk')
    actual = mod.impulse_responses([1.0], steps)
    assert_allclose(actual, 1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mod = structural.UnobservedComponents([0], 'fixed slope')
    actual = mod.impulse_responses([1.0], steps)
    assert_allclose(actual, 0)
    mod = structural.UnobservedComponents([0], 'deterministic trend')
    actual = mod.impulse_responses([1.0], steps)
    assert_allclose(actual, 0)
    mod = structural.UnobservedComponents([0], 'local linear deterministic trend')
    actual = mod.impulse_responses([1.0, 1.0], steps)
    assert_allclose(actual, 1)
    mod = structural.UnobservedComponents([0], 'random walk with drift')
    actual = mod.impulse_responses([1.0], steps)
    assert_allclose(actual, 1)
    mod = structural.UnobservedComponents([0], 'local linear trend')
    actual = mod.impulse_responses([1.0, 1.0, 1.0], steps)
    assert_allclose(actual, 1)
    actual = mod.impulse_responses([1.0, 1.0, 1.0], steps, impulse=1)
    assert_allclose(actual, np.arange(steps + 1))
    mod = structural.UnobservedComponents([0], 'smooth trend')
    actual = mod.impulse_responses([1.0, 1.0], steps)
    assert_allclose(actual, np.arange(steps + 1))
    mod = structural.UnobservedComponents([0], 'random trend')
    actual = mod.impulse_responses([1.0, 1.0], steps)
    assert_allclose(actual, np.arange(steps + 1))
    mod = structural.UnobservedComponents([0], 'irregular', seasonal=2, stochastic_seasonal=False)
    actual = mod.impulse_responses([1.0], steps)
    assert_allclose(actual, 0)
    mod = structural.UnobservedComponents([0], 'irregular', seasonal=2)
    actual = mod.impulse_responses([1.0, 1.0], steps)
    desired = np.r_[1, np.tile([-1, 1], steps // 2)]
    assert_allclose(actual, desired)
    mod = structural.UnobservedComponents([0], 'irregular', cycle=True)
    actual = mod.impulse_responses([1.0, 1.2], steps)
    assert_allclose(actual, 0)
    mod = structural.UnobservedComponents([0], 'irregular', cycle=True, stochastic_cycle=True)
    actual = mod.impulse_responses([1.0, 1.0, 1.2], steps=10)
    x1 = [np.cos(1.2), np.sin(1.2)]
    x2 = [-np.sin(1.2), np.cos(1.2)]
    T = np.array([x1, x2])
    desired = np.zeros(steps + 1)
    states = [1, 0]
    for i in range(steps + 1):
        desired[i] += states[0]
        states = np.dot(T, states)
    assert_allclose(actual, desired)

def test_varmax():
    if False:
        while True:
            i = 10
    steps = 10
    varmax.__warningregistry__ = {}
    mod1 = varmax.VARMAX([[0]], order=(2, 0), trend='n')
    mod2 = sarimax.SARIMAX([0], order=(2, 0, 0))
    actual = mod1.impulse_responses([0.5, 0.2, 1], steps)
    desired = mod2.impulse_responses([0.5, 0.2, 1], steps)
    assert_allclose(actual, desired)
    mod1 = varmax.VARMAX([[0]], order=(0, 2), trend='n')
    mod2 = sarimax.SARIMAX([0], order=(0, 0, 2))
    actual = mod1.impulse_responses([0.5, 0.2, 1], steps)
    desired = mod2.impulse_responses([0.5, 0.2, 1], steps)
    assert_allclose(actual, desired)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mod1 = varmax.VARMAX([[0]], order=(2, 2), trend='n')
    mod2 = sarimax.SARIMAX([0], order=(2, 0, 2))
    actual = mod1.impulse_responses([0.5, 0.2, 0.1, -0.2, 1], steps)
    desired = mod2.impulse_responses([0.5, 0.2, 0.1, -0.2, 1], steps)
    assert_allclose(actual, desired)
    warning = EstimationWarning
    match = 'VARMA\\(p,q\\) models is not'
    with pytest.warns(warning, match=match):
        mod1 = varmax.VARMAX([[0]], order=(2, 2), trend='c')
    mod2 = sarimax.SARIMAX([0], order=(2, 0, 2), trend='c')
    actual = mod1.impulse_responses([10, 0.5, 0.2, 0.1, -0.2, 1], steps)
    desired = mod2.impulse_responses([10, 0.5, 0.2, 0.1, -0.2, 1], steps)
    assert_allclose(actual, desired)
    params = [-0.00122728, 0.01503679, -0.22741923, 0.71030531, -0.11596357, 0.51494891, 0.05974659, 0.02094608, 0.05635125, 0.08332519, 0.04297918, 0.00159473, 0.01096298]
    irf_00 = [1, -0.227419, -0.021806, 0.093362, -0.001875, -0.00906, 0.009605, 0.001323, -0.001041, 0.000769, 0.00032]
    irf_01 = [0, 0.059747, 0.044015, -0.008218, 0.007845, 0.004629, 0.000104, 0.000451, 0.000638, 6.3e-05, 4.2e-05]
    irf_10 = [0, 0.710305, 0.36829, -0.065697, 0.084398, 0.043038, 0.000533, 0.005755, 0.006051, 0.000548, 0.000526]
    irf_11 = [1, 0.020946, 0.126202, 0.066419, 0.028735, 0.007477, 0.009878, 0.003287, 0.001266, 0.000986, 0.0005]
    oirf_00 = [0.042979, -0.008642, -0.00035, 0.003908, 5.4e-05, -0.000321, 0.000414, 6.6e-05, -3.5e-05, 3.4e-05, 1.5e-05]
    oirf_01 = [0.001595, 0.002601, 0.002093, -0.000247, 0.000383, 0.000211, 2e-05, 2.5e-05, 2.9e-05, 4.3e-06, 2.6e-06]
    oirf_10 = [0, 0.007787, 0.004037, -0.00072, 0.000925, 0.000472, 5.8e-06, 6.3e-05, 6.6e-05, 6e-06, 5.8e-06]
    oirf_11 = [0.010963, 0.00023, 0.001384, 0.000728, 0.000315, 8.2e-05, 0.000108, 3.6e-05, 1.4e-05, 1.1e-05, 5.5e-06]
    mod = varmax.VARMAX([[0, 0]], order=(2, 0), trend='c')
    actual = mod.impulse_responses(params, steps, impulse=0)
    assert_allclose(actual, np.c_[irf_00, irf_01], atol=1e-06)
    actual = mod.impulse_responses(params, steps, impulse=1)
    assert_allclose(actual, np.c_[irf_10, irf_11], atol=1e-06)
    actual = mod.impulse_responses(params, steps, impulse=0, orthogonalized=True)
    assert_allclose(actual, np.c_[oirf_00, oirf_01], atol=1e-06)
    actual = mod.impulse_responses(params, steps, impulse=1, orthogonalized=True)
    assert_allclose(actual, np.c_[oirf_10, oirf_11], atol=1e-06)
    data = get_macrodata().view((float, 3), type=np.ndarray)
    df = pd.DataFrame({'a': data[:, 0], 'b': data[:, 1], 'c': data[:, 2]})
    mod1 = varmax.VARMAX(df, order=(1, 0), trend='c')
    mod1_result = mod1.fit()
    mod2 = varmax.VARMAX(data, order=(1, 0), trend='c')
    mod2_result = mod2.fit()
    with pytest.raises(ValueError, match='Endog must be pd.DataFrame.'):
        mod2_result.impulse_responses(6, impulse='b')
    response1 = mod1_result.impulse_responses(6, impulse='b')
    response2 = mod1_result.impulse_responses(6, impulse=[0, 1, 0])
    assert_allclose(response1, response2)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mod = varmax.VARMAX(np.random.normal(size=(steps, 2)), order=(2, 2), trend='c', exog=np.ones(steps), enforce_stationarity=False, enforce_invertibility=False)
    mod.impulse_responses(mod.start_params, steps)

def test_dynamic_factor():
    if False:
        return 10
    steps = 10
    exog = np.random.normal(size=steps)
    mod1 = dynamic_factor.DynamicFactor([[0, 0]], k_factors=1, factor_order=2)
    mod2 = sarimax.SARIMAX([0], order=(2, 0, 0))
    actual = mod1.impulse_responses([-0.9, 0.8, 1.0, 1.0, 0.5, 0.2], steps)
    desired = mod2.impulse_responses([0.5, 0.2, 1], steps)
    assert_allclose(actual[:, 0], -0.9 * desired)
    assert_allclose(actual[:, 1], 0.8 * desired)
    mod1 = dynamic_factor.DynamicFactor(np.zeros((steps, 2)), k_factors=1, factor_order=2, exog=exog)
    mod2 = sarimax.SARIMAX([0], order=(2, 0, 0))
    actual = mod1.impulse_responses([-0.9, 0.8, 5, -2, 1.0, 1.0, 0.5, 0.2], steps)
    desired = mod2.impulse_responses([0.5, 0.2, 1], steps)
    assert_allclose(actual[:, 0], -0.9 * desired)
    assert_allclose(actual[:, 1], 0.8 * desired)
    mod = dynamic_factor.DynamicFactor(np.random.normal(size=(steps, 3)), k_factors=2, factor_order=2, exog=exog, error_order=2, error_var=True, enforce_stationarity=False)
    mod.impulse_responses(mod.start_params, steps)

def test_time_varying_ssm():
    if False:
        for i in range(10):
            print('nop')
    mod = sarimax.SARIMAX([0] * 11, order=(1, 0, 0))
    mod.update([0.5, 1.0])
    T = np.zeros((1, 1, 11))
    T[..., :5] = 0.5
    T[..., 5:] = 0.2
    mod['transition'] = T
    irfs = mod.ssm.impulse_responses()
    desired = np.cumprod(np.r_[1, [0.5] * 4, [0.2] * 5]).reshape(10, 1)
    assert_allclose(irfs, desired)

class TVSS(mlemodel.MLEModel):
    """
    Time-varying state space model for testing

    This creates a state space model with randomly generated time-varying
    system matrices. When used in a test, that test should use
    `reset_randomstate` to ensure consistent test runs.
    """

    def __init__(self, endog, _k_states=None):
        if False:
            return 10
        k_states = 2
        k_posdef = 2
        if _k_states is None:
            _k_states = k_states
        super(TVSS, self).__init__(endog, k_states=_k_states, k_posdef=k_posdef, initialization='diffuse')
        self['obs_intercept'] = np.random.normal(size=(self.k_endog, self.nobs))
        self['design'] = np.zeros((self.k_endog, self.k_states, self.nobs))
        self['transition'] = np.zeros((self.k_states, self.k_states, self.nobs))
        self['selection'] = np.zeros((self.k_states, self.ssm.k_posdef, self.nobs))
        self['design', :, :k_states, :] = np.random.normal(size=(self.k_endog, k_states, self.nobs))
        D = [np.diag(d) for d in np.random.uniform(-1.1, 1.1, size=(self.nobs, k_states))]
        Q = ortho_group.rvs(k_states, size=self.nobs)
        self['transition', :k_states, :k_states, :] = (Q @ D @ Q.transpose(0, 2, 1)).transpose(1, 2, 0)
        self['selection', :k_states, :, :] = np.random.normal(size=(k_states, self.ssm.k_posdef, self.nobs))
        H05 = np.random.normal(size=(self.k_endog, self.k_endog, self.nobs))
        Q05 = np.random.normal(size=(self.ssm.k_posdef, self.ssm.k_posdef, self.nobs))
        H = np.zeros_like(H05)
        Q = np.zeros_like(Q05)
        for t in range(self.nobs):
            H[..., t] = np.dot(H05[..., t], H05[..., t].T)
            Q[..., t] = np.dot(Q05[..., t], Q05[..., t].T)
        self['obs_cov'] = H
        self['state_cov'] = Q

    def clone(self, endog, exog=None, **kwargs):
        if False:
            return 10
        mod = self.__class__(endog, **kwargs)
        for key in self.ssm.shapes.keys():
            if key in ['obs', 'state_intercept']:
                continue
            n = min(self.nobs, mod.nobs)
            mod[key, ..., :n] = self.ssm[key, ..., :n]
        return mod

def test_time_varying_in_sample(reset_randomstate):
    if False:
        for i in range(10):
            print('nop')
    mod = TVSS(np.zeros((10, 2)))
    irfs = mod.impulse_responses([], steps=mod.nobs - 1)
    irfs_anchor = mod.impulse_responses([], steps=mod.nobs - 1, anchor=0)
    cirfs = mod.impulse_responses([], steps=mod.nobs - 1, cumulative=True)
    oirfs = mod.impulse_responses([], steps=mod.nobs - 1, orthogonalized=True)
    coirfs = mod.impulse_responses([], steps=mod.nobs - 1, cumulative=True, orthogonalized=True)
    Z = mod['design']
    T = mod['transition']
    R = mod['selection']
    Q = mod['state_cov', ..., 0]
    L = np.linalg.cholesky(Q)
    desired_irfs = np.zeros((mod.nobs - 1, 2)) * np.nan
    desired_oirfs = np.zeros((mod.nobs - 1, 2)) * np.nan
    tmp = R[..., 0]
    for i in range(1, mod.nobs):
        desired_irfs[i - 1] = Z[:, :, i].dot(tmp)[:, 0]
        desired_oirfs[i - 1] = Z[:, :, i].dot(tmp).dot(L)[:, 0]
        tmp = T[:, :, i].dot(tmp)
    assert_allclose(irfs, desired_irfs)
    assert_allclose(irfs_anchor, desired_irfs)
    assert_allclose(cirfs, np.cumsum(desired_irfs, axis=0))
    assert_allclose(oirfs, desired_oirfs)
    assert_allclose(coirfs, np.cumsum(desired_oirfs, axis=0))

def test_time_varying_out_of_sample(reset_randomstate):
    if False:
        for i in range(10):
            print('nop')
    mod = TVSS(np.zeros((10, 2)))
    new_Z = np.random.normal(size=mod['design', :, :, -1].shape)
    new_T = np.random.normal(size=mod['transition', :, :, -1].shape)
    irfs = mod.impulse_responses([], steps=mod.nobs, design=new_Z[:, :, None], transition=new_T[:, :, None])
    irfs_anchor = mod.impulse_responses([], steps=mod.nobs, anchor=0, design=new_Z[:, :, None], transition=new_T[:, :, None])
    cirfs = mod.impulse_responses([], steps=mod.nobs, design=new_Z[:, :, None], transition=new_T[:, :, None], cumulative=True)
    oirfs = mod.impulse_responses([], steps=mod.nobs, design=new_Z[:, :, None], transition=new_T[:, :, None], orthogonalized=True)
    coirfs = mod.impulse_responses([], steps=mod.nobs, design=new_Z[:, :, None], transition=new_T[:, :, None], cumulative=True, orthogonalized=True)
    Z = mod['design']
    T = mod['transition']
    R = mod['selection']
    Q = mod['state_cov', ..., 0]
    L = np.linalg.cholesky(Q)
    desired_irfs = np.zeros((mod.nobs, 2)) * np.nan
    desired_oirfs = np.zeros((mod.nobs, 2)) * np.nan
    tmp = R[..., 0]
    for i in range(1, mod.nobs):
        desired_irfs[i - 1] = Z[:, :, i].dot(tmp)[:, 0]
        desired_oirfs[i - 1] = Z[:, :, i].dot(tmp).dot(L)[:, 0]
        tmp = T[:, :, i].dot(tmp)
    desired_irfs[mod.nobs - 1] = new_Z.dot(tmp)[:, 0]
    desired_oirfs[mod.nobs - 1] = new_Z.dot(tmp).dot(L)[:, 0]
    assert_allclose(irfs, desired_irfs)
    assert_allclose(irfs_anchor, desired_irfs)
    assert_allclose(cirfs, np.cumsum(desired_irfs, axis=0))
    assert_allclose(oirfs, desired_oirfs)
    assert_allclose(coirfs, np.cumsum(desired_oirfs, axis=0))

def test_time_varying_in_sample_anchored(reset_randomstate):
    if False:
        return 10
    mod = TVSS(np.zeros((10, 2)))
    anchor = 2
    irfs = mod.impulse_responses([], steps=mod.nobs - 1 - anchor, anchor=anchor)
    cirfs = mod.impulse_responses([], steps=mod.nobs - 1 - anchor, anchor=anchor, cumulative=True)
    oirfs = mod.impulse_responses([], steps=mod.nobs - 1 - anchor, anchor=anchor, orthogonalized=True)
    coirfs = mod.impulse_responses([], steps=mod.nobs - 1 - anchor, anchor=anchor, cumulative=True, orthogonalized=True)
    Z = mod['design']
    T = mod['transition']
    R = mod['selection']
    Q = mod['state_cov', ..., anchor]
    L = np.linalg.cholesky(Q)
    desired_irfs = np.zeros((mod.nobs - anchor - 1, 2)) * np.nan
    desired_oirfs = np.zeros((mod.nobs - anchor - 1, 2)) * np.nan
    tmp = R[..., anchor]
    for i in range(1, mod.nobs - anchor):
        desired_irfs[i - 1] = Z[:, :, i + anchor].dot(tmp)[:, 0]
        desired_oirfs[i - 1] = Z[:, :, i + anchor].dot(tmp).dot(L)[:, 0]
        tmp = T[:, :, i + anchor].dot(tmp)
    assert_allclose(irfs, desired_irfs)
    assert_allclose(cirfs, np.cumsum(desired_irfs, axis=0))
    assert_allclose(oirfs, desired_oirfs)
    assert_allclose(coirfs, np.cumsum(desired_oirfs, axis=0))

def test_time_varying_out_of_sample_anchored(reset_randomstate):
    if False:
        for i in range(10):
            print('nop')
    mod = TVSS(np.zeros((10, 2)))
    anchor = 2
    new_Z = mod['design', :, :, -1]
    new_T = mod['transition', :, :, -1]
    irfs = mod.impulse_responses([], steps=mod.nobs - anchor, anchor=anchor, design=new_Z[:, :, None], transition=new_T[:, :, None])
    cirfs = mod.impulse_responses([], steps=mod.nobs - anchor, anchor=anchor, design=new_Z[:, :, None], transition=new_T[:, :, None], cumulative=True)
    oirfs = mod.impulse_responses([], steps=mod.nobs - anchor, anchor=anchor, design=new_Z[:, :, None], transition=new_T[:, :, None], orthogonalized=True)
    coirfs = mod.impulse_responses([], steps=mod.nobs - anchor, anchor=anchor, design=new_Z[:, :, None], transition=new_T[:, :, None], cumulative=True, orthogonalized=True)
    Z = mod['design']
    T = mod['transition']
    R = mod['selection']
    Q = mod['state_cov', ..., anchor]
    L = np.linalg.cholesky(Q)
    desired_irfs = np.zeros((mod.nobs - anchor, 2)) * np.nan
    desired_oirfs = np.zeros((mod.nobs - anchor, 2)) * np.nan
    tmp = R[..., anchor]
    for i in range(1, mod.nobs - anchor):
        desired_irfs[i - 1] = Z[:, :, i + anchor].dot(tmp)[:, 0]
        desired_oirfs[i - 1] = Z[:, :, i + anchor].dot(tmp).dot(L)[:, 0]
        tmp = T[:, :, i + anchor].dot(tmp)
    desired_irfs[mod.nobs - anchor - 1] = new_Z.dot(tmp)[:, 0]
    desired_oirfs[mod.nobs - anchor - 1] = new_Z.dot(tmp).dot(L)[:, 0]
    assert_allclose(irfs, desired_irfs)
    assert_allclose(cirfs, np.cumsum(desired_irfs, axis=0))
    assert_allclose(oirfs, desired_oirfs)
    assert_allclose(coirfs, np.cumsum(desired_oirfs, axis=0))

def test_time_varying_out_of_sample_anchored_end(reset_randomstate):
    if False:
        while True:
            i = 10
    mod = TVSS(np.zeros((10, 2)))
    with pytest.raises(ValueError, match='Model has time-varying'):
        mod.impulse_responses([], steps=2, anchor='end')
    new_Z = np.random.normal(size=mod['design', :, :, -2:].shape)
    new_T = np.random.normal(size=mod['transition', :, :, -2:].shape)
    irfs = mod.impulse_responses([], steps=2, anchor='end', design=new_Z, transition=new_T)
    cirfs = mod.impulse_responses([], steps=2, anchor='end', design=new_Z, transition=new_T, cumulative=True)
    oirfs = mod.impulse_responses([], steps=2, anchor='end', design=new_Z, transition=new_T, orthogonalized=True)
    coirfs = mod.impulse_responses([], steps=2, anchor='end', design=new_Z, transition=new_T, cumulative=True, orthogonalized=True)
    R = mod['selection']
    Q = mod['state_cov', ..., -1]
    L = np.linalg.cholesky(Q)
    desired_irfs = np.zeros((2, 2)) * np.nan
    desired_oirfs = np.zeros((2, 2)) * np.nan
    tmp = R[..., -1]
    desired_irfs[0] = new_Z[:, :, 0].dot(tmp)[:, 0]
    desired_oirfs[0] = new_Z[:, :, 0].dot(tmp).dot(L)[:, 0]
    tmp = new_T[..., 0].dot(tmp)
    desired_irfs[1] = new_Z[:, :, 1].dot(tmp)[:, 0]
    desired_oirfs[1] = new_Z[:, :, 1].dot(tmp).dot(L)[:, 0]
    assert_allclose(irfs, desired_irfs)
    assert_allclose(cirfs, np.cumsum(desired_irfs, axis=0))
    assert_allclose(oirfs, desired_oirfs)
    assert_allclose(coirfs, np.cumsum(desired_oirfs, axis=0))

def test_pandas_univariate_rangeindex():
    if False:
        for i in range(10):
            print('nop')
    endog = pd.Series(np.zeros(1))
    mod = sarimax.SARIMAX(endog)
    res = mod.filter([0.5, 1.0])
    actual = res.impulse_responses(2)
    desired = pd.Series([1.0, 0.5, 0.25])
    assert_allclose(res.impulse_responses(2), desired)
    assert_(actual.index.equals(desired.index))

def test_pandas_univariate_dateindex():
    if False:
        while True:
            i = 10
    ix = pd.date_range(start='2000', periods=1, freq='M')
    endog = pd.Series(np.zeros(1), index=ix)
    mod = sarimax.SARIMAX(endog)
    res = mod.filter([0.5, 1.0])
    actual = res.impulse_responses(2)
    desired = pd.Series([1.0, 0.5, 0.25])
    assert_allclose(res.impulse_responses(2), desired)
    assert_(actual.index.equals(desired.index))

def test_pandas_multivariate_rangeindex():
    if False:
        while True:
            i = 10
    endog = pd.DataFrame(np.zeros((1, 2)))
    mod = varmax.VARMAX(endog, trend='n')
    res = mod.filter([0.5, 0.0, 0.0, 0.2, 1.0, 0.0, 1.0])
    actual = res.impulse_responses(2)
    desired = pd.DataFrame([[1.0, 0.5, 0.25], [0.0, 0.0, 0.0]]).T
    assert_allclose(actual, desired)
    assert_(actual.index.equals(desired.index))

def test_pandas_multivariate_dateindex():
    if False:
        for i in range(10):
            print('nop')
    ix = pd.date_range(start='2000', periods=1, freq='M')
    endog = pd.DataFrame(np.zeros((1, 2)), index=ix)
    mod = varmax.VARMAX(endog, trend='n')
    res = mod.filter([0.5, 0.0, 0.0, 0.2, 1.0, 0.0, 1.0])
    actual = res.impulse_responses(2)
    desired = pd.DataFrame([[1.0, 0.5, 0.25], [0.0, 0.0, 0.0]]).T
    assert_allclose(actual, desired)
    assert_(actual.index.equals(desired.index))

def test_pandas_anchor():
    if False:
        return 10
    ix = pd.date_range(start='2000', periods=10, freq='M')
    endog = pd.DataFrame(np.zeros((10, 2)), index=ix)
    mod = TVSS(endog)
    res = mod.filter([])
    desired = res.impulse_responses(2, anchor=1)
    actual = res.impulse_responses(2, anchor=ix[1])
    assert_allclose(actual, desired)
    assert_(actual.index.equals(desired.index))
    actual = res.impulse_responses(2, anchor=-9)
    assert_allclose(actual, desired)
    assert_(actual.index.equals(desired.index))