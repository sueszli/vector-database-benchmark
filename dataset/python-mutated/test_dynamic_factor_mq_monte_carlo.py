"""
Monte Carlo-type tests for the BM model

Note that that the actual tests that run are just regression tests against
previously estimated values with small sample sizes that can be run quickly
for continuous integration. However, this file can be used to re-run (slow)
large-sample Monte Carlo tests.
"""
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy.signal import lfilter
from statsmodels.tsa.statespace import dynamic_factor_mq, sarimax, varmax, dynamic_factor

def simulate_k_factor1(nobs=1000):
    if False:
        print('Hello World!')
    mod_sim = dynamic_factor.DynamicFactor(np.zeros((1, 4)), k_factors=1, factor_order=1, error_order=1)
    loadings = [1.0, -0.75, 0.25, -0.3, 0.5]
    p = np.r_[loadings[:mod_sim.k_endog], [10] * mod_sim.k_endog, 0.5, [0.0] * mod_sim.k_endog]
    ix = pd.period_range(start='1935-01', periods=nobs, freq='M')
    endog = pd.DataFrame(mod_sim.simulate(p, nobs), index=ix)
    true = pd.Series(p, index=mod_sim.param_names)
    ix = pd.period_range(start=endog.index[0] - 1, end=endog.index[-1], freq=endog.index.freq)
    levels_M = 1 + endog.reindex(ix) / 100
    levels_M.iloc[0] = 100
    levels_M = levels_M.cumprod()
    log_levels_M = np.log(levels_M) * 100
    log_levels_Q = np.log(levels_M).resample('Q', convention='e').sum().iloc[:-1] * 100
    endog_M = log_levels_M.iloc[:, :3].diff()
    endog_Q = log_levels_Q.iloc[:, 3:].diff()
    return (endog_M, endog_Q, log_levels_M, log_levels_Q, true)

def simulate_k_factors3_blocks2(nobs=1000, idiosyncratic_ar1=False):
    if False:
        for i in range(10):
            print('nop')
    ix = pd.period_range(start='2000-01', periods=1, freq='M')
    endog = pd.DataFrame(np.zeros((1, 2)), columns=['f1', 'f2'], index=ix)
    mod_f_12 = varmax.VARMAX(endog, order=(1, 0), trend='n')
    params = [0.5, 0.1, -0.2, 0.9, 1.0, 0, 1.0]
    f_12 = mod_f_12.simulate(params, nobs)
    endog = pd.Series([0], name='f3', index=ix)
    mod_f_3 = sarimax.SARIMAX(endog, order=(2, 0, 0))
    params = [0.7, 0.1, 1.0]
    f_3 = mod_f_3.simulate(params, nobs)
    f = pd.concat([f_12, f_3], axis=1)
    k_endog = 8
    design = np.zeros((k_endog, 3))
    design[0] = [1.0, 1.0, 1.0]
    design[1] = [0.5, -0.8, 0.0]
    design[2] = [1.0, 0.0, 0.0]
    design[3] = [0.2, 0.0, -0.1]
    design[4] = [0.5, 0.0, 0.0]
    design[5] = [-0.2, 0.0, 0.0]
    design[6] = [1.0, 1.0, 1.0]
    design[7] = [-1.0, 0.0, 0.0]
    rho = np.array([0.5, 0.2, -0.1, 0.0, 0.4, 0.9, 0.05, 0.05])
    if not idiosyncratic_ar1:
        rho *= 0.0
    eps = [lfilter([1], [1, -rho[i]], np.random.normal(size=nobs)) for i in range(k_endog)]
    endog = (design @ f.T).T + eps
    endog.columns = [f'y{i + 1}' for i in range(k_endog)]
    tmp1 = design.ravel()
    tmp2 = np.linalg.cholesky(mod_f_12['state_cov'])
    tmp3 = rho if idiosyncratic_ar1 else []
    true = np.r_[tmp1[tmp1 != 0], mod_f_12['transition', :2, :].ravel(), mod_f_3['transition', :, 0], tmp2[np.tril_indices_from(tmp2)], mod_f_3['state_cov', 0, 0], tmp3, [1] * k_endog]
    ix = pd.period_range(endog.index[0] - 1, endog.index[-1], freq='M')
    levels_M = 1 + endog.reindex(ix) / 100
    levels_M.iloc[0] = 100
    levels_M = levels_M.cumprod()
    log_levels_M = np.log(levels_M) * 100
    log_levels_Q = np.log(levels_M).resample('Q', convention='e').sum().iloc[:-1] * 100
    endog_M = log_levels_M.iloc[:, :7].diff().iloc[1:]
    endog_Q = log_levels_Q.iloc[:, 7:].diff().iloc[2:]
    factor_names = np.array(['global', 'second', 'third'])
    factors = {endog.columns[i]: factor_names[design[i] != 0] for i in range(k_endog)}
    factor_orders = {('global', 'second'): 1, 'third': 2}
    return (endog_M, endog_Q, log_levels_M, log_levels_Q, factors, factor_orders, true, f)

@pytest.mark.skip(reason='Monte carlo test, very slow, kept for manual runs')
def test_k_factor1(reset_randomstate):
    if False:
        print('Hello World!')
    np.random.seed(1234)
    (endog_M, endog_Q, _, _, true_params, _) = simulate_k_factor1(nobs=100000)
    mod = dynamic_factor_mq.DynamicFactorMQ(endog_M, endog_quarterly=endog_Q, factors=1, factor_orders=1, idiosyncratic_ar1=False)
    with mod.fix_params({'fb(0).cov.chol[1,1]': 1.0}):
        mod.fit(method='lbfgs', disp=False)

def gen_k_factor1_nonstationary(nobs=1000, k=1, idiosyncratic_ar1=False, idiosyncratic_var=0.4, k_ar=1):
    if False:
        return 10
    ix = pd.period_range(start='1950-01', periods=1, freq='M')
    faux = pd.Series([0], index=ix)
    mod = sarimax.SARIMAX(faux, order=(k_ar, 0, 0), initialization='diffuse')
    params = np.r_[[0] * (k_ar - 1), [1.0001], 1.0]
    factor = mod.simulate(params, nobs)
    if idiosyncratic_ar1:
        mod_idio = sarimax.SARIMAX(faux, order=(1, 0, 0))
        endog = pd.concat([factor + mod_idio.simulate([0.7, idiosyncratic_var], nobs) for i in range(2 * k)], axis=1)
    else:
        endog = pd.concat([factor + np.random.normal(scale=idiosyncratic_var ** 0.5, size=nobs) for i in range(2 * k)], axis=1)
    levels_M = 1 + endog / 100
    levels_M.iloc[0] = 100
    levels_M = levels_M.cumprod()
    log_levels_M = np.log(levels_M) * 100
    log_levels_Q = np.log(levels_M).resample('Q', convention='e').sum().iloc[:-1] * 100
    endog_M = log_levels_M.diff().iloc[1:, :k]
    if k > 1:
        endog_M.columns = ['yM%d_f1' % (i + 1) for i in range(k)]
    else:
        endog_M.columns = ['yM_f1']
    endog_Q = log_levels_Q.diff().iloc[1:, k:]
    if k > 1:
        endog_Q.columns = ['yQ%d_f1' % (i + 1) for i in range(k)]
    else:
        endog_Q.columns = ['yQ_f1']
    return (endog_M, endog_Q, factor)

def test_em_nonstationary(reset_randomstate):
    if False:
        return 10
    ix = pd.period_range(start='2000', periods=20, freq='M')
    endog_M = pd.Series(np.arange(20), index=ix, dtype=float)
    endog_M.iloc[10:12] += [0.4, -0.2]
    ix = pd.period_range(start='2000', periods=5, freq='Q')
    endog_Q = pd.Series(np.arange(5), index=ix)
    mod = dynamic_factor_mq.DynamicFactorMQ(endog_M, endog_quarterly=endog_Q, idiosyncratic_ar1=False, standardize=False, factors=['global'])
    msg = 'Non-stationary parameters found at EM iteration 1, which is not compatible with stationary initialization. Initialization was switched to diffuse for the following:  \\["factor block: \\(\\\'global\\\',\\)"\\], and fitting was restarted.'
    with pytest.warns(UserWarning, match=msg):
        mod.fit(maxiter=2, em_initialization=False)

def gen_k_factor1(nobs=10000, k=1, idiosyncratic_ar1=False, idiosyncratic_var=0.4, k_ar=6):
    if False:
        while True:
            i = 10
    ix = pd.period_range(start='1950-01', periods=1, freq='M')
    faux = pd.Series([0], index=ix)
    mod = sarimax.SARIMAX(faux, order=(k_ar, 0, 0))
    params = np.r_[[0] * (k_ar - 1), [0.5], 1.0]
    factor = mod.simulate(params, nobs)
    if idiosyncratic_ar1:
        mod_idio = sarimax.SARIMAX(faux, order=(1, 0, 0))
        endog = pd.concat([factor + mod_idio.simulate([0.7, idiosyncratic_var], nobs) for i in range(2 * k)], axis=1)
    else:
        endog = pd.concat([factor + np.random.normal(scale=idiosyncratic_var ** 0.5, size=nobs) for i in range(2 * k)], axis=1)
    levels_M = 1 + endog / 100
    levels_M.iloc[0] = 100
    levels_M = levels_M.cumprod()
    log_levels_M = np.log(levels_M) * 100
    log_levels_Q = np.log(levels_M).resample('Q', convention='e').sum().iloc[:-1] * 100
    endog_M = log_levels_M.diff().iloc[1:, :k]
    if k > 1:
        endog_M.columns = ['yM%d_f1' % (i + 1) for i in range(k)]
    else:
        endog_M.columns = ['yM_f1']
    endog_Q = log_levels_Q.diff().iloc[1:, k:]
    if k > 1:
        endog_Q.columns = ['yQ%d_f1' % (i + 1) for i in range(k)]
    else:
        endog_Q.columns = ['yQ_f1']
    return (endog_M, endog_Q, factor)

@pytest.mark.filterwarnings('ignore:Log-likelihood decreased')
def test_k_factor1_factor_order_6(reset_randomstate):
    if False:
        for i in range(10):
            print('nop')
    (endog_M, endog_Q, _) = gen_k_factor1(nobs=100, idiosyncratic_var=0.0)
    mod = dynamic_factor_mq.DynamicFactorMQ(endog_M, endog_quarterly=endog_Q, factor_orders=6, idiosyncratic_ar1=False, standardize=False)
    mod.fit()

def gen_k_factor2(nobs=10000, k=2, idiosyncratic_ar1=False, idiosyncratic_var=0.4, k_ar=6):
    if False:
        i = 10
        return i + 15
    ix = pd.period_range(start='1950-01', periods=1, freq='M')
    faux = pd.DataFrame([[0, 0]], index=ix, columns=['f1', 'f2'])
    mod = varmax.VARMAX(faux, order=(k_ar, 0), trend='n')
    A = np.zeros((2, 2 * k_ar))
    A[:, -2:] = np.array([[0.5, -0.2], [0.1, 0.3]])
    Q = np.array([[1.5, 0.2], [0.2, 0.5]])
    L = np.linalg.cholesky(Q)
    params = np.r_[A.ravel(), L[np.tril_indices_from(L)]]
    factors = mod.simulate(params, nobs)
    faux = pd.Series([0], index=ix)
    mod_idio = sarimax.SARIMAX(faux, order=(1, 0, 0))
    phi = [0.7, -0.2] if idiosyncratic_ar1 else [0, 0.0]
    tmp = factors.iloc[:, 0] + factors.iloc[:, 1]
    endog_M = pd.concat([tmp.copy() for i in range(k)], axis=1)
    columns = []
    for i in range(k):
        endog_M.iloc[:, i] = endog_M.iloc[:, i] + mod_idio.simulate([phi[0], idiosyncratic_var], nobs)
        columns += [f'yM{i + 1}_f2']
    endog_M.columns = columns
    endog_Q_M = pd.concat([tmp.copy() for i in range(k)], axis=1)
    columns = []
    for i in range(k):
        endog_Q_M.iloc[:, i] = endog_Q_M.iloc[:, i] + mod_idio.simulate([phi[0], idiosyncratic_var], nobs)
        columns += [f'yQ{i + 1}_f2']
    endog_Q_M.columns = columns
    levels_M = 1 + endog_Q_M / 100
    levels_M.iloc[0] = 100
    levels_M = levels_M.cumprod()
    log_levels_Q = np.log(levels_M).resample('Q', convention='e').sum().iloc[:-1] * 100
    endog_Q = log_levels_Q.diff()
    return (endog_M, endog_Q, factors)

@pytest.mark.skip(reason='Monte carlo test, very slow, kept for manual runs')
def test_k_factor2_factor_order_6(reset_randomstate):
    if False:
        for i in range(10):
            print('nop')
    (endog_M, endog_Q, factors) = gen_k_factor2()
    endog_M_aug = pd.concat([factors, endog_M], axis=1)
    mod = dynamic_factor_mq.DynamicFactorMQ(endog_M_aug, endog_quarterly=endog_Q, factor_multiplicities=2, factor_orders=6, idiosyncratic_ar1=False, standardize=False)
    res = mod.fit()
    M = np.kron(np.eye(6), mod['design', :2, :2])
    Mi = np.linalg.inv(M)
    Z = mod['design', :, :12]
    A = mod['transition', :12, :12]
    R = mod['selection', :12, :2]
    Q = mod['state_cov', :2, :2]
    RQR = R @ Q @ R.T
    Z2 = Z @ Mi
    A2 = M @ A @ Mi
    Q2 = M @ RQR @ M.T
    print(Z2.round(2))
    desired = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 2, 2, 3, 3, 2, 2, 1, 1, 0, 0], [1, 1, 2, 2, 3, 3, 2, 2, 1, 1, 0, 0]])
    assert_allclose(Z2, desired, atol=0.1)
    print(A2.round(2))
    desired = np.array([[0, 0, 0.02, 0, 0.01, -0.03, 0.01, 0.02, 0, -0.01, 0.5, -0.2], [0, 0, 0, 0.02, 0, -0.01, 0, 0, 0, 0.01, 0.1, 0.3], [1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0]])
    assert_allclose(A2, desired, atol=0.01)
    print(Q2.round(2))
    desired = np.array([[1.49, 0.21], [0.21, 0.49]])
    assert_allclose(Q2[:2, :2], desired, atol=0.01)
    assert_allclose(Q2[:2, 2:], 0, atol=0.01)
    assert_allclose(Q2[2:, :2], 0, atol=0.01)
    assert_allclose(Q2[2:, 2:], 0, atol=0.01)
    a = res.states.smoothed
    a2 = (M @ a.T.iloc[:12]).T
    assert_allclose(endog_M.values, a2.iloc[:, :2].values, atol=1e-10)

@pytest.mark.skip(reason='Monte carlo test, very slow, kept for manual runs')
def test_two_blocks_factor_orders_6(reset_randomstate):
    if False:
        for i in range(10):
            print('nop')
    nobs = 1000
    idiosyncratic_ar1 = True
    k1 = 3
    k2 = 10
    (endog1_M, endog1_Q, f1) = gen_k_factor1(nobs, k=k1, idiosyncratic_ar1=idiosyncratic_ar1)
    (endog2_M, endog2_Q, f2) = gen_k_factor2(nobs, k=k2, idiosyncratic_ar1=idiosyncratic_ar1)
    endog_M = pd.concat([endog1_M, f2, endog2_M], axis=1)
    endog_Q = pd.concat([endog1_Q, endog2_Q], axis=1)
    factors = {f'yM{i + 1}_f1': ['a'] for i in range(k1)}
    factors.update({f'yQ{i + 1}_f1': ['a'] for i in range(k1)})
    factors.update({f'f{i + 1}': ['b'] for i in range(2)})
    factors.update({f'yM{i + 1}_f2': ['b'] for i in range(k2)})
    factors.update({f'yQ{i + 1}_f2': ['b'] for i in range(k2)})
    factor_multiplicities = {'b': 2}
    mod = dynamic_factor_mq.DynamicFactorMQ(endog_M, endog_quarterly=endog_Q, factors=factors, factor_multiplicities=factor_multiplicities, factor_orders=6, idiosyncratic_ar1=idiosyncratic_ar1, standardize=False)
    mod.fit()
    from scipy.linalg import block_diag
    M1 = np.kron(np.eye(6), mod['design', 3:5, :2])
    M2 = np.kron(np.eye(6), mod['design', 0:1, 12:13])
    M = block_diag(M1, M2)
    Mi = np.linalg.inv(M)
    Z = mod['design', :, :18]
    A = mod['transition', :18, :18]
    R = mod['selection', :18, :3]
    Q = block_diag(mod['state_cov', :2, :2], mod['state_cov', 12:13, 12:13])
    RQR = R @ Q @ R.T
    Z2 = Z @ Mi
    A2 = M @ A @ Mi
    Q2 = M @ RQR @ M.T
    print(Z2.round(2))
    print(A2.round(2))
    print(Q2.round(2))