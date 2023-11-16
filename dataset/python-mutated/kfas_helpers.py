import numpy as np
import pandas as pd
from statsmodels.tools.tools import Bunch

def parse(path, ssm):
    if False:
        for i in range(10):
            print('nop')
    n = ssm.nobs
    p = ssm.k_endog
    m = ssm.k_states
    r = ssm.k_posdef
    p2 = p ** 2
    m2 = m ** 2
    mp = m * p
    r2 = r ** 2
    kfas = pd.read_csv(path)
    components = [('r', m), ('r0', m), ('r1', m), ('N', m2), ('N0', m2), ('N1', m2), ('N2', m2), ('m', p), ('v', p), ('F', p), ('Finf', p), ('K', mp), ('Kinf', mp), ('a', m), ('P', m2), ('Pinf', m2), ('att', m), ('Ptt', m2), ('alphahat', m), ('V', m2), ('muhat', p), ('V_mu', p2), ('etahat', r), ('V_eta', r2), ('epshat', p), ('V_eps', p), ('llf', 1)]
    dta = {}
    ix = 0
    for (key, length) in components:
        dta[key] = kfas.iloc[:, ix:ix + length].fillna(0)
        dta[key].name = None
        ix += length
    res = Bunch()
    d = len(kfas['Pinf_1'].dropna())
    res['forecasts'] = dta['m'].values[:n].T
    res['forecasts_error'] = dta['v'].values[:n].T
    res['forecasts_error_cov'] = np.c_[[np.diag(x) for (y, x) in dta['F'].iloc[:n].iterrows()]].T
    res['forecasts_error_diffuse_cov'] = np.c_[[np.diag(x) for (y, x) in dta['Finf'].iloc[:n].iterrows()]].T
    res['kalman_gain'] = dta['K'].values[:n].reshape(n, m, p, order='F').T
    res['Kinf'] = dta['Kinf'].values[:n].reshape(n, m, p, order='F')
    res['filtered_state'] = dta['att'].values[:n].T
    res['filtered_state_cov'] = dta['Ptt'].values[:n].reshape(n, m, m, order='F').T
    res['predicted_state'] = dta['a'].values.T
    res['predicted_state_cov'] = dta['P'].values.reshape(n + 1, m, m, order='F').T
    res['predicted_diffuse_state_cov'] = dta['Pinf'].values.reshape(n + 1, m, m, order='F').T
    res['llf_obs'] = dta['llf'].values[0, 0]
    res['smoothed_state'] = dta['alphahat'].values[:n].T
    res['smoothed_state_cov'] = dta['V'].values[:n].reshape(n, m, m, order='F').T
    res['smoothed_measurement_disturbance'] = dta['epshat'].values[:n].T
    res['smoothed_measurement_disturbance_cov'] = np.c_[[np.diag(x) for (y, x) in dta['V_eps'].iloc[:n].iterrows()]].T
    res['smoothed_state_disturbance'] = dta['etahat'].values[:n].T
    res['smoothed_state_disturbance_cov'] = dta['V_eta'].values[:n].reshape(n, r, r, order='F').T
    res['scaled_smoothed_estimator'] = np.c_[dta['r0'][:d].T, dta['r'][d:].T][..., 1:]
    res['scaled_smoothed_diffuse_estimator'] = dta['r1'].values.T
    N0 = dta['N0'].values[:d].reshape(d, m, m, order='F')
    N = dta['N'].values[d:].reshape(n + 1 - d, m, m, order='F')
    res['scaled_smoothed_estimator_cov'] = np.c_[N0.T, N.T][..., 1:]
    res['scaled_smoothed_diffuse1_estimator_cov'] = dta['N1'].values.reshape(n + 1, m, m, order='F').transpose(1, 2, 0)
    res['scaled_smoothed_diffuse2_estimator_cov'] = dta['N2'].values.reshape(n + 1, m, m, order='F').transpose(1, 2, 0)
    return res