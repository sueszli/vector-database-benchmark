import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
np.set_printoptions(precision=4, suppress=True, linewidth=120)
from pandas_datareader.data import DataReader
start = '1979-01-01'
end = '2014-12-01'
indprod = DataReader('IPMAN', 'fred', start=start, end=end)
income = DataReader('W875RX1', 'fred', start=start, end=end)
sales = DataReader('CMRMTSPL', 'fred', start=start, end=end)
emp = DataReader('PAYEMS', 'fred', start=start, end=end)
dta = pd.concat((indprod, income, sales, emp), axis=1)
dta.columns = ['indprod', 'income', 'sales', 'emp']
dta.index.freq = dta.index.inferred_freq
dta.loc[:, 'indprod':'emp'].plot(subplots=True, layout=(2, 2), figsize=(15, 6))
dta['dln_indprod'] = np.log(dta.indprod).diff() * 100
dta['dln_income'] = np.log(dta.income).diff() * 100
dta['dln_sales'] = np.log(dta.sales).diff() * 100
dta['dln_emp'] = np.log(dta.emp).diff() * 100
dta['std_indprod'] = (dta['dln_indprod'] - dta['dln_indprod'].mean()) / dta['dln_indprod'].std()
dta['std_income'] = (dta['dln_income'] - dta['dln_income'].mean()) / dta['dln_income'].std()
dta['std_sales'] = (dta['dln_sales'] - dta['dln_sales'].mean()) / dta['dln_sales'].std()
dta['std_emp'] = (dta['dln_emp'] - dta['dln_emp'].mean()) / dta['dln_emp'].std()
endog = dta.loc['1979-02-01':, 'std_indprod':'std_emp']
mod = sm.tsa.DynamicFactor(endog, k_factors=1, factor_order=2, error_order=2)
initial_res = mod.fit(method='powell', disp=False)
res = mod.fit(initial_res.params, disp=False)
print(res.summary(separate_params=False))
(fig, ax) = plt.subplots(figsize=(13, 3))
dates = endog.index._mpl_repr()
ax.plot(dates, res.factors.filtered[0], label='Factor')
ax.legend()
rec = DataReader('USREC', 'fred', start=start, end=end)
ylim = ax.get_ylim()
ax.fill_between(dates[:-3], ylim[0], ylim[1], rec.values[:-4, 0], facecolor='k', alpha=0.1)
res.plot_coefficients_of_determination(figsize=(8, 2))
usphci = DataReader('USPHCI', 'fred', start='1979-01-01', end='2014-12-01')['USPHCI']
usphci.plot(figsize=(13, 3))
dusphci = usphci.diff()[1:].values

def compute_coincident_index(mod, res):
    if False:
        return 10
    spec = res.specification
    design = mod.ssm['design']
    transition = mod.ssm['transition']
    ss_kalman_gain = res.filter_results.kalman_gain[:, :, -1]
    k_states = ss_kalman_gain.shape[0]
    W1 = np.linalg.inv(np.eye(k_states) - np.dot(np.eye(k_states) - np.dot(ss_kalman_gain, design), transition)).dot(ss_kalman_gain)[0]
    factor_mean = np.dot(W1, dta.loc['1972-02-01':, 'dln_indprod':'dln_emp'].mean())
    factor = res.factors.filtered[0]
    factor *= np.std(usphci.diff()[1:]) / np.std(factor)
    coincident_index = np.zeros(mod.nobs + 1)
    coincident_index[0] = usphci.iloc[0] * factor_mean / dusphci.mean()
    for t in range(0, mod.nobs):
        coincident_index[t + 1] = coincident_index[t] + factor[t] + factor_mean
    coincident_index = pd.Series(coincident_index, index=dta.index).iloc[1:]
    coincident_index *= usphci.loc['1992-07-01'] / coincident_index.loc['1992-07-01']
    return coincident_index
(fig, ax) = plt.subplots(figsize=(13, 3))
coincident_index = compute_coincident_index(mod, res)
dates = endog.index._mpl_repr()
ax.plot(dates, coincident_index, label='Coincident index')
ax.plot(usphci.index._mpl_repr(), usphci, label='USPHCI')
ax.legend(loc='lower right')
ylim = ax.get_ylim()
ax.fill_between(dates[:-3], ylim[0], ylim[1], rec.values[:-4, 0], facecolor='k', alpha=0.1)
from statsmodels.tsa.statespace import tools

class ExtendedDFM(sm.tsa.DynamicFactor):

    def __init__(self, endog, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(ExtendedDFM, self).__init__(endog, k_factors=1, factor_order=4, error_order=2, **kwargs)
        self.parameters['new_loadings'] = 3
        offset = self.parameters['factor_loadings'] + self.parameters['exog'] + self.parameters['error_cov']
        self._params_factor_ar = np.s_[offset:offset + 2]
        self._params_factor_zero = np.s_[offset + 2:offset + 4]

    @property
    def start_params(self):
        if False:
            while True:
                i = 10
        return np.r_[super(ExtendedDFM, self).start_params, 0, 0, 0]

    @property
    def param_names(self):
        if False:
            print('Hello World!')
        return super(ExtendedDFM, self).param_names + ['loading.L%d.f1.%s' % (i, self.endog_names[3]) for i in range(1, 4)]

    def transform_params(self, unconstrained):
        if False:
            for i in range(10):
                print('nop')
        constrained = super(ExtendedDFM, self).transform_params(unconstrained[:-3])
        ar_params = unconstrained[self._params_factor_ar]
        constrained[self._params_factor_ar] = tools.constrain_stationary_univariate(ar_params)
        return np.r_[constrained, unconstrained[-3:]]

    def untransform_params(self, constrained):
        if False:
            i = 10
            return i + 15
        unconstrained = super(ExtendedDFM, self).untransform_params(constrained[:-3])
        ar_params = constrained[self._params_factor_ar]
        unconstrained[self._params_factor_ar] = tools.unconstrain_stationary_univariate(ar_params)
        return np.r_[unconstrained, constrained[-3:]]

    def update(self, params, transformed=True, **kwargs):
        if False:
            i = 10
            return i + 15
        if not transformed:
            params = self.transform_params(params)
        params[self._params_factor_zero] = 0
        super(ExtendedDFM, self).update(params[:-3], transformed=True, **kwargs)
        self.ssm['design', 3, 1:4] = params[-3:]
extended_mod = ExtendedDFM(endog)
initial_extended_res = extended_mod.fit(maxiter=1000, disp=False)
extended_res = extended_mod.fit(initial_extended_res.params, method='nm', maxiter=1000)
print(extended_res.summary(separate_params=False))
extended_res.plot_coefficients_of_determination(figsize=(8, 2))
(fig, ax) = plt.subplots(figsize=(13, 3))
extended_coincident_index = compute_coincident_index(extended_mod, extended_res)
dates = endog.index._mpl_repr()
ax.plot(dates, coincident_index, '-', linewidth=1, label='Basic model')
ax.plot(dates, extended_coincident_index, '--', linewidth=3, label='Extended model')
ax.plot(usphci.index._mpl_repr(), usphci, label='USPHCI')
ax.legend(loc='lower right')
ax.set(title='Coincident indices, comparison')
ylim = ax.get_ylim()
ax.fill_between(dates[:-3], ylim[0], ylim[1], rec.values[:-4, 0], facecolor='k', alpha=0.1)