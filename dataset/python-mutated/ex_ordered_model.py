"""
Created on Mon Aug 24 11:17:06 2015

Author: Josef Perktold
License: BSD-3
"""
import numpy as np
from scipy import stats
import pandas
from statsmodels.miscmodels.ordinal_model import OrderedModel
(nobs, k_vars) = (1000, 3)
x = np.random.randn(nobs, k_vars)
xb = x.dot(np.ones(k_vars))
y_latent = xb + np.random.randn(nobs)
y = np.round(np.clip(y_latent, -2.4, 2.4)).astype(int) + 2
print(np.unique(y))
print(np.bincount(y))
mod = OrderedModel(y, x)
start_ppf = stats.norm.ppf((np.bincount(y) / len(y)).cumsum())
start_threshold = np.concatenate((start_ppf[:1], np.log(np.diff(start_ppf[:-1]))))
start_params = np.concatenate((np.zeros(k_vars), start_threshold))
res = mod.fit(start_params=start_params, maxiter=5000, maxfun=5000)
print(res.params)
res = mod.fit(start_params=start_params, method='bfgs')
print(res.params)
print(np.exp(res.params[-(mod.k_levels - 1):]).cumsum())
predicted = res.model.predict(res.params)
pred_choice = predicted.argmax(1)
print('Fraction of correct choice predictions')
print((y == pred_choice).mean())
print('\ncomparing bincount')
print(np.bincount(res.model.predict(res.params).argmax(1)))
print(np.bincount(res.model.endog))
res_log = OrderedModel(y, x, distr='logit').fit(method='bfgs')
pred_choice_log = res_log.predict().argmax(1)
print((y == pred_choice_log).mean())
print(res_log.summary())
dataf = pandas.read_stata('M:\\josef_new\\scripts\\ologit_ucla.dta')
res_log2 = OrderedModel(np.asarray(dataf['apply']), np.asarray(dataf[['pared', 'public', 'gpa']], float), distr='logit').fit(method='bfgs')
res_log3 = OrderedModel(dataf['apply'].values.codes, np.asarray(dataf[['pared', 'public', 'gpa']], float), distr='logit').fit(method='bfgs')
print(res_log3.summary())
print(OrderedModel(dataf['apply'].values.codes, np.asarray(dataf[['pared', 'public', 'gpa']], float), distr='probit').fit(method='bfgs').summary())

class CLogLog(stats.rv_continuous):

    def _ppf(self, q):
        if False:
            return 10
        return np.log(-np.log(1 - q))

    def _cdf(self, x):
        if False:
            return 10
        return 1 - np.exp(-np.exp(x))
cloglog = CLogLog()
res_cloglog = OrderedModel(dataf['apply'], dataf[['pared', 'public', 'gpa']], distr=cloglog).fit(method='bfgs', disp=False)
print(res_cloglog.summary())