import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel
data = sm.datasets.spector.load_pandas()
exog = data.exog
endog = data.endog
print(sm.datasets.spector.NOTE)
print(data.exog.head())
exog = sm.add_constant(exog, prepend=True)

class MyProbit(GenericLikelihoodModel):

    def loglike(self, params):
        if False:
            return 10
        exog = self.exog
        endog = self.endog
        q = 2 * endog - 1
        return stats.norm.logcdf(q * np.dot(exog, params)).sum()
sm_probit_manual = MyProbit(endog, exog).fit()
print(sm_probit_manual.summary())
sm_probit_canned = sm.Probit(endog, exog).fit()
print(sm_probit_canned.params)
print(sm_probit_manual.params)
print(sm_probit_canned.cov_params())
print(sm_probit_manual.cov_params())
import numpy as np
from scipy.stats import nbinom

def _ll_nb2(y, X, beta, alph):
    if False:
        i = 10
        return i + 15
    mu = np.exp(np.dot(X, beta))
    size = 1 / alph
    prob = size / (size + mu)
    ll = nbinom.logpmf(y, size, prob)
    return ll
from statsmodels.base.model import GenericLikelihoodModel

class NBin(GenericLikelihoodModel):

    def __init__(self, endog, exog, **kwds):
        if False:
            for i in range(10):
                print('nop')
        super(NBin, self).__init__(endog, exog, **kwds)

    def nloglikeobs(self, params):
        if False:
            return 10
        alph = params[-1]
        beta = params[:-1]
        ll = _ll_nb2(self.endog, self.exog, beta, alph)
        return -ll

    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        if False:
            return 10
        self.exog_names.append('alpha')
        if start_params is None:
            start_params = np.append(np.zeros(self.exog.shape[1]), 0.5)
            start_params[-2] = np.log(self.endog.mean())
        return super(NBin, self).fit(start_params=start_params, maxiter=maxiter, maxfun=maxfun, **kwds)
import statsmodels.api as sm
medpar = sm.datasets.get_rdataset('medpar', 'COUNT', cache=True).data
medpar.head()
y = medpar.los
X = medpar[['type2', 'type3', 'hmo', 'white']].copy()
X['constant'] = 1
mod = NBin(y, X)
res = mod.fit()
print('Parameters: ', res.params)
print('Standard errors: ', res.bse)
print('P-values: ', res.pvalues)
print('AIC: ', res.aic)
print(res.summary())
res_nbin = sm.NegativeBinomial(y, X).fit(disp=0)
print(res_nbin.summary())
print(res_nbin.params)
print(res_nbin.bse)