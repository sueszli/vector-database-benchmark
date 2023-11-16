"""Panel data analysis for short T and large N

Created on Sat Dec 17 19:32:00 2011

Author: Josef Perktold
License: BSD-3


starting from scratch before looking at references again
just a stub to get the basic structure for group handling
target outsource as much as possible for reuse

Notes
-----

this is the basic version using a loop over individuals which will be more
widely applicable. Depending on the special cases, there will be faster
implementations possible (sparse, kroneker, ...)

the only two group specific methods or get_within_cov and whiten

"""
import numpy as np
from statsmodels.regression.linear_model import OLS, GLS
from statsmodels.tools.grouputils import GroupSorted

def sum_outer_product_loop(x, group_iter):
    if False:
        for i in range(10):
            print('nop')
    'sum outerproduct dot(x_i, x_i.T) over individuals\n\n    loop version\n\n    '
    mom = 0
    for g in group_iter():
        x_g = x[g]
        mom += np.outer(x_g, x_g)
    return mom

def sum_outer_product_balanced(x, n_groups):
    if False:
        print('Hello World!')
    'sum outerproduct dot(x_i, x_i.T) over individuals\n\n    where x_i is (nobs_i, 1), and result is (nobs_i, nobs_i)\n\n    reshape-dot version, for x.ndim=1 only\n\n    '
    xrs = x.reshape(-1, n_groups, order='F')
    return np.dot(xrs, xrs.T)

def whiten_individuals_loop(x, transform, group_iter):
    if False:
        print('Hello World!')
    'apply linear transform for each individual\n\n    loop version\n    '
    x_new = []
    for g in group_iter():
        x_g = x[g]
        x_new.append(np.dot(transform, x_g))
    return np.concatenate(x_new)

class ShortPanelGLS2:
    """Short Panel with general intertemporal within correlation

    assumes data is stacked by individuals, panel is balanced and
    within correlation structure is identical across individuals.

    It looks like this can just inherit GLS and overwrite whiten
    """

    def __init__(self, endog, exog, group):
        if False:
            for i in range(10):
                print('nop')
        self.endog = endog
        self.exog = exog
        self.group = GroupSorted(group)
        self.n_groups = self.group.n_groups

    def fit_ols(self):
        if False:
            return 10
        self.res_pooled = OLS(self.endog, self.exog).fit()
        return self.res_pooled

    def get_within_cov(self, resid):
        if False:
            while True:
                i = 10
        mom = sum_outer_product_loop(resid, self.group.group_iter)
        return mom / self.n_groups

    def whiten_groups(self, x, cholsigmainv_i):
        if False:
            while True:
                i = 10
        wx = whiten_individuals_loop(x, cholsigmainv_i, self.group.group_iter)
        return wx

    def fit(self):
        if False:
            while True:
                i = 10
        res_pooled = self.fit_ols()
        sigma_i = self.get_within_cov(res_pooled.resid)
        self.cholsigmainv_i = np.linalg.cholesky(np.linalg.pinv(sigma_i)).T
        wendog = self.whiten_groups(self.endog, self.cholsigmainv_i)
        wexog = self.whiten_groups(self.exog, self.cholsigmainv_i)
        self.res1 = OLS(wendog, wexog).fit()
        return self.res1

class ShortPanelGLS(GLS):
    """Short Panel with general intertemporal within correlation

    assumes data is stacked by individuals, panel is balanced and
    within correlation structure is identical across individuals.

    It looks like this can just inherit GLS and overwrite whiten
    """

    def __init__(self, endog, exog, group, sigma_i=None):
        if False:
            print('Hello World!')
        self.group = GroupSorted(group)
        self.n_groups = self.group.n_groups
        nobs_i = len(endog) / self.n_groups
        if sigma_i is None:
            sigma_i = np.eye(int(nobs_i))
        self.cholsigmainv_i = np.linalg.cholesky(np.linalg.pinv(sigma_i)).T
        super(self.__class__, self).__init__(endog, exog, sigma=None)

    def get_within_cov(self, resid):
        if False:
            return 10
        mom = sum_outer_product_loop(resid, self.group.group_iter)
        return mom / self.n_groups

    def whiten_groups(self, x, cholsigmainv_i):
        if False:
            return 10
        wx = whiten_individuals_loop(x, cholsigmainv_i, self.group.group_iter)
        return wx

    def _fit_ols(self):
        if False:
            for i in range(10):
                print('nop')
        self.res_pooled = OLS(self.endog, self.exog).fit()
        return self.res_pooled

    def _fit_old(self):
        if False:
            while True:
                i = 10
        res_pooled = self._fit_ols()
        sigma_i = self.get_within_cov(res_pooled.resid)
        self.cholsigmainv_i = np.linalg.cholesky(np.linalg.pinv(sigma_i)).T
        wendog = self.whiten_groups(self.endog, self.cholsigmainv_i)
        wexog = self.whiten_groups(self.exog, self.cholsigmainv_i)
        self.res1 = OLS(wendog, wexog).fit()
        return self.res1

    def whiten(self, x):
        if False:
            i = 10
            return i + 15
        wx = whiten_individuals_loop(x, self.cholsigmainv_i, self.group.group_iter)
        return wx

    def fit_iterative(self, maxiter=3):
        if False:
            return 10
        '\n        Perform an iterative two-step procedure to estimate the GLS model.\n\n        Parameters\n        ----------\n        maxiter : int, optional\n            the number of iterations\n\n        Notes\n        -----\n        maxiter=1: returns the estimated based on given weights\n        maxiter=2: performs a second estimation with the updated weights,\n                   this is 2-step estimation\n        maxiter>2: iteratively estimate and update the weights\n\n        TODO: possible extension stop iteration if change in parameter\n            estimates is smaller than x_tol\n\n        Repeated calls to fit_iterative, will do one redundant pinv_wexog\n        calculation. Calling fit_iterative(maxiter) once does not do any\n        redundant recalculations (whitening or calculating pinv_wexog).\n        '
        if maxiter < 1:
            raise ValueError('maxiter needs to be at least 1')
        import collections
        self.history = collections.defaultdict(list)
        for i in range(maxiter):
            if hasattr(self, 'pinv_wexog'):
                del self.pinv_wexog
            results = self.fit()
            self.history['self_params'].append(results.params)
            if not i == maxiter - 1:
                self.results_old = results
                sigma_i = self.get_within_cov(results.resid)
                self.cholsigmainv_i = np.linalg.cholesky(np.linalg.pinv(sigma_i)).T
                self.initialize()
        return results