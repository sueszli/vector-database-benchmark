"""
Created on Wed Jul 28 08:28:04 2010

Author: josef-pktd
"""
import numpy as np
from scipy import stats, special, optimize
import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel
from statsmodels.tools.numdiff import approx_hess
import statsmodels.sandbox.distributions.sppatch
np_log = np.log
np_pi = np.pi
sps_gamln = special.gammaln

def maxabs(arr1, arr2):
    if False:
        return 10
    return np.max(np.abs(arr1 - arr2))

def maxabsrel(arr1, arr2):
    if False:
        return 10
    return np.max(np.abs(arr2 / arr1 - 1))
store_params = []

class MyT(GenericLikelihoodModel):
    """Maximum Likelihood Estimation of Linear Model with t-distributed errors

    This is an example for generic MLE which has the same
    statistical model as discretemod.Poisson.

    Except for defining the negative log-likelihood method, all
    methods and results are generic. Gradients and Hessian
    and all resulting statistics are based on numerical
    differentiation.

    """

    def loglike(self, params):
        if False:
            print('Hello World!')
        return -self.nloglikeobs(params).sum(0)

    def nloglikeobs(self, params):
        if False:
            while True:
                i = 10
        '\n        Loglikelihood of Poisson model\n\n        Parameters\n        ----------\n        params : array_like\n            The parameters of the model.\n\n        Returns\n        -------\n        The log likelihood of the model evaluated at `params`\n\n        Notes\n        -----\n        .. math:: \\ln L=\\sum_{i=1}^{n}\\left[-\\lambda_{i}+y_{i}x_{i}^{\\prime}\\beta-\\ln y_{i}!\\right]\n        '
        store_params.append(params)
        if self.fixed_params is not None:
            params = self.expandparams(params)
        beta = params[:-2]
        df = params[-2]
        scale = params[-1]
        loc = np.dot(self.exog, beta)
        endog = self.endog
        x = (endog - loc) / scale
        lPx = sps_gamln((df + 1) / 2) - sps_gamln(df / 2.0)
        lPx -= 0.5 * np_log(df * np_pi) + (df + 1) / 2.0 * np_log(1 + x ** 2 / df)
        lPx -= np_log(scale)
        return -lPx
np.random.seed(98765678)
nobs = 1000
nvars = 6
df = 5
rvs = np.random.randn(nobs, nvars - 1)
data_exog = sm.add_constant(rvs, prepend=False)
xbeta = 0.9 + 0.1 * rvs.sum(1)
data_endog = xbeta + 0.1 * np.random.standard_t(df, size=nobs)
print(data_endog.var())
res_ols = sm.OLS(data_endog, data_exog).fit()
print(res_ols.scale)
print(np.sqrt(res_ols.scale))
print(res_ols.params)
kurt = stats.kurtosis(res_ols.resid)
df_fromkurt = 6.0 / kurt + 4
print(stats.t.stats(df_fromkurt, moments='mvsk'))
print(stats.t.stats(df, moments='mvsk'))
modp = MyT(data_endog, data_exog)
start_value = 0.1 * np.ones(data_exog.shape[1] + 2)
start_value[:nvars] = res_ols.params
start_value[-2] = df_fromkurt
start_value[-1] = np.sqrt(res_ols.scale)
modp.start_params = start_value
fixdf = np.nan * np.zeros(modp.start_params.shape)
fixdf[-2] = 100
fixone = 0
if fixone:
    modp.fixed_params = fixdf
    modp.fixed_paramsmask = np.isnan(fixdf)
    modp.start_params = modp.start_params[modp.fixed_paramsmask]
else:
    modp.fixed_params = None
    modp.fixed_paramsmask = None
resp = modp.fit(start_params=modp.start_params, disp=1, method='nm')
print('\nestimation results t-dist')
print(resp.params)
print(resp.bse)
resp2 = modp.fit(start_params=resp.params, method='Newton')
print('using Newton')
print(resp2.params)
print(resp2.bse)
hb = -approx_hess(modp.start_params, modp.loglike, epsilon=-0.0001)
tmp = modp.loglike(modp.start_params)
print(tmp.shape)
pp = np.array(store_params)
print(pp.min(0))
print(pp.max(0))

class MyPareto(GenericLikelihoodModel):
    """Maximum Likelihood Estimation pareto distribution

    first version: iid case, with constant parameters
    """

    def pdf(self, x, b):
        if False:
            i = 10
            return i + 15
        return b * x ** (-b - 1)

    def loglike(self, params):
        if False:
            return 10
        return -self.nloglikeobs(params).sum(0)

    def nloglikeobs(self, params):
        if False:
            while True:
                i = 10
        if self.fixed_params is not None:
            params = self.expandparams(params)
        b = params[0]
        loc = params[1]
        scale = params[2]
        endog = self.endog
        x = (endog - loc) / scale
        logpdf = np_log(b) - (b + 1.0) * np_log(x)
        logpdf -= np.log(scale)
        logpdf[x < 1] = -10000
        return -logpdf

    def fit_ks(self):
        if False:
            i = 10
            return i + 15
        'fit Pareto with nested optimization\n\n        originally published on stackoverflow\n        this does not trim lower values during ks optimization\n\n        '
        rvs = self.endog
        rvsmin = rvs.min()
        fixdf = np.nan * np.ones(3)
        self.fixed_params = fixdf
        self.fixed_paramsmask = np.isnan(fixdf)

        def pareto_ks(loc, rvs):
            if False:
                print('Hello World!')
            self.fixed_params[1] = loc
            est = self.fit(start_params=self.start_params[self.fixed_paramsmask]).params
            args = (est[0], loc, est[1])
            return stats.kstest(rvs, 'pareto', args)[0]
        locest = optimize.fmin(pareto_ks, rvsmin - 1.5, (rvs,))
        est = stats.pareto.fit_fr(rvs, 0.0, frozen=[np.nan, locest, np.nan])
        args = (est[0], locest[0], est[1])
        return args

    def fit_ks1_trim(self):
        if False:
            while True:
                i = 10
        'fit Pareto with nested optimization\n\n        originally published on stackoverflow\n\n        '
        self.nobs = self.endog.shape[0]
        rvs = np.sort(self.endog)
        rvsmin = rvs.min()

        def pareto_ks(loc, rvs):
            if False:
                for i in range(10):
                    print('nop')
            est = stats.pareto.fit_fr(rvs, frozen=[np.nan, loc, np.nan])
            args = (est[0], loc, est[1])
            return stats.kstest(rvs, 'pareto', args)[0]
        maxind = min(np.floor(self.nobs * 0.95).astype(int), self.nobs - 10)
        res = []
        for trimidx in range(self.nobs // 2, maxind):
            xmin = loc = rvs[trimidx]
            res.append([trimidx, pareto_ks(loc - 1e-10, rvs[trimidx:])])
        res = np.array(res)
        bestidx = res[np.argmin(res[:, 1]), 0].astype(int)
        print(bestidx)
        locest = rvs[bestidx]
        est = stats.pareto.fit_fr(rvs[bestidx:], 1.0, frozen=[np.nan, locest, np.nan])
        args = (est[0], locest, est[1])
        return args

    def fit_ks1(self):
        if False:
            while True:
                i = 10
        'fit Pareto with nested optimization\n\n        originally published on stackoverflow\n\n        '
        rvs = self.endog
        rvsmin = rvs.min()

        def pareto_ks(loc, rvs):
            if False:
                print('Hello World!')
            est = stats.pareto.fit_fr(rvs, 1.0, frozen=[np.nan, loc, np.nan])
            args = (est[0], loc, est[1])
            return stats.kstest(rvs, 'pareto', args)[0]
        locest = optimize.fmin(pareto_ks, rvsmin - 1.5, (rvs,))
        est = stats.pareto.fit_fr(rvs, 1.0, frozen=[np.nan, locest, np.nan])
        args = (est[0], locest[0], est[1])
        return args
y = stats.pareto.rvs(1, loc=0, scale=2, size=nobs)
par_start_params = np.array([1.0, 9.0, 2.0])
mod_par = MyPareto(y)
mod_par.start_params = np.array([1.0, 10.0, 2.0])
mod_par.start_params = np.array([1.0, -9.0, 2.0])
mod_par.fixed_params = None
fixdf = np.nan * np.ones(mod_par.start_params.shape)
fixdf[1] = 9.9
fixone = 0
if fixone:
    mod_par.fixed_params = fixdf
    mod_par.fixed_paramsmask = np.isnan(fixdf)
    mod_par.start_params = mod_par.start_params[mod_par.fixed_paramsmask]
    mod_par.df_model = 2
    mod_par.df_resid = mod_par.endog.shape[0] - mod_par.df_model
    mod_par.data.xnames = ['shape', 'scale']
else:
    mod_par.fixed_params = None
    mod_par.fixed_paramsmask = None
    mod_par.df_model = 3
    mod_par.df_resid = mod_par.endog.shape[0] - mod_par.df_model
    mod_par.data.xnames = ['shape', 'loc', 'scale']
res_par = mod_par.fit(start_params=mod_par.start_params, method='nm', maxfun=10000, maxiter=5000)
res_parks = mod_par.fit_ks1()
print(res_par.params)
print(res_parks)
print(res_par.params[1:].sum(), sum(res_parks[1:]), mod_par.endog.min())
mod_par = MyPareto(y)
mod_par.fixed_params = fixdf
mod_par.fixed_paramsmask = np.isnan(fixdf)
mod_par.df_model = mod_par.fixed_paramsmask.sum()
mod_par.df_resid = mod_par.endog.shape[0] - mod_par.df_model
mod_par.data.xnames = [name for (name, incl) in zip(['shape', 'loc', 'scale'], mod_par.fixed_paramsmask) if incl]
res_par3 = mod_par.start_params = par_start_params[mod_par.fixed_paramsmask]
res5 = mod_par.fit(start_params=mod_par.start_params)
print(res5.summary())
print(res5.t_test([[1, 0]]))
'\n0.0686702747648\n0.0164150896481\n0.128121386381\n[ 0.10370428  0.09921315  0.09676723  0.10457413  0.10201618  0.89964496]\n(array(0.0), array(1.4552599885729831), array(0.0), array(2.5072143354058238))\n(array(0.0), array(1.6666666666666667), array(0.0), array(6.0))\nrepr(start_params) array([ 0.10370428,  0.09921315,  0.09676723,  0.10457413,  0.10201618,\n        0.89964496,  6.39309417,  0.12812139])\nOptimization terminated successfully.\n         Current function value: -679.951339\n         Iterations: 398\n         Function evaluations: 609\n\nestimation results t-dist\n[ 0.10400826  0.10111893  0.09725133  0.10507788  0.10086163  0.8996041\n  4.72131318  0.09825355]\n[ 0.00365493  0.00356149  0.00349329  0.00362333  0.003732    0.00362716\n  0.7232824   0.00388829]\nrepr(start_params) array([ 0.10400826,  0.10111893,  0.09725133,  0.10507788,  0.10086163,\n        0.8996041 ,  4.72131318,  0.09825355])\nOptimization terminated successfully.\n         Current function value: -679.950443\n         Iterations 3\nusing Newton\n[ 0.10395383  0.10106762  0.09720665  0.10503384  0.10080599  0.89954546\n  4.70918964  0.09815885]\n[ 0.00365299  0.00355968  0.00349147  0.00362166  0.00373015  0.00362533\n  0.72014031  0.00388434]\n()\n[ 0.09992709  0.09786601  0.09387356  0.10229919  0.09756623  0.85466272\n  4.60459182  0.09661986]\n[ 0.11308292  0.10828401  0.1028508   0.11268895  0.10934726  0.94462721\n  7.15412655  0.13452746]\nrepr(start_params) array([ 1.,  2.])\nWarning: Maximum number of function evaluations has been exceeded.\n>>> res_par.params\narray([  7.42705803e+152,   2.17339053e+153])\n\n>>> mod_par.loglike(mod_par.start_params)\n-1085.1993430947232\n>>> np.log(mod_par.pdf(*mod_par.start_params))\n0.69314718055994529\n>>> mod_par.loglike(mod_par.start_params)\n-1085.1993430947232\n>>> np.log(stats.pareto.pdf(y[0],*mod_par.start_params))\n-4.6414308627431353\n>>> mod_par.loglike(mod_par.start_params)\n-1085.1993430947232\n>>> mod_par.nloglikeobs(mod_par.start_params)[0]\n0.29377232943845044\n>>> mod_par.start_params\narray([ 1.,  2.])\n>>> np.log(stats.pareto.pdf(y[0],1,9.5,2))\n-1.2806918394368461\n>>> mod_par.fixed_params= None\n>>> mod_par.nloglikeobs(np.array([1., 10., 2.]))[0]\n0.087533156771285828\n>>> y[0]\n12.182956907488885\n\n>>> mod_par.endog[0]\n12.182956907488885\n>>> np.log(stats.pareto.pdf(y[0],1,10,2))\n-0.86821349410251702\n>>> np.log(stats.pareto.pdf(y[0],1.,10.,2.))\n-0.86821349410251702\n>>> stats.pareto.pdf(y[0],1.,10.,2.)\n0.41970067762301644\n>>> mod_par.loglikeobs(np.array([1., 10., 2.]))[0]\n-0.087533156771285828\n>>>\n'
'\n>>> mod_par.nloglikeobs(np.array([1., 10., 2.]))[0]\n0.86821349410251691\n>>> np.log(stats.pareto.pdf(y,1.,10.,2.)).sum()\n-2627.9403758026938\n'
'\nrepr(start_params) array([  1.,  10.,   2.])\nOptimization terminated successfully.\n         Current function value: 2626.436870\n         Iterations: 102\n         Function evaluations: 210\nOptimization terminated successfully.\n         Current function value: 0.016555\n         Iterations: 16\n         Function evaluations: 35\n[  1.03482659  10.00737039   1.9944777 ]\n(1.0596088578825995, 9.9043376069230007, 2.0975104813987118)\n>>> 9.9043376069230007 + 2.0975104813987118\n12.001848088321712\n>>> y.min()\n12.001848089426717\n\n'
"\n0.0686702747648\n0.0164150896481\n0.128121386381\n[ 0.10370428  0.09921315  0.09676723  0.10457413  0.10201618  0.89964496]\n(array(0.0), array(1.4552599885729829), array(0.0), array(2.5072143354058221))\n(array(0.0), array(1.6666666666666667), array(0.0), array(6.0))\nrepr(start_params) array([ 0.10370428,  0.09921315,  0.09676723,  0.10457413,  0.10201618,\n        0.89964496,  6.39309417,  0.12812139])\nOptimization terminated successfully.\n         Current function value: -679.951339\n         Iterations: 398\n         Function evaluations: 609\n\nestimation results t-dist\n[ 0.10400826  0.10111893  0.09725133  0.10507788  0.10086163  0.8996041\n  4.72131318  0.09825355]\n[ 0.00365493  0.00356149  0.00349329  0.00362333  0.003732    0.00362716\n  0.72329352  0.00388832]\nrepr(start_params) array([ 0.10400826,  0.10111893,  0.09725133,  0.10507788,  0.10086163,\n        0.8996041 ,  4.72131318,  0.09825355])\nOptimization terminated successfully.\n         Current function value: -679.950443\n         Iterations 3\nusing Newton\n[ 0.10395383  0.10106762  0.09720665  0.10503384  0.10080599  0.89954546\n  4.70918964  0.09815885]\n[ 0.00365299  0.00355968  0.00349147  0.00362166  0.00373015  0.00362533\n  0.7201488   0.00388437]\n()\n[ 0.09992709  0.09786601  0.09387356  0.10229919  0.09756623  0.85466272\n  4.60459182  0.09661986]\n[ 0.11308292  0.10828401  0.1028508   0.11268895  0.10934726  0.94462721\n  7.15412655  0.13452746]\nrepr(start_params) array([ 1.,  9.,  2.])\nOptimization terminated successfully.\n         Current function value: 2636.129089\n         Iterations: 147\n         Function evaluations: 279\nOptimization terminated successfully.\n         Current function value: 0.016555\n         Iterations: 16\n         Function evaluations: 35\n[  0.84856418  10.2197801    1.78206799]\n(1.0596088578825995, 9.9043376069230007, 2.0975104813987118)\n12.0018480891 12.0018480883 12.0018480894\nrepr(start_params) array([ 1.,  2.])\nWarning: Desired error not necessarily achieveddue to precision loss\n         Current function value: 2643.549907\n         Iterations: 2\n         Function evaluations: 13\n         Gradient evaluations: 12\n>>> res_parks2 = mod_par.fit_ks()\nrepr(start_params) array([ 1.,  2.])\nOptimization terminated successfully.\n         Current function value: 2642.465273\n         Iterations: 92\n         Function evaluations: 172\nrepr(start_params) array([ 1.,  2.])\nOptimization terminated successfully.\n         Current function value: 2636.639863\n         Iterations: 73\n         Function evaluations: 136\nrepr(start_params) array([ 1.,  2.])\nOptimization terminated successfully.\n         Current function value: 2631.568778\n         Iterations: 75\n         Function evaluations: 133\nrepr(start_params) array([ 1.,  2.])\nOptimization terminated successfully.\n         Current function value: 2627.821044\n         Iterations: 75\n         Function evaluations: 135\nrepr(start_params) array([ 1.,  2.])\nWarning: Maximum number of function evaluations has been exceeded.\nrepr(start_params) array([ 1.,  2.])\nOptimization terminated successfully.\n         Current function value: 2631.568778\n         Iterations: 75\n         Function evaluations: 133\nrepr(start_params) array([ 1.,  2.])\nOptimization terminated successfully.\n         Current function value: 2626.431596\n         Iterations: 58\n         Function evaluations: 109\nrepr(start_params) array([ 1.,  2.])\nWarning: Maximum number of function evaluations has been exceeded.\nrepr(start_params) array([ 1.,  2.])\nWarning: Maximum number of function evaluations has been exceeded.\nrepr(start_params) array([ 1.,  2.])\nOptimization terminated successfully.\n         Current function value: 2626.737426\n         Iterations: 60\n         Function evaluations: 109\nrepr(start_params) array([ 1.,  2.])\nOptimization terminated successfully.\n         Current function value: 2627.821044\n         Iterations: 75\n         Function evaluations: 135\nrepr(start_params) array([ 1.,  2.])\nOptimization terminated successfully.\n         Current function value: 2626.471666\n         Iterations: 48\n         Function evaluations: 94\nrepr(start_params) array([ 1.,  2.])\nOptimization terminated successfully.\n         Current function value: 2627.196314\n         Iterations: 66\n         Function evaluations: 119\nrepr(start_params) array([ 1.,  2.])\nOptimization terminated successfully.\n         Current function value: 2626.578538\n         Iterations: 56\n         Function evaluations: 103\nrepr(start_params) array([ 1.,  2.])\nOptimization terminated successfully.\n         Current function value: 2626.471666\n         Iterations: 48\n         Function evaluations: 94\nrepr(start_params) array([ 1.,  2.])\nOptimization terminated successfully.\n         Current function value: 2626.651702\n         Iterations: 67\n         Function evaluations: 122\nrepr(start_params) array([ 1.,  2.])\nOptimization terminated successfully.\n         Current function value: 2626.737426\n         Iterations: 60\n         Function evaluations: 109\nrepr(start_params) array([ 1.,  2.])\nOptimization terminated successfully.\n         Current function value: 2626.613505\n         Iterations: 73\n         Function evaluations: 141\nrepr(start_params) array([ 1.,  2.])\nOptimization terminated successfully.\n         Current function value: 2626.578538\n         Iterations: 56\n         Function evaluations: 103\nrepr(start_params) array([ 1.,  2.])\nOptimization terminated successfully.\n         Current function value: 2626.632218\n         Iterations: 64\n         Function evaluations: 119\nrepr(start_params) array([ 1.,  2.])\nOptimization terminated successfully.\n         Current function value: 2626.651702\n         Iterations: 67\n         Function evaluations: 122\nrepr(start_params) array([ 1.,  2.])\nOptimization terminated successfully.\n         Current function value: 2626.622789\n         Iterations: 63\n         Function evaluations: 114\nrepr(start_params) array([ 1.,  2.])\nOptimization terminated successfully.\n         Current function value: 2626.613505\n         Iterations: 73\n         Function evaluations: 141\nrepr(start_params) array([ 1.,  2.])\nOptimization terminated successfully.\n         Current function value: 2626.627465\n         Iterations: 59\n         Function evaluations: 109\nrepr(start_params) array([ 1.,  2.])\nOptimization terminated successfully.\n         Current function value: 2626.632218\n         Iterations: 64\n         Function evaluations: 119\nrepr(start_params) array([ 1.,  2.])\nOptimization terminated successfully.\n         Current function value: 2626.625104\n         Iterations: 59\n         Function evaluations: 108\nrepr(start_params) array([ 1.,  2.])\nOptimization terminated successfully.\n         Current function value: 2626.629829\n         Iterations: 66\n         Function evaluations: 118\nrepr(start_params) array([ 1.,  2.])\nOptimization terminated successfully.\n         Current function value: 2626.632218\n         Iterations: 64\n         Function evaluations: 119\nrepr(start_params) array([ 1.,  2.])\nOptimization terminated successfully.\n         Current function value: 2626.632218\n         Iterations: 64\n         Function evaluations: 119\nrepr(start_params) array([ 1.,  2.])\nOptimization terminated successfully.\n         Current function value: 2626.628642\n         Iterations: 67\n         Function evaluations: 122\nrepr(start_params) array([ 1.,  2.])\nOptimization terminated successfully.\n         Current function value: 2626.631023\n         Iterations: 68\n         Function evaluations: 129\nrepr(start_params) array([ 1.,  2.])\nOptimization terminated successfully.\n         Current function value: 2626.630430\n         Iterations: 57\n         Function evaluations: 108\nrepr(start_params) array([ 1.,  2.])\nOptimization terminated successfully.\n         Current function value: 2626.629598\n         Iterations: 60\n         Function evaluations: 112\nrepr(start_params) array([ 1.,  2.])\nOptimization terminated successfully.\n         Current function value: 2626.630430\n         Iterations: 57\n         Function evaluations: 108\nrepr(start_params) array([ 1.,  2.])\nOptimization terminated successfully.\n         Current function value: 2626.630130\n         Iterations: 65\n         Function evaluations: 122\nrepr(start_params) array([ 1.,  2.])\nOptimization terminated successfully.\n         Current function value: 2626.629536\n         Iterations: 62\n         Function evaluations: 111\nrepr(start_params) array([ 1.,  2.])\nOptimization terminated successfully.\n         Current function value: 2626.630130\n         Iterations: 65\n         Function evaluations: 122\nrepr(start_params) array([ 1.,  2.])\nOptimization terminated successfully.\n         Current function value: 2626.629984\n         Iterations: 67\n         Function evaluations: 123\nOptimization terminated successfully.\n         Current function value: 0.016560\n         Iterations: 18\n         Function evaluations: 38\n>>> res_parks2\n(1.0592352626264809, 9.9051580457572399, 2.0966900385041591)\n>>> res_parks\n(1.0596088578825995, 9.9043376069230007, 2.0975104813987118)\n>>> res_par.params\narray([  0.84856418,  10.2197801 ,   1.78206799])\n>>> np.sqrt(np.diag(mod_par.hessian(res_par.params)))\narray([ NaN,  NaN,  NaN])\n>>> mod_par.hessian(res_par.params\n... )\narray([[ NaN,  NaN,  NaN],\n       [ NaN,  NaN,  NaN],\n       [ NaN,  NaN,  NaN]])\n>>> mod_par.hessian(res_parks)\narray([[ NaN,  NaN,  NaN],\n       [ NaN,  NaN,  NaN],\n       [ NaN,  NaN,  NaN]])\n\n>>> mod_par.hessian(np.array(res_parks))\narray([[ NaN,  NaN,  NaN],\n       [ NaN,  NaN,  NaN],\n       [ NaN,  NaN,  NaN]])\n>>> mod_par.fixed_params\narray([        NaN,  9.90510677,         NaN])\n>>> mod_par.fixed_params=None\n>>> mod_par.hessian(np.array(res_parks))\narray([[-890.48553491,           NaN,           NaN],\n       [          NaN,           NaN,           NaN],\n       [          NaN,           NaN,           NaN]])\n>>> mod_par.loglike(np.array(res_parks))\n-2626.6322080820569\n>>> mod_par.bsejac\nTraceback (most recent call last):\n  [...]\nAttributeError: 'MyPareto' object has no attribute 'bsejac'\n\n>>> hasattr(mod_par, 'start_params')\nTrue\n>>> mod_par.start_params\narray([ 1.,  2.])\n>>> stats.pareto.stats(1., 9., 2., moments='mvsk')\n(array(1.#INF), array(1.#INF), array(1.#QNAN), array(1.#QNAN))\n>>> stats.pareto.stats(1., 8., 2., moments='mvsk')\n(array(1.#INF), array(1.#INF), array(1.#QNAN), array(1.#QNAN))\n>>> stats.pareto.stats(1., 8., 1., moments='mvsk')\n(array(1.#INF), array(1.#INF), array(1.#QNAN), array(1.#QNAN))\n>>> stats.pareto.stats(1., moments='mvsk')\n(array(1.#INF), array(1.#INF), array(1.#QNAN), array(1.#QNAN))\n>>> stats.pareto.stats(0.5, moments='mvsk')\n(array(1.#INF), array(1.#INF), array(1.#QNAN), array(1.#QNAN))\n>>> stats.pareto.stats(2, moments='mvsk')\n(array(2.0), array(1.#INF), array(1.#QNAN), array(1.#QNAN))\n>>> stats.pareto.stats(10, moments='mvsk')\n(array(1.1111111111111112), array(0.015432098765432098), array(2.8110568859997356), array(14.828571428571429))\n>>> stats.pareto.rvs(10, size=10)\narray([ 1.07716265,  1.18977526,  1.07093   ,  1.05157081,  1.15991232,\n        1.31015589,  1.06675107,  1.08082475,  1.19501243,  1.34967158])\n>>> r = stats.pareto.rvs(10, size=1000)\n\n>>> import matplotlib.pyplot as plt\n>>> plt.hist(r)\n(array([962,  32,   3,   2,   0,   0,   0,   0,   0,   1]), array([ 1.00013046,  1.3968991 ,  1.79366773,  2.19043637,  2.587205  ,\n        2.98397364,  3.38074227,  3.77751091,  4.17427955,  4.57104818,\n        4.96781682]), <a list of 10 Patch objects>)\n>>> plt.show()\n\n"