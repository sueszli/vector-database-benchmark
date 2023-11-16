"""Non-linear least squares



Author: Josef Perktold based on scipy.optimize.curve_fit

"""
import numpy as np
from scipy import optimize
from statsmodels.base.model import Model

class Results:
    """just a dummy placeholder for now
    most results from RegressionResults can be used here
    """
    pass

class NonlinearLS(Model):
    """Base class for estimation of a non-linear model with least squares

    This class is supposed to be subclassed, and the subclass has to provide a method
    `_predict` that defines the non-linear function `f(params) that is predicting the endogenous
    variable. The model is assumed to be

    :math: y = f(params) + error

    and the estimator minimizes the sum of squares of the estimated error.

    :math: min_parmas \\sum (y - f(params))**2

    f has to return the prediction for each observation. Exogenous or explanatory variables
    should be accessed as attributes of the class instance, and can be given as arguments
    when the instance is created.

    Warning:
    Weights are not correctly handled yet in the results statistics,
    but included when estimating the parameters.

    similar to scipy.optimize.curve_fit
    API difference: params are array_like not split up, need n_params information

    includes now weights similar to curve_fit
    no general sigma yet (OLS and WLS, but no GLS)

    This is currently holding on to intermediate results that are not necessary
    but useful for testing.

    Fit returns and instance of RegressionResult, in contrast to the linear
    model, results in this case are based on a local approximation, essentially
    y = f(X, params) is replaced by y = grad * params where grad is the Gradient
    or Jacobian with the shape (nobs, nparams). See for example Greene

    Examples
    --------

    class Myfunc(NonlinearLS):

        def _predict(self, params):
            x = self.exog
            a, b, c = params
            return a*np.exp(-b*x) + c

    Ff we have data (y, x), we can create an instance and fit it with

    mymod = Myfunc(y, x)
    myres = mymod.fit(nparams=3)

    and use the non-linear regression results, for example

    myres.params
    myres.bse
    myres.tvalues


    """

    def __init__(self, endog=None, exog=None, weights=None, sigma=None, missing='none'):
        if False:
            i = 10
            return i + 15
        self.endog = endog
        self.exog = exog
        if sigma is not None:
            sigma = np.asarray(sigma)
            if sigma.ndim < 2:
                self.sigma = sigma
                self.weights = 1.0 / sigma
            else:
                raise ValueError('correlated errors are not handled yet')
        else:
            self.weights = None

    def predict(self, exog, params=None):
        if False:
            while True:
                i = 10
        return self._predict(params)

    def _predict(self, params):
        if False:
            for i in range(10):
                print('nop')
        pass

    def start_value(self):
        if False:
            return 10
        return None

    def geterrors(self, params, weights=None):
        if False:
            for i in range(10):
                print('nop')
        if weights is None:
            if self.weights is None:
                return self.endog - self._predict(params)
            else:
                weights = self.weights
        return weights * (self.endog - self._predict(params))

    def errorsumsquares(self, params):
        if False:
            while True:
                i = 10
        return (self.geterrors(params) ** 2).sum()

    def fit(self, start_value=None, nparams=None, **kw):
        if False:
            while True:
                i = 10
        if start_value is not None:
            p0 = start_value
        else:
            p0 = self.start_value()
            if p0 is not None:
                pass
            elif nparams is not None:
                p0 = 0.1 * np.ones(nparams)
            else:
                raise ValueError('need information about start values for' + 'optimization')
        func = self.geterrors
        res = optimize.leastsq(func, p0, full_output=1, **kw)
        (popt, pcov, infodict, errmsg, ier) = res
        if ier not in [1, 2, 3, 4]:
            msg = 'Optimal parameters not found: ' + errmsg
            raise RuntimeError(msg)
        err = infodict['fvec']
        ydata = self.endog
        if len(ydata) > len(p0) and pcov is not None:
            s_sq = (err ** 2).sum() / (len(ydata) - len(p0))
            pcov = pcov * s_sq
        else:
            pcov = None
        self.df_resid = len(ydata) - len(p0)
        self.df_model = len(p0)
        fitres = Results()
        fitres.params = popt
        fitres.pcov = pcov
        fitres.rawres = res
        self.wendog = self.endog
        self.wexog = self.jac_predict(popt)
        pinv_wexog = np.linalg.pinv(self.wexog)
        self.normalized_cov_params = np.dot(pinv_wexog, np.transpose(pinv_wexog))
        from statsmodels.regression import RegressionResults
        results = RegressionResults
        beta = popt
        lfit = RegressionResults(self, beta, normalized_cov_params=self.normalized_cov_params)
        lfit.fitres = fitres
        self._results = lfit
        return lfit

    def fit_minimal(self, start_value, **kwargs):
        if False:
            while True:
                i = 10
        'minimal fitting with no extra calculations'
        func = self.geterrors
        res = optimize.leastsq(func, start_value, full_output=0, **kwargs)
        return res

    def fit_random(self, ntries=10, rvs_generator=None, nparams=None):
        if False:
            for i in range(10):
                print('nop')
        'fit with random starting values\n\n        this could be replaced with a global fitter\n\n        '
        if nparams is None:
            nparams = self.nparams
        if rvs_generator is None:
            rvs = np.random.uniform(low=-10, high=10, size=(ntries, nparams))
        else:
            rvs = rvs_generator(size=(ntries, nparams))
        results = np.array([np.r_[self.fit_minimal(rv), rv] for rv in rvs])
        return results

    def jac_predict(self, params):
        if False:
            return 10
        'jacobian of prediction function using complex step derivative\n\n        This assumes that the predict function does not use complex variable\n        but is designed to do so.\n\n        '
        from statsmodels.tools.numdiff import approx_fprime_cs
        jaccs_err = approx_fprime_cs(params, self._predict)
        return jaccs_err

class Myfunc(NonlinearLS):

    def _predict(self, params):
        if False:
            while True:
                i = 10
        x = self.exog
        (a, b, c) = params
        return a * np.exp(-b * x) + c
if __name__ == '__main__':

    def func0(x, a, b, c):
        if False:
            print('Hello World!')
        return a * np.exp(-b * x) + c

    def func(params, x):
        if False:
            i = 10
            return i + 15
        (a, b, c) = params
        return a * np.exp(-b * x) + c

    def error(params, x, y):
        if False:
            i = 10
            return i + 15
        return y - func(params, x)

    def error2(params, x, y):
        if False:
            while True:
                i = 10
        return (y - func(params, x)) ** 2
    x = np.linspace(0, 4, 50)
    params = np.array([2.5, 1.3, 0.5])
    y0 = func(params, x)
    y = y0 + 0.2 * np.random.normal(size=len(x))
    res = optimize.leastsq(error, params, args=(x, y), full_output=True)
    mod = Myfunc(y, x)
    resmy = mod.fit(nparams=3)
    (cf_params, cf_pcov) = optimize.curve_fit(func0, x, y)
    cf_bse = np.sqrt(np.diag(cf_pcov))
    print(res[0])
    print(cf_params)
    print(resmy.params)
    print(cf_bse)
    print(resmy.bse)