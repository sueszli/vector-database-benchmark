"""
Author: Josef Perktold
License: BSD-3

"""
from __future__ import print_function
import numpy as np
from scipy import stats

class NonlinearDeltaCov:
    """Asymptotic covariance by Deltamethod

    The function is designed for 2d array, with rows equal to
    the number of equations or constraints and columns equal to the number
    of parameters. 1d params work by chance ?

    fun: R^{m*k) -> R^{m}  where m is number of equations and k is
    the number of parameters.

    equations follow Greene

    This class does not use any caching. The intended usage is as a helper
    function. Extra methods have been added for convenience but might move
    to calling functions.

    The naming in this class uses params for the original random variable, and
    cov_params for it's covariance matrix. However, this class is independent
    of the use cases in support of the models.

    Parameters
    ----------
    func : callable, f(params)
        Nonlinear function of the estimation parameters. The return of
        the function can be vector valued, i.e. a 1-D array.
    params : ndarray
        Parameters at which function `func` is evaluated.
    cov_params : ndarray
        Covariance matrix of the parameters `params`.
    deriv : function or None
        First derivative or Jacobian of func. If deriv is None, then a
        numerical derivative will be used. If func returns a 1-D array,
        then the `deriv` should have rows corresponding to the elements
        of the return of func.
    func_args : None
        Not yet implemented.


    """

    def __init__(self, func, params, cov_params, deriv=None, func_args=None):
        if False:
            return 10
        self.fun = func
        self.params = params
        self.cov_params = cov_params
        self._grad = deriv
        self.func_args = func_args if func_args is not None else ()
        if func_args is not None:
            raise NotImplementedError('func_args not yet implemented')

    def grad(self, params=None, **kwds):
        if False:
            for i in range(10):
                print('nop')
        'First derivative, jacobian of func evaluated at params.\n\n        Parameters\n        ----------\n        params : None or ndarray\n            Values at which gradient is evaluated. If params is None, then\n            the attached params are used.\n            TODO: should we drop this\n        kwds : keyword arguments\n            This keyword arguments are used without changes in the calulation\n            of numerical derivatives. These are only used if a `deriv` function\n            was not provided.\n\n        Returns\n        -------\n        grad : ndarray\n            gradient or jacobian of the function\n        '
        if params is None:
            params = self.params
        if self._grad is not None:
            return self._grad(params)
        else:
            try:
                from statsmodels.tools.numdiff import approx_fprime_cs
                jac = approx_fprime_cs(params, self.fun, **kwds)
            except TypeError:
                from statsmodels.tools.numdiff import approx_fprime
                jac = approx_fprime(params, self.fun, **kwds)
            return jac

    def cov(self):
        if False:
            for i in range(10):
                print('nop')
        'Covariance matrix of the transformed random variable.\n        '
        g = self.grad()
        covar = np.dot(np.dot(g, self.cov_params), g.T)
        return covar

    def predicted(self):
        if False:
            for i in range(10):
                print('nop')
        'Value of the function evaluated at the attached params.\n\n        Note: This is not equal to the expected value if the transformation is\n        nonlinear. If params is the maximum likelihood estimate, then\n        `predicted` is the maximum likelihood estimate of the value of the\n        nonlinear function.\n        '
        predicted = self.fun(self.params)
        if predicted.ndim > 1:
            predicted = predicted.squeeze()
        return predicted

    def wald_test(self, value):
        if False:
            for i in range(10):
                print('nop')
        'Joint hypothesis tests that H0: f(params) = value.\n\n        The alternative hypothesis is two-sided H1: f(params) != value.\n\n        Warning: this might be replaced with more general version that returns\n        ContrastResults.\n        currently uses chisquare distribution, use_f option not yet implemented\n\n        Parameters\n        ----------\n        value : float or ndarray\n            value of f(params) under the Null Hypothesis\n\n        Returns\n        -------\n        statistic : float\n            Value of the test statistic.\n        pvalue : float\n            The p-value for the hypothesis test, based and chisquare\n            distribution and implies a two-sided hypothesis test\n        '
        m = self.predicted()
        v = self.cov()
        df_constraints = np.size(m)
        diff = m - value
        lmstat = np.dot(np.dot(diff.T, np.linalg.inv(v)), diff)
        return (lmstat, stats.chi2.sf(lmstat, df_constraints))

    def var(self):
        if False:
            return 10
        'standard error for each equation (row) treated separately\n\n        '
        g = self.grad()
        var = (np.dot(g, self.cov_params) * g).sum(-1)
        if var.ndim == 2:
            var = var.T
        return var

    def se_vectorized(self):
        if False:
            while True:
                i = 10
        'standard error for each equation (row) treated separately\n\n        '
        var = self.var()
        return np.sqrt(var)

    def conf_int(self, alpha=0.05, use_t=False, df=None, var_extra=None, predicted=None, se=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Confidence interval for predicted based on delta method.\n\n        Parameters\n        ----------\n        alpha : float, optional\n            The significance level for the confidence interval.\n            ie., The default `alpha` = .05 returns a 95% confidence interval.\n        use_t : boolean\n            If use_t is False (default), then the normal distribution is used\n            for the confidence interval, otherwise the t distribution with\n            `df` degrees of freedom is used.\n        df : int or float\n            degrees of freedom for t distribution. Only used and required if\n            use_t is True.\n        var_extra : None or array_like float\n            Additional variance that is added to the variance based on the\n            delta method. This can be used to obtain confidence intervalls for\n            new observations (prediction interval).\n        predicted : ndarray (float)\n            Predicted value, can be used to avoid repeated calculations if it\n            is already available.\n        se : ndarray (float)\n            Standard error, can be used to avoid repeated calculations if it\n            is already available.\n\n        Returns\n        -------\n        conf_int : array\n            Each row contains [lower, upper] limits of the confidence interval\n            for the corresponding parameter. The first column contains all\n            lower, the second column contains all upper limits.\n        '
        if not use_t:
            dist = stats.norm
            dist_args = ()
        else:
            if df is None:
                raise ValueError('t distribution requires df')
            dist = stats.t
            dist_args = (df,)
        if predicted is None:
            predicted = self.predicted()
        if se is None:
            se = self.se_vectorized()
        if var_extra is not None:
            se = np.sqrt(se ** 2 + var_extra)
        q = dist.ppf(1 - alpha / 2.0, *dist_args)
        lower = predicted - q * se
        upper = predicted + q * se
        ci = np.column_stack((lower, upper))
        if ci.shape[1] != 2:
            raise RuntimeError('something wrong: ci not 2 columns')
        return ci

    def summary(self, xname=None, alpha=0.05, title=None, use_t=False, df=None):
        if False:
            i = 10
            return i + 15
        'Summarize the Results of the nonlinear transformation.\n\n        This provides a parameter table equivalent to `t_test` and reuses\n        `ContrastResults`.\n\n        Parameters\n        -----------\n        xname : list of strings, optional\n            Default is `c_##` for ## in p the number of regressors\n        alpha : float\n            Significance level for the confidence intervals. Default is\n            alpha = 0.05 which implies a confidence level of 95%.\n        title : string, optional\n            Title for the params table. If not None, then this replaces the\n            default title\n        use_t : boolean\n            If use_t is False (default), then the normal distribution is used\n            for the confidence interval, otherwise the t distribution with\n            `df` degrees of freedom is used.\n        df : int or float\n            degrees of freedom for t distribution. Only used and required if\n            use_t is True.\n\n        Returns\n        -------\n        smry : string or Summary instance\n            This contains a parameter results table in the case of t or z test\n            in the same form as the parameter results table in the model\n            results summary.\n            For F or Wald test, the return is a string.\n        '
        from statsmodels.stats.contrast import ContrastResults
        predicted = self.predicted()
        se = self.se_vectorized()
        predicted = np.atleast_1d(predicted)
        if predicted.ndim > 1:
            predicted = predicted.squeeze()
        se = np.atleast_1d(se)
        statistic = predicted / se
        if use_t:
            df_resid = df
            cr = ContrastResults(effect=predicted, t=statistic, sd=se, df_denom=df_resid)
        else:
            cr = ContrastResults(effect=predicted, statistic=statistic, sd=se, df_denom=None, distribution='norm')
        return cr.summary(xname=xname, alpha=alpha, title=title)