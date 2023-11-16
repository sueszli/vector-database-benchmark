"""
Markov switching regression models

Author: Chad Fulton
License: BSD-3
"""
import numpy as np
import statsmodels.base.wrapper as wrap
from statsmodels.tsa.regime_switching import markov_switching

class MarkovRegression(markov_switching.MarkovSwitching):
    """
    First-order k-regime Markov switching regression model

    Parameters
    ----------
    endog : array_like
        The endogenous variable.
    k_regimes : int
        The number of regimes.
    trend : {'n', 'c', 't', 'ct'}
        Whether or not to include a trend. To include an intercept, time trend,
        or both, set `trend='c'`, `trend='t'`, or `trend='ct'`. For no trend,
        set `trend='n'`. Default is an intercept.
    exog : array_like, optional
        Array of exogenous regressors, shaped nobs x k.
    order : int, optional
        The order of the model describes the dependence of the likelihood on
        previous regimes. This depends on the model in question and should be
        set appropriately by subclasses.
    exog_tvtp : array_like, optional
        Array of exogenous or lagged variables to use in calculating
        time-varying transition probabilities (TVTP). TVTP is only used if this
        variable is provided. If an intercept is desired, a column of ones must
        be explicitly included in this array.
    switching_trend : bool or iterable, optional
        If a boolean, sets whether or not all trend coefficients are
        switching across regimes. If an iterable, should be of length equal
        to the number of trend variables, where each element is
        a boolean describing whether the corresponding coefficient is
        switching. Default is True.
    switching_exog : bool or iterable, optional
        If a boolean, sets whether or not all regression coefficients are
        switching across regimes. If an iterable, should be of length equal
        to the number of exogenous variables, where each element is
        a boolean describing whether the corresponding coefficient is
        switching. Default is True.
    switching_variance : bool, optional
        Whether or not there is regime-specific heteroskedasticity, i.e.
        whether or not the error term has a switching variance. Default is
        False.

    Notes
    -----
    This model is new and API stability is not guaranteed, although changes
    will be made in a backwards compatible way if possible.

    The model can be written as:

    .. math::

        y_t = a_{S_t} + x_t' \\beta_{S_t} + \\varepsilon_t \\\\
        \\varepsilon_t \\sim N(0, \\sigma_{S_t}^2)

    i.e. the model is a dynamic linear regression where the coefficients and
    the variance of the error term may be switching across regimes.

    The `trend` is accommodated by prepending columns to the `exog` array. Thus
    if `trend='c'`, the passed `exog` array should not already have a column of
    ones.

    See the notebook `Markov switching dynamic regression
    <../examples/notebooks/generated/markov_regression.html>`__ for an
    overview.

    References
    ----------
    Kim, Chang-Jin, and Charles R. Nelson. 1999.
    "State-Space Models with Regime Switching:
    Classical and Gibbs-Sampling Approaches with Applications".
    MIT Press Books. The MIT Press.
    """

    def __init__(self, endog, k_regimes, trend='c', exog=None, order=0, exog_tvtp=None, switching_trend=True, switching_exog=True, switching_variance=False, dates=None, freq=None, missing='none'):
        if False:
            print('Hello World!')
        from statsmodels.tools.validation import string_like
        self.trend = string_like(trend, 'trend', options=('n', 'c', 'ct', 't'))
        self.switching_trend = switching_trend
        self.switching_exog = switching_exog
        self.switching_variance = switching_variance
        (self.k_exog, exog) = markov_switching.prepare_exog(exog)
        nobs = len(endog)
        self.k_trend = 0
        self._k_exog = self.k_exog
        trend_exog = None
        if trend == 'c':
            trend_exog = np.ones((nobs, 1))
            self.k_trend = 1
        elif trend == 't':
            trend_exog = (np.arange(nobs) + 1)[:, np.newaxis]
            self.k_trend = 1
        elif trend == 'ct':
            trend_exog = np.c_[np.ones((nobs, 1)), (np.arange(nobs) + 1)[:, np.newaxis]]
            self.k_trend = 2
        if trend_exog is not None:
            exog = trend_exog if exog is None else np.c_[trend_exog, exog]
            self._k_exog += self.k_trend
        super(MarkovRegression, self).__init__(endog, k_regimes, order=order, exog_tvtp=exog_tvtp, exog=exog, dates=dates, freq=freq, missing=missing)
        if self.switching_trend is True or self.switching_trend is False:
            self.switching_trend = [self.switching_trend] * self.k_trend
        elif not len(self.switching_trend) == self.k_trend:
            raise ValueError('Invalid iterable passed to `switching_trend`.')
        if self.switching_exog is True or self.switching_exog is False:
            self.switching_exog = [self.switching_exog] * self.k_exog
        elif not len(self.switching_exog) == self.k_exog:
            raise ValueError('Invalid iterable passed to `switching_exog`.')
        self.switching_coeffs = np.r_[self.switching_trend, self.switching_exog].astype(bool).tolist()
        self.parameters['exog'] = self.switching_coeffs
        self.parameters['variance'] = [1] if self.switching_variance else [0]

    def predict_conditional(self, params):
        if False:
            while True:
                i = 10
        '\n        In-sample prediction, conditional on the current regime\n\n        Parameters\n        ----------\n        params : array_like\n            Array of parameters at which to perform prediction.\n\n        Returns\n        -------\n        predict : array_like\n            Array of predictions conditional on current, and possibly past,\n            regimes\n        '
        params = np.array(params, ndmin=1)
        predict = np.zeros((self.k_regimes, self.nobs), dtype=params.dtype)
        for i in range(self.k_regimes):
            if self._k_exog > 0:
                coeffs = params[self.parameters[i, 'exog']]
                predict[i] = np.dot(self.exog, coeffs)
        return predict[:, None, :]

    def _resid(self, params):
        if False:
            i = 10
            return i + 15
        predict = np.repeat(self.predict_conditional(params), self.k_regimes, axis=1)
        return self.endog - predict

    def _conditional_loglikelihoods(self, params):
        if False:
            return 10
        "\n        Compute loglikelihoods conditional on the current period's regime\n        "
        resid = self._resid(params)
        variance = params[self.parameters['variance']].squeeze()
        if self.switching_variance:
            variance = np.reshape(variance, (self.k_regimes, 1, 1))
        conditional_loglikelihoods = -0.5 * resid ** 2 / variance - 0.5 * np.log(2 * np.pi * variance)
        return conditional_loglikelihoods

    @property
    def _res_classes(self):
        if False:
            while True:
                i = 10
        return {'fit': (MarkovRegressionResults, MarkovRegressionResultsWrapper)}

    def _em_iteration(self, params0):
        if False:
            print('Hello World!')
        '\n        EM iteration\n\n        Notes\n        -----\n        This uses the inherited _em_iteration method for computing the\n        non-TVTP transition probabilities and then performs the EM step for\n        regression coefficients and variances.\n        '
        (result, params1) = super(MarkovRegression, self)._em_iteration(params0)
        tmp = np.sqrt(result.smoothed_marginal_probabilities)
        coeffs = None
        if self._k_exog > 0:
            coeffs = self._em_exog(result, self.endog, self.exog, self.parameters.switching['exog'], tmp)
            for i in range(self.k_regimes):
                params1[self.parameters[i, 'exog']] = coeffs[i]
        params1[self.parameters['variance']] = self._em_variance(result, self.endog, self.exog, coeffs, tmp)
        return (result, params1)

    def _em_exog(self, result, endog, exog, switching, tmp=None):
        if False:
            print('Hello World!')
        '\n        EM step for regression coefficients\n        '
        k_exog = exog.shape[1]
        coeffs = np.zeros((self.k_regimes, k_exog))
        if not np.all(switching):
            nonswitching_exog = exog[:, ~switching]
            nonswitching_coeffs = np.dot(np.linalg.pinv(nonswitching_exog), endog)
            coeffs[:, ~switching] = nonswitching_coeffs
            endog = endog - np.dot(nonswitching_exog, nonswitching_coeffs)
        if np.any(switching):
            switching_exog = exog[:, switching]
            if tmp is None:
                tmp = np.sqrt(result.smoothed_marginal_probabilities)
            for i in range(self.k_regimes):
                tmp_endog = tmp[i] * endog
                tmp_exog = tmp[i][:, np.newaxis] * switching_exog
                coeffs[i, switching] = np.dot(np.linalg.pinv(tmp_exog), tmp_endog)
        return coeffs

    def _em_variance(self, result, endog, exog, betas, tmp=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        EM step for variances\n        '
        k_exog = 0 if exog is None else exog.shape[1]
        if self.switching_variance:
            variance = np.zeros(self.k_regimes)
            for i in range(self.k_regimes):
                if k_exog > 0:
                    resid = endog - np.dot(exog, betas[i])
                else:
                    resid = endog
                variance[i] = np.sum(resid ** 2 * result.smoothed_marginal_probabilities[i]) / np.sum(result.smoothed_marginal_probabilities[i])
        else:
            variance = 0
            if tmp is None:
                tmp = np.sqrt(result.smoothed_marginal_probabilities)
            for i in range(self.k_regimes):
                tmp_endog = tmp[i] * endog
                if k_exog > 0:
                    tmp_exog = tmp[i][:, np.newaxis] * exog
                    resid = tmp_endog - np.dot(tmp_exog, betas[i])
                else:
                    resid = tmp_endog
                variance += np.sum(resid ** 2)
            variance /= self.nobs
        return variance

    @property
    def start_params(self):
        if False:
            while True:
                i = 10
        '\n        (array) Starting parameters for maximum likelihood estimation.\n\n        Notes\n        -----\n        These are not very sophisticated and / or good. We set equal transition\n        probabilities and interpolate regression coefficients between zero and\n        the OLS estimates, where the interpolation is based on the regime\n        number. We rely heavily on the EM algorithm to quickly find much better\n        starting parameters, which are then used by the typical scoring\n        approach.\n        '
        params = markov_switching.MarkovSwitching.start_params.fget(self)
        if self._k_exog > 0:
            beta = np.dot(np.linalg.pinv(self.exog), self.endog)
            variance = np.var(self.endog - np.dot(self.exog, beta))
            if np.any(self.switching_coeffs):
                for i in range(self.k_regimes):
                    params[self.parameters[i, 'exog']] = beta * (i / self.k_regimes)
            else:
                params[self.parameters['exog']] = beta
        else:
            variance = np.var(self.endog)
        if self.switching_variance:
            params[self.parameters['variance']] = np.linspace(variance / 10.0, variance, num=self.k_regimes)
        else:
            params[self.parameters['variance']] = variance
        return params

    @property
    def param_names(self):
        if False:
            i = 10
            return i + 15
        '\n        (list of str) List of human readable parameter names (for parameters\n        actually included in the model).\n        '
        param_names = np.array(markov_switching.MarkovSwitching.param_names.fget(self), dtype=object)
        if np.any(self.switching_coeffs):
            for i in range(self.k_regimes):
                param_names[self.parameters[i, 'exog']] = ['%s[%d]' % (exog_name, i) for exog_name in self.exog_names]
        else:
            param_names[self.parameters['exog']] = self.exog_names
        if self.switching_variance:
            for i in range(self.k_regimes):
                param_names[self.parameters[i, 'variance']] = 'sigma2[%d]' % i
        else:
            param_names[self.parameters['variance']] = 'sigma2'
        return param_names.tolist()

    def transform_params(self, unconstrained):
        if False:
            i = 10
            return i + 15
        '\n        Transform unconstrained parameters used by the optimizer to constrained\n        parameters used in likelihood evaluation\n\n        Parameters\n        ----------\n        unconstrained : array_like\n            Array of unconstrained parameters used by the optimizer, to be\n            transformed.\n\n        Returns\n        -------\n        constrained : array_like\n            Array of constrained parameters which may be used in likelihood\n            evaluation.\n        '
        constrained = super(MarkovRegression, self).transform_params(unconstrained)
        constrained[self.parameters['exog']] = unconstrained[self.parameters['exog']]
        constrained[self.parameters['variance']] = unconstrained[self.parameters['variance']] ** 2
        return constrained

    def untransform_params(self, constrained):
        if False:
            i = 10
            return i + 15
        '\n        Transform constrained parameters used in likelihood evaluation\n        to unconstrained parameters used by the optimizer\n\n        Parameters\n        ----------\n        constrained : array_like\n            Array of constrained parameters used in likelihood evaluation, to\n            be transformed.\n\n        Returns\n        -------\n        unconstrained : array_like\n            Array of unconstrained parameters used by the optimizer.\n        '
        unconstrained = super(MarkovRegression, self).untransform_params(constrained)
        unconstrained[self.parameters['exog']] = constrained[self.parameters['exog']]
        unconstrained[self.parameters['variance']] = constrained[self.parameters['variance']] ** 0.5
        return unconstrained

class MarkovRegressionResults(markov_switching.MarkovSwitchingResults):
    """
    Class to hold results from fitting a Markov switching regression model

    Parameters
    ----------
    model : MarkovRegression instance
        The fitted model instance
    params : ndarray
        Fitted parameters
    filter_results : HamiltonFilterResults or KimSmootherResults instance
        The underlying filter and, optionally, smoother output
    cov_type : str
        The type of covariance matrix estimator to use. Can be one of 'approx',
        'opg', 'robust', or 'none'.

    Attributes
    ----------
    model : Model instance
        A reference to the model that was fit.
    filter_results : HamiltonFilterResults or KimSmootherResults instance
        The underlying filter and, optionally, smoother output
    nobs : float
        The number of observations used to fit the model.
    params : ndarray
        The parameters of the model.
    scale : float
        This is currently set to 1.0 and not used by the model or its results.
    """
    pass

class MarkovRegressionResultsWrapper(markov_switching.MarkovSwitchingResultsWrapper):
    pass
wrap.populate_wrapper(MarkovRegressionResultsWrapper, MarkovRegressionResults)