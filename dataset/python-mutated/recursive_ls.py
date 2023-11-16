"""
Recursive least squares model

Author: Chad Fulton
License: Simplified-BSD
"""
import numpy as np
import pandas as pd
from statsmodels.compat.pandas import Appender
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tsa.statespace.mlemodel import MLEModel, MLEResults, MLEResultsWrapper, PredictionResults, PredictionResultsWrapper
from statsmodels.tsa.statespace.tools import concat
from statsmodels.tools.tools import Bunch
from statsmodels.tools.decorators import cache_readonly
import statsmodels.base.wrapper as wrap
_cusum_squares_scalars = np.array([[1.072983, 1.2238734, 1.3581015, 1.5174271, 1.6276236], [-0.6698868, -0.6700069, -0.6701218, -0.6702672, -0.6703724], [-0.5816458, -0.7351697, -0.8858694, -1.0847745, -1.2365861]])

class RecursiveLS(MLEModel):
    """
    Recursive least squares

    Parameters
    ----------
    endog : array_like
        The observed time-series process :math:`y`
    exog : array_like
        Array of exogenous regressors, shaped nobs x k.
    constraints : array_like, str, or tuple
            - array : An r x k array where r is the number of restrictions to
              test and k is the number of regressors. It is assumed that the
              linear combination is equal to zero.
            - str : The full hypotheses to test can be given as a string.
              See the examples.
            - tuple : A tuple of arrays in the form (R, q), ``q`` can be
              either a scalar or a length p row vector.

    Notes
    -----
    Recursive least squares (RLS) corresponds to expanding window ordinary
    least squares (OLS).

    This model applies the Kalman filter to compute recursive estimates of the
    coefficients and recursive residuals.

    References
    ----------
    .. [*] Durbin, James, and Siem Jan Koopman. 2012.
       Time Series Analysis by State Space Methods: Second Edition.
       Oxford University Press.
    """

    def __init__(self, endog, exog, constraints=None, **kwargs):
        if False:
            while True:
                i = 10
        endog_using_pandas = _is_using_pandas(endog, None)
        if not endog_using_pandas:
            endog = np.asanyarray(endog)
        exog_is_using_pandas = _is_using_pandas(exog, None)
        if not exog_is_using_pandas:
            exog = np.asarray(exog)
        if exog.ndim == 1:
            if not exog_is_using_pandas:
                exog = exog[:, None]
            else:
                exog = pd.DataFrame(exog)
        self.k_exog = exog.shape[1]
        self.k_constraints = 0
        self._r_matrix = self._q_matrix = None
        if constraints is not None:
            from patsy import DesignInfo
            from statsmodels.base.data import handle_data
            data = handle_data(endog, exog, **kwargs)
            names = data.param_names
            LC = DesignInfo(names).linear_constraint(constraints)
            (self._r_matrix, self._q_matrix) = (LC.coefs, LC.constants)
            self.k_constraints = self._r_matrix.shape[0]
            nobs = len(endog)
            constraint_endog = np.zeros((nobs, len(self._r_matrix)))
            if endog_using_pandas:
                constraint_endog = pd.DataFrame(constraint_endog, index=endog.index)
                endog = concat([endog, constraint_endog], axis=1)
                endog.iloc[:, 1:] = np.tile(self._q_matrix.T, (nobs, 1))
            else:
                endog[:, 1:] = self._q_matrix[:, 0]
        kwargs.setdefault('initialization', 'diffuse')
        formula_kwargs = ['missing', 'missing_idx', 'formula', 'design_info']
        for name in formula_kwargs:
            if name in kwargs:
                del kwargs[name]
        super(RecursiveLS, self).__init__(endog, k_states=self.k_exog, exog=exog, **kwargs)
        self.ssm.filter_univariate = True
        self.ssm.filter_concentrated = True
        self['design'] = np.zeros((self.k_endog, self.k_states, self.nobs))
        self['design', 0] = self.exog[:, :, None].T
        if self._r_matrix is not None:
            self['design', 1:, :] = self._r_matrix[:, :, None]
        self['transition'] = np.eye(self.k_states)
        self['obs_cov', 0, 0] = 1.0
        self['transition'] = np.eye(self.k_states)
        if self._r_matrix is not None:
            self.k_endog = 1

    @classmethod
    def from_formula(cls, formula, data, subset=None, constraints=None):
        if False:
            while True:
                i = 10
        return super(MLEModel, cls).from_formula(formula, data, subset, constraints=constraints)

    def _validate_can_fix_params(self, param_names):
        if False:
            for i in range(10):
                print('nop')
        raise ValueError('Linear constraints on coefficients should be given using the `constraints` argument in constructing. the model. Other parameter constraints are not available in the resursive least squares model.')

    def fit(self):
        if False:
            return 10
        '\n        Fits the model by application of the Kalman filter\n\n        Returns\n        -------\n        RecursiveLSResults\n        '
        smoother_results = self.smooth(return_ssm=True)
        with self.ssm.fixed_scale(smoother_results.scale):
            res = self.smooth()
        return res

    def filter(self, return_ssm=False, **kwargs):
        if False:
            print('Hello World!')
        result = super(RecursiveLS, self).filter([], transformed=True, cov_type='none', return_ssm=True, **kwargs)
        if not return_ssm:
            params = result.filtered_state[:, -1]
            cov_kwds = {'custom_cov_type': 'nonrobust', 'custom_cov_params': result.filtered_state_cov[:, :, -1], 'custom_description': 'Parameters and covariance matrix estimates are RLS estimates conditional on the entire sample.'}
            result = RecursiveLSResultsWrapper(RecursiveLSResults(self, params, result, cov_type='custom', cov_kwds=cov_kwds))
        return result

    def smooth(self, return_ssm=False, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        result = super(RecursiveLS, self).smooth([], transformed=True, cov_type='none', return_ssm=True, **kwargs)
        if not return_ssm:
            params = result.filtered_state[:, -1]
            cov_kwds = {'custom_cov_type': 'nonrobust', 'custom_cov_params': result.filtered_state_cov[:, :, -1], 'custom_description': 'Parameters and covariance matrix estimates are RLS estimates conditional on the entire sample.'}
            result = RecursiveLSResultsWrapper(RecursiveLSResults(self, params, result, cov_type='custom', cov_kwds=cov_kwds))
        return result

    @property
    def endog_names(self):
        if False:
            for i in range(10):
                print('nop')
        endog_names = super(RecursiveLS, self).endog_names
        return endog_names[0] if isinstance(endog_names, list) else endog_names

    @property
    def param_names(self):
        if False:
            print('Hello World!')
        return self.exog_names

    @property
    def start_params(self):
        if False:
            return 10
        return np.zeros(0)

    def update(self, params, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Update the parameters of the model\n\n        Updates the representation matrices to fill in the new parameter\n        values.\n\n        Parameters\n        ----------\n        params : array_like\n            Array of new parameters.\n        transformed : bool, optional\n            Whether or not `params` is already transformed. If set to False,\n            `transform_params` is called. Default is True..\n\n        Returns\n        -------\n        params : array_like\n            Array of parameters.\n        '
        pass

class RecursiveLSResults(MLEResults):
    """
    Class to hold results from fitting a recursive least squares model.

    Parameters
    ----------
    model : RecursiveLS instance
        The fitted model instance

    Attributes
    ----------
    specification : dictionary
        Dictionary including all attributes from the recursive least squares
        model instance.

    See Also
    --------
    statsmodels.tsa.statespace.kalman_filter.FilterResults
    statsmodels.tsa.statespace.mlemodel.MLEResults
    """

    def __init__(self, model, params, filter_results, cov_type='opg', **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(RecursiveLSResults, self).__init__(model, params, filter_results, cov_type, **kwargs)
        q = max(self.loglikelihood_burn, self.k_diffuse_states)
        self.df_model = q - self.model.k_constraints
        self.df_resid = self.nobs_effective - self.df_model
        self._init_kwds = self.model._get_init_kwds()
        self.specification = Bunch(**{'k_exog': self.model.k_exog, 'k_constraints': self.model.k_constraints})
        if self.model._r_matrix is not None:
            for name in ['forecasts', 'forecasts_error', 'forecasts_error_cov', 'standardized_forecasts_error', 'forecasts_error_diffuse_cov']:
                setattr(self, name, getattr(self, name)[0:1])

    @property
    def recursive_coefficients(self):
        if False:
            while True:
                i = 10
        '\n        Estimates of regression coefficients, recursively estimated\n\n        Returns\n        -------\n        out: Bunch\n            Has the following attributes:\n\n            - `filtered`: a time series array with the filtered estimate of\n                          the component\n            - `filtered_cov`: a time series array with the filtered estimate of\n                          the variance/covariance of the component\n            - `smoothed`: a time series array with the smoothed estimate of\n                          the component\n            - `smoothed_cov`: a time series array with the smoothed estimate of\n                          the variance/covariance of the component\n            - `offset`: an integer giving the offset in the state vector where\n                        this component begins\n        '
        out = None
        spec = self.specification
        start = offset = 0
        end = offset + spec.k_exog
        out = Bunch(filtered=self.filtered_state[start:end], filtered_cov=self.filtered_state_cov[start:end, start:end], smoothed=None, smoothed_cov=None, offset=offset)
        if self.smoothed_state is not None:
            out.smoothed = self.smoothed_state[start:end]
        if self.smoothed_state_cov is not None:
            out.smoothed_cov = self.smoothed_state_cov[start:end, start:end]
        return out

    @cache_readonly
    def resid_recursive(self):
        if False:
            return 10
        '\n        Recursive residuals\n\n        Returns\n        -------\n        resid_recursive : array_like\n            An array of length `nobs` holding the recursive\n            residuals.\n\n        Notes\n        -----\n        These quantities are defined in, for example, Harvey (1989)\n        section 5.4. In fact, there he defines the standardized innovations in\n        equation 5.4.1, but in his version they have non-unit variance, whereas\n        the standardized forecast errors computed by the Kalman filter here\n        assume unit variance. To convert to Harvey\'s definition, we need to\n        multiply by the standard deviation.\n\n        Harvey notes that in smaller samples, "although the second moment\n        of the :math:`\\tilde \\sigma_*^{-1} \\tilde v_t`\'s is unity, the\n        variance is not necessarily equal to unity as the mean need not be\n        equal to zero", and he defines an alternative version (which are\n        not provided here).\n        '
        return self.filter_results.standardized_forecasts_error[0] * self.scale ** 0.5

    @cache_readonly
    def cusum(self):
        if False:
            i = 10
            return i + 15
        '\n        Cumulative sum of standardized recursive residuals statistics\n\n        Returns\n        -------\n        cusum : array_like\n            An array of length `nobs - k_exog` holding the\n            CUSUM statistics.\n\n        Notes\n        -----\n        The CUSUM statistic takes the form:\n\n        .. math::\n\n            W_t = \\frac{1}{\\hat \\sigma} \\sum_{j=k+1}^t w_j\n\n        where :math:`w_j` is the recursive residual at time :math:`j` and\n        :math:`\\hat \\sigma` is the estimate of the standard deviation\n        from the full sample.\n\n        Excludes the first `k_exog` datapoints.\n\n        Due to differences in the way :math:`\\hat \\sigma` is calculated, the\n        output of this function differs slightly from the output in the\n        R package strucchange and the Stata contributed .ado file cusum6. The\n        calculation in this package is consistent with the description of\n        Brown et al. (1975)\n\n        References\n        ----------\n        .. [*] Brown, R. L., J. Durbin, and J. M. Evans. 1975.\n           "Techniques for Testing the Constancy of\n           Regression Relationships over Time."\n           Journal of the Royal Statistical Society.\n           Series B (Methodological) 37 (2): 149-92.\n        '
        d = max(self.nobs_diffuse, self.loglikelihood_burn)
        return np.cumsum(self.resid_recursive[d:]) / np.std(self.resid_recursive[d:], ddof=1)

    @cache_readonly
    def cusum_squares(self):
        if False:
            while True:
                i = 10
        '\n        Cumulative sum of squares of standardized recursive residuals\n        statistics\n\n        Returns\n        -------\n        cusum_squares : array_like\n            An array of length `nobs - k_exog` holding the\n            CUSUM of squares statistics.\n\n        Notes\n        -----\n        The CUSUM of squares statistic takes the form:\n\n        .. math::\n\n            s_t = \\left ( \\sum_{j=k+1}^t w_j^2 \\right ) \\Bigg /\n                  \\left ( \\sum_{j=k+1}^T w_j^2 \\right )\n\n        where :math:`w_j` is the recursive residual at time :math:`j`.\n\n        Excludes the first `k_exog` datapoints.\n\n        References\n        ----------\n        .. [*] Brown, R. L., J. Durbin, and J. M. Evans. 1975.\n           "Techniques for Testing the Constancy of\n           Regression Relationships over Time."\n           Journal of the Royal Statistical Society.\n           Series B (Methodological) 37 (2): 149-92.\n        '
        d = max(self.nobs_diffuse, self.loglikelihood_burn)
        numer = np.cumsum(self.resid_recursive[d:] ** 2)
        denom = numer[-1]
        return numer / denom

    @cache_readonly
    def llf_recursive_obs(self):
        if False:
            print('Hello World!')
        '\n        (float) Loglikelihood at observation, computed from recursive residuals\n        '
        from scipy.stats import norm
        return np.log(norm.pdf(self.resid_recursive, loc=0, scale=self.scale ** 0.5))

    @cache_readonly
    def llf_recursive(self):
        if False:
            print('Hello World!')
        '\n        (float) Loglikelihood defined by recursive residuals, equivalent to OLS\n        '
        return np.sum(self.llf_recursive_obs)

    @cache_readonly
    def ssr(self):
        if False:
            i = 10
            return i + 15
        'ssr'
        d = max(self.nobs_diffuse, self.loglikelihood_burn)
        return (self.nobs - d) * self.filter_results.obs_cov[0, 0, 0]

    @cache_readonly
    def centered_tss(self):
        if False:
            return 10
        'Centered tss'
        return np.sum((self.filter_results.endog[0] - np.mean(self.filter_results.endog)) ** 2)

    @cache_readonly
    def uncentered_tss(self):
        if False:
            while True:
                i = 10
        'uncentered tss'
        return np.sum(self.filter_results.endog[0] ** 2)

    @cache_readonly
    def ess(self):
        if False:
            return 10
        'ess'
        if self.k_constant:
            return self.centered_tss - self.ssr
        else:
            return self.uncentered_tss - self.ssr

    @cache_readonly
    def rsquared(self):
        if False:
            for i in range(10):
                print('nop')
        'rsquared'
        if self.k_constant:
            return 1 - self.ssr / self.centered_tss
        else:
            return 1 - self.ssr / self.uncentered_tss

    @cache_readonly
    def mse_model(self):
        if False:
            for i in range(10):
                print('nop')
        'mse_model'
        return self.ess / self.df_model

    @cache_readonly
    def mse_resid(self):
        if False:
            print('Hello World!')
        'mse_resid'
        return self.ssr / self.df_resid

    @cache_readonly
    def mse_total(self):
        if False:
            i = 10
            return i + 15
        'mse_total'
        if self.k_constant:
            return self.centered_tss / (self.df_resid + self.df_model)
        else:
            return self.uncentered_tss / (self.df_resid + self.df_model)

    @Appender(MLEResults.get_prediction.__doc__)
    def get_prediction(self, start=None, end=None, dynamic=False, information_set='predicted', signal_only=False, index=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if start is None:
            start = self.model._index[0]
        (start, end, out_of_sample, prediction_index) = self.model._get_prediction_index(start, end, index)
        if isinstance(dynamic, (bytes, str)):
            (dynamic, _, _) = self.model._get_index_loc(dynamic)
        if self.model._r_matrix is not None and (out_of_sample or dynamic):
            raise NotImplementedError('Cannot yet perform out-of-sample or dynamic prediction in models with constraints.')
        prediction_results = self.filter_results.predict(start, end + out_of_sample + 1, dynamic, **kwargs)
        res_obj = PredictionResults(self, prediction_results, information_set=information_set, signal_only=signal_only, row_labels=prediction_index)
        return PredictionResultsWrapper(res_obj)

    def plot_recursive_coefficient(self, variables=0, alpha=0.05, legend_loc='upper left', fig=None, figsize=None):
        if False:
            print('Hello World!')
        '\n        Plot the recursively estimated coefficients on a given variable\n\n        Parameters\n        ----------\n        variables : {int, str, list[int], list[str]}, optional\n            Integer index or string name of the variable whose coefficient will\n            be plotted. Can also be an iterable of integers or strings. Default\n            is the first variable.\n        alpha : float, optional\n            The confidence intervals for the coefficient are (1 - alpha) %\n        legend_loc : str, optional\n            The location of the legend in the plot. Default is upper left.\n        fig : Figure, optional\n            If given, subplots are created in this figure instead of in a new\n            figure. Note that the grid will be created in the provided\n            figure using `fig.add_subplot()`.\n        figsize : tuple, optional\n            If a figure is created, this argument allows specifying a size.\n            The tuple is (width, height).\n\n        Notes\n        -----\n        All plots contain (1 - `alpha`) %  confidence intervals.\n        '
        if isinstance(variables, (int, str)):
            variables = [variables]
        k_variables = len(variables)
        exog_names = self.model.exog_names
        for i in range(k_variables):
            variable = variables[i]
            if isinstance(variable, str):
                variables[i] = exog_names.index(variable)
        from scipy.stats import norm
        from statsmodels.graphics.utils import _import_mpl, create_mpl_fig
        plt = _import_mpl()
        fig = create_mpl_fig(fig, figsize)
        for i in range(k_variables):
            variable = variables[i]
            ax = fig.add_subplot(k_variables, 1, i + 1)
            if hasattr(self.data, 'dates') and self.data.dates is not None:
                dates = self.data.dates._mpl_repr()
            else:
                dates = np.arange(self.nobs)
            d = max(self.nobs_diffuse, self.loglikelihood_burn)
            coef = self.recursive_coefficients
            ax.plot(dates[d:], coef.filtered[variable, d:], label='Recursive estimates: %s' % exog_names[variable])
            (handles, labels) = ax.get_legend_handles_labels()
            if alpha is not None:
                critical_value = norm.ppf(1 - alpha / 2.0)
                std_errors = np.sqrt(coef.filtered_cov[variable, variable, :])
                ci_lower = coef.filtered[variable] - critical_value * std_errors
                ci_upper = coef.filtered[variable] + critical_value * std_errors
                ci_poly = ax.fill_between(dates[d:], ci_lower[d:], ci_upper[d:], alpha=0.2)
                ci_label = '$%.3g \\%%$ confidence interval' % ((1 - alpha) * 100)
                if i == 0:
                    p = plt.Rectangle((0, 0), 1, 1, fc=ci_poly.get_facecolor()[0])
                    handles.append(p)
                    labels.append(ci_label)
            ax.legend(handles, labels, loc=legend_loc)
            if i < k_variables - 1:
                ax.xaxis.set_ticklabels([])
        fig.tight_layout()
        return fig

    def _cusum_significance_bounds(self, alpha, ddof=0, points=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Parameters\n        ----------\n        alpha : float, optional\n            The significance bound is alpha %.\n        ddof : int, optional\n            The number of periods additional to `k_exog` to exclude in\n            constructing the bounds. Default is zero. This is usually used\n            only for testing purposes.\n        points : iterable, optional\n            The points at which to evaluate the significance bounds. Default is\n            two points, beginning and end of the sample.\n\n        Notes\n        -----\n        Comparing against the cusum6 package for Stata, this does not produce\n        exactly the same confidence bands (which are produced in cusum6 by\n        lw, uw) because they burn the first k_exog + 1 periods instead of the\n        first k_exog. If this change is performed\n        (so that `tmp = (self.nobs - d - 1)**0.5`), then the output here\n        matches cusum6.\n\n        The cusum6 behavior does not seem to be consistent with\n        Brown et al. (1975); it is likely they did that because they needed\n        three initial observations to get the initial OLS estimates, whereas\n        we do not need to do that.\n        '
        if alpha == 0.01:
            scalar = 1.143
        elif alpha == 0.05:
            scalar = 0.948
        elif alpha == 0.1:
            scalar = 0.95
        else:
            raise ValueError('Invalid significance level.')
        d = max(self.nobs_diffuse, self.loglikelihood_burn)
        tmp = (self.nobs - d - ddof) ** 0.5

        def upper_line(x):
            if False:
                while True:
                    i = 10
            return scalar * tmp + 2 * scalar * (x - d) / tmp
        if points is None:
            points = np.array([d, self.nobs])
        return (-upper_line(points), upper_line(points))

    def plot_cusum(self, alpha=0.05, legend_loc='upper left', fig=None, figsize=None):
        if False:
            while True:
                i = 10
        '\n        Plot the CUSUM statistic and significance bounds.\n\n        Parameters\n        ----------\n        alpha : float, optional\n            The plotted significance bounds are alpha %.\n        legend_loc : str, optional\n            The location of the legend in the plot. Default is upper left.\n        fig : Figure, optional\n            If given, subplots are created in this figure instead of in a new\n            figure. Note that the grid will be created in the provided\n            figure using `fig.add_subplot()`.\n        figsize : tuple, optional\n            If a figure is created, this argument allows specifying a size.\n            The tuple is (width, height).\n\n        Notes\n        -----\n        Evidence of parameter instability may be found if the CUSUM statistic\n        moves out of the significance bounds.\n\n        References\n        ----------\n        .. [*] Brown, R. L., J. Durbin, and J. M. Evans. 1975.\n           "Techniques for Testing the Constancy of\n           Regression Relationships over Time."\n           Journal of the Royal Statistical Society.\n           Series B (Methodological) 37 (2): 149-92.\n        '
        from statsmodels.graphics.utils import _import_mpl, create_mpl_fig
        _import_mpl()
        fig = create_mpl_fig(fig, figsize)
        ax = fig.add_subplot(1, 1, 1)
        if hasattr(self.data, 'dates') and self.data.dates is not None:
            dates = self.data.dates._mpl_repr()
        else:
            dates = np.arange(self.nobs)
        d = max(self.nobs_diffuse, self.loglikelihood_burn)
        ax.plot(dates[d:], self.cusum, label='CUSUM')
        ax.hlines(0, dates[d], dates[-1], color='k', alpha=0.3)
        (lower_line, upper_line) = self._cusum_significance_bounds(alpha)
        ax.plot([dates[d], dates[-1]], upper_line, 'k--', label='%d%% significance' % (alpha * 100))
        ax.plot([dates[d], dates[-1]], lower_line, 'k--')
        ax.legend(loc=legend_loc)
        return fig

    def _cusum_squares_significance_bounds(self, alpha, points=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Notes\n        -----\n        Comparing against the cusum6 package for Stata, this does not produce\n        exactly the same confidence bands (which are produced in cusum6 by\n        lww, uww) because they use a different method for computing the\n        critical value; in particular, they use tabled values from\n        Table C, pp. 364-365 of "The Econometric Analysis of Time Series"\n        Harvey, (1990), and use the value given to 99 observations for any\n        larger number of observations. In contrast, we use the approximating\n        critical values suggested in Edgerton and Wells (1994) which allows\n        computing relatively good approximations for any number of\n        observations.\n        '
        d = max(self.nobs_diffuse, self.loglikelihood_burn)
        n = 0.5 * (self.nobs - d) - 1
        try:
            ix = [0.1, 0.05, 0.025, 0.01, 0.005].index(alpha / 2)
        except ValueError:
            raise ValueError('Invalid significance level.')
        scalars = _cusum_squares_scalars[:, ix]
        crit = scalars[0] / n ** 0.5 + scalars[1] / n + scalars[2] / n ** 1.5
        if points is None:
            points = np.array([d, self.nobs])
        line = (points - d) / (self.nobs - d)
        return (line - crit, line + crit)

    def plot_cusum_squares(self, alpha=0.05, legend_loc='upper left', fig=None, figsize=None):
        if False:
            return 10
        '\n        Plot the CUSUM of squares statistic and significance bounds.\n\n        Parameters\n        ----------\n        alpha : float, optional\n            The plotted significance bounds are alpha %.\n        legend_loc : str, optional\n            The location of the legend in the plot. Default is upper left.\n        fig : Figure, optional\n            If given, subplots are created in this figure instead of in a new\n            figure. Note that the grid will be created in the provided\n            figure using `fig.add_subplot()`.\n        figsize : tuple, optional\n            If a figure is created, this argument allows specifying a size.\n            The tuple is (width, height).\n\n        Notes\n        -----\n        Evidence of parameter instability may be found if the CUSUM of squares\n        statistic moves out of the significance bounds.\n\n        Critical values used in creating the significance bounds are computed\n        using the approximate formula of [1]_.\n\n        References\n        ----------\n        .. [*] Brown, R. L., J. Durbin, and J. M. Evans. 1975.\n           "Techniques for Testing the Constancy of\n           Regression Relationships over Time."\n           Journal of the Royal Statistical Society.\n           Series B (Methodological) 37 (2): 149-92.\n        .. [1] Edgerton, David, and Curt Wells. 1994.\n           "Critical Values for the Cusumsq Statistic\n           in Medium and Large Sized Samples."\n           Oxford Bulletin of Economics and Statistics 56 (3): 355-65.\n        '
        from statsmodels.graphics.utils import _import_mpl, create_mpl_fig
        _import_mpl()
        fig = create_mpl_fig(fig, figsize)
        ax = fig.add_subplot(1, 1, 1)
        if hasattr(self.data, 'dates') and self.data.dates is not None:
            dates = self.data.dates._mpl_repr()
        else:
            dates = np.arange(self.nobs)
        d = max(self.nobs_diffuse, self.loglikelihood_burn)
        ax.plot(dates[d:], self.cusum_squares, label='CUSUM of squares')
        ref_line = (np.arange(d, self.nobs) - d) / (self.nobs - d)
        ax.plot(dates[d:], ref_line, 'k', alpha=0.3)
        (lower_line, upper_line) = self._cusum_squares_significance_bounds(alpha)
        ax.plot([dates[d], dates[-1]], upper_line, 'k--', label='%d%% significance' % (alpha * 100))
        ax.plot([dates[d], dates[-1]], lower_line, 'k--')
        ax.legend(loc=legend_loc)
        return fig

class RecursiveLSResultsWrapper(MLEResultsWrapper):
    _attrs = {}
    _wrap_attrs = wrap.union_dicts(MLEResultsWrapper._wrap_attrs, _attrs)
    _methods = {}
    _wrap_methods = wrap.union_dicts(MLEResultsWrapper._wrap_methods, _methods)
wrap.populate_wrapper(RecursiveLSResultsWrapper, RecursiveLSResults)