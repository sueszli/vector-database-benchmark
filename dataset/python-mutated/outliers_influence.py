"""Influence and Outlier Measures

Created on Sun Jan 29 11:16:09 2012

Author: Josef Perktold
License: BSD-3
"""
import warnings
from statsmodels.compat.pandas import Appender
from statsmodels.compat.python import lzip
from collections import defaultdict
import numpy as np
from statsmodels.graphics._regressionplots_doc import _plot_influence_doc
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.tools import maybe_unwrap_results

def outlier_test(model_results, method='bonf', alpha=0.05, labels=None, order=False, cutoff=None):
    if False:
        print('Hello World!')
    '\n    Outlier Tests for RegressionResults instances.\n\n    Parameters\n    ----------\n    model_results : RegressionResults\n        Linear model results\n    method : str\n        - `bonferroni` : one-step correction\n        - `sidak` : one-step correction\n        - `holm-sidak` :\n        - `holm` :\n        - `simes-hochberg` :\n        - `hommel` :\n        - `fdr_bh` : Benjamini/Hochberg\n        - `fdr_by` : Benjamini/Yekutieli\n        See `statsmodels.stats.multitest.multipletests` for details.\n    alpha : float\n        familywise error rate\n    labels : None or array_like\n        If `labels` is not None, then it will be used as index to the\n        returned pandas DataFrame. See also Returns below\n    order : bool\n        Whether or not to order the results by the absolute value of the\n        studentized residuals. If labels are provided they will also be sorted.\n    cutoff : None or float in [0, 1]\n        If cutoff is not None, then the return only includes observations with\n        multiple testing corrected p-values strictly below the cutoff. The\n        returned array or dataframe can be empty if there are no outlier\n        candidates at the specified cutoff.\n\n    Returns\n    -------\n    table : ndarray or DataFrame\n        Returns either an ndarray or a DataFrame if labels is not None.\n        Will attempt to get labels from model_results if available. The\n        columns are the Studentized residuals, the unadjusted p-value,\n        and the corrected p-value according to method.\n\n    Notes\n    -----\n    The unadjusted p-value is stats.t.sf(abs(resid), df) where\n    df = df_resid - 1.\n    '
    from scipy import stats
    if labels is None:
        labels = getattr(model_results.model.data, 'row_labels', None)
    infl = getattr(model_results, 'get_influence', None)
    if infl is None:
        results = maybe_unwrap_results(model_results)
        raise AttributeError('model_results object %s does not have a get_influence method.' % results.__class__.__name__)
    resid = infl().resid_studentized_external
    if order:
        idx = np.abs(resid).argsort()[::-1]
        resid = resid[idx]
        if labels is not None:
            labels = np.asarray(labels)[idx]
    df = model_results.df_resid - 1
    unadj_p = stats.t.sf(np.abs(resid), df) * 2
    adj_p = multipletests(unadj_p, alpha=alpha, method=method)
    data = np.c_[resid, unadj_p, adj_p[1]]
    if cutoff is not None:
        mask = data[:, -1] < cutoff
        data = data[mask]
    else:
        mask = slice(None)
    if labels is not None:
        from pandas import DataFrame
        return DataFrame(data, columns=['student_resid', 'unadj_p', method + '(p)'], index=np.asarray(labels)[mask])
    return data

def reset_ramsey(res, degree=5):
    if False:
        print('Hello World!')
    "Ramsey's RESET specification test for linear models\n\n    This is a general specification test, for additional non-linear effects\n    in a model.\n\n    Parameters\n    ----------\n    degree : int\n        Maximum power to include in the RESET test.  Powers 0 and 1 are\n        excluded, so that degree tests powers 2, ..., degree of the fitted\n        values.\n\n    Notes\n    -----\n    The test fits an auxiliary OLS regression where the design matrix, exog,\n    is augmented by powers 2 to degree of the fitted values. Then it performs\n    an F-test whether these additional terms are significant.\n\n    If the p-value of the f-test is below a threshold, e.g. 0.1, then this\n    indicates that there might be additional non-linear effects in the model\n    and that the linear model is mis-specified.\n\n    References\n    ----------\n    https://en.wikipedia.org/wiki/Ramsey_RESET_test\n    "
    order = degree + 1
    k_vars = res.model.exog.shape[1]
    norm_values = np.asarray(res.fittedvalues)
    norm_values = norm_values / np.sqrt((norm_values ** 2).mean())
    y_fitted_vander = np.vander(norm_values, order)[:, :-2]
    exog = np.column_stack((res.model.exog, y_fitted_vander))
    exog /= np.sqrt((exog ** 2).mean(0))
    endog = res.model.endog / (res.model.endog ** 2).mean()
    res_aux = OLS(endog, exog).fit()
    r_matrix = np.eye(degree - 1, exog.shape[1], k_vars)
    return res_aux.f_test(r_matrix)

def variance_inflation_factor(exog, exog_idx):
    if False:
        i = 10
        return i + 15
    '\n    Variance inflation factor, VIF, for one exogenous variable\n\n    The variance inflation factor is a measure for the increase of the\n    variance of the parameter estimates if an additional variable, given by\n    exog_idx is added to the linear regression. It is a measure for\n    multicollinearity of the design matrix, exog.\n\n    One recommendation is that if VIF is greater than 5, then the explanatory\n    variable given by exog_idx is highly collinear with the other explanatory\n    variables, and the parameter estimates will have large standard errors\n    because of this.\n\n    Parameters\n    ----------\n    exog : {ndarray, DataFrame}\n        design matrix with all explanatory variables, as for example used in\n        regression\n    exog_idx : int\n        index of the exogenous variable in the columns of exog\n\n    Returns\n    -------\n    float\n        variance inflation factor\n\n    Notes\n    -----\n    This function does not save the auxiliary regression.\n\n    See Also\n    --------\n    xxx : class for regression diagnostics  TODO: does not exist yet\n\n    References\n    ----------\n    https://en.wikipedia.org/wiki/Variance_inflation_factor\n    '
    k_vars = exog.shape[1]
    exog = np.asarray(exog)
    x_i = exog[:, exog_idx]
    mask = np.arange(k_vars) != exog_idx
    x_noti = exog[:, mask]
    r_squared_i = OLS(x_i, x_noti).fit().rsquared
    vif = 1.0 / (1.0 - r_squared_i)
    return vif

class _BaseInfluenceMixin:
    """common methods between OLSInfluence and MLE/GLMInfluence
    """

    @Appender(_plot_influence_doc.format(**{'extra_params_doc': ''}))
    def plot_influence(self, external=None, alpha=0.05, criterion='cooks', size=48, plot_alpha=0.75, ax=None, **kwargs):
        if False:
            while True:
                i = 10
        if external is None:
            external = hasattr(self, '_cache') and 'res_looo' in self._cache
        from statsmodels.graphics.regressionplots import _influence_plot
        if self.hat_matrix_diag is not None:
            res = _influence_plot(self.results, self, external=external, alpha=alpha, criterion=criterion, size=size, plot_alpha=plot_alpha, ax=ax, **kwargs)
        else:
            warnings.warn('Plot uses pearson residuals and exog hat matrix.')
            res = _influence_plot(self.results, self, external=external, alpha=alpha, criterion=criterion, size=size, leverage=self.hat_matrix_exog_diag, resid=self.resid, plot_alpha=plot_alpha, ax=ax, **kwargs)
        return res

    def _plot_index(self, y, ylabel, threshold=None, title=None, ax=None, **kwds):
        if False:
            return 10
        from statsmodels.graphics import utils
        (fig, ax) = utils.create_mpl_ax(ax)
        if title is None:
            title = 'Index Plot'
        nobs = len(self.endog)
        index = np.arange(nobs)
        ax.scatter(index, y, **kwds)
        if threshold == 'all':
            large_points = np.ones(nobs, np.bool_)
        else:
            large_points = np.abs(y) > threshold
        psize = 3 * np.ones(nobs)
        labels = self.results.model.data.row_labels
        if labels is None:
            labels = np.arange(nobs)
        ax = utils.annotate_axes(np.where(large_points)[0], labels, lzip(index, y), lzip(-psize, psize), 'large', ax)
        font = {'fontsize': 16, 'color': 'black'}
        ax.set_ylabel(ylabel, **font)
        ax.set_xlabel('Observation', **font)
        ax.set_title(title, **font)
        return fig

    def plot_index(self, y_var='cooks', threshold=None, title=None, ax=None, idx=None, **kwds):
        if False:
            for i in range(10):
                print('nop')
        'index plot for influence attributes\n\n        Parameters\n        ----------\n        y_var : str\n            Name of attribute or shortcut for predefined attributes that will\n            be plotted on the y-axis.\n        threshold : None or float\n            Threshold for adding annotation with observation labels.\n            Observations for which the absolute value of the y_var is larger\n            than the threshold will be annotated. Set to a negative number to\n            label all observations or to a large number to have no annotation.\n        title : str\n            If provided, the title will replace the default "Index Plot" title.\n        ax : matplolib axis instance\n            The plot will be added to the `ax` if provided, otherwise a new\n            figure is created.\n        idx : {None, int}\n            Some attributes require an additional index to select the y-var.\n            In dfbetas this refers to the column indes.\n        kwds : optional keywords\n            Keywords will be used in the call to matplotlib scatter function.\n        '
        criterion = y_var
        if threshold is None:
            threshold = 'all'
        if criterion == 'dfbeta':
            y = self.dfbetas[:, idx]
            ylabel = 'DFBETA for ' + self.results.model.exog_names[idx]
        elif criterion.startswith('cook'):
            y = self.cooks_distance[0]
            ylabel = "Cook's distance"
        elif criterion.startswith('hat') or criterion.startswith('lever'):
            y = self.hat_matrix_diag
            ylabel = 'Leverage (diagonal of hat matrix)'
        elif criterion.startswith('cook'):
            y = self.cooks_distance[0]
            ylabel = "Cook's distance"
        elif criterion.startswith('resid_stu'):
            y = self.resid_studentized
            ylabel = 'Internally Studentized Residuals'
        else:
            y = getattr(self, y_var)
            if idx is not None:
                y = y[idx]
            ylabel = y_var
        fig = self._plot_index(y, ylabel, threshold=threshold, title=title, ax=ax, **kwds)
        return fig

class MLEInfluence(_BaseInfluenceMixin):
    """Global Influence and outlier measures (experimental)

    Parameters
    ----------
    results : instance of results class
        This only works for model and results classes that have the necessary
        helper methods.
    other arguments :
        Those are only available to override default behavior and are used
        instead of the corresponding attribute of the results class.
        By default resid_pearson is used as resid.

    Attributes
    ----------
    hat_matrix_diag (hii) : This is the generalized leverage computed as the
        local derivative of fittedvalues (predicted mean) with respect to the
        observed response for each observation.
        Not available for ZeroInflated models because of nondifferentiability.
    d_params : Change in parameters computed with one Newton step using the
        full Hessian corrected by division by (1 - hii).
        If hat_matrix_diag is not available, then the division by (1 - hii) is
        not included.
    dbetas : change in parameters divided by the standard error of parameters
        from the full model results, ``bse``.
    cooks_distance : quadratic form for change in parameters weighted by
        ``cov_params`` from the full model divided by the number of variables.
        It includes p-values based on the F-distribution which are only
        approximate outside of linear Gaussian models.
    resid_studentized : In the general MLE case resid_studentized are
        computed from the score residuals scaled by hessian factor and
        leverage. This does not use ``cov_params``.
    d_fittedvalues : local change of expected mean given the change in the
        parameters as computed in ``d_params``.
    d_fittedvalues_scaled : same as d_fittedvalues but scaled by the standard
        errors of a predicted mean of the response.
    params_one : is the one step parameter estimate computed as ``params``
        from the full sample minus ``d_params``.

    Notes
    -----
    MLEInfluence uses generic definitions based on maximum likelihood models.

    MLEInfluence produces the same results as GLMInfluence for canonical
    links (verified for GLM Binomial, Poisson and Gaussian). There will be
    some differences for non-canonical links or if a robust cov_type is used.
    For example, the generalized leverage differs from the definition of the
    GLM hat matrix in the case of Probit, which corresponds to family
    Binomial with a non-canonical link.

    The extension to non-standard models, e.g. multi-link model like
    BetaModel and the ZeroInflated models is still experimental and might still
    change.
    Additonally, ZeroInflated and some threshold models have a
    nondifferentiability in the generalized leverage. How this case is treated
    might also change.

    Warning: This does currently not work for constrained or penalized models,
    e.g. models estimated with fit_constrained or fit_regularized.

    This has not yet been tested for correctness when offset or exposure
    are used, although they should be supported by the code.

    status: experimental,
    This class will need changes to support different kinds of models, e.g.
    extra parameters in discrete.NegativeBinomial or two-part models like
    ZeroInflatedPoisson.
    """

    def __init__(self, results, resid=None, endog=None, exog=None, hat_matrix_diag=None, cov_params=None, scale=None):
        if False:
            for i in range(10):
                print('nop')
        self.results = results = maybe_unwrap_results(results)
        (self.nobs, self.k_vars) = results.model.exog.shape
        self.k_params = np.size(results.params)
        self.endog = endog if endog is not None else results.model.endog
        self.exog = exog if exog is not None else results.model.exog
        self.scale = scale if scale is not None else results.scale
        if resid is not None:
            self.resid = resid
        else:
            self.resid = getattr(results, 'resid_pearson', None)
            if self.resid is not None:
                self.resid = self.resid / np.sqrt(self.scale)
        self.cov_params = cov_params if cov_params is not None else results.cov_params()
        self.model_class = results.model.__class__
        self.hessian = self.results.model.hessian(self.results.params)
        self.score_obs = self.results.model.score_obs(self.results.params)
        if hat_matrix_diag is not None:
            self._hat_matrix_diag = hat_matrix_diag

    @cache_readonly
    def hat_matrix_diag(self):
        if False:
            for i in range(10):
                print('nop')
        'Diagonal of the generalized leverage\n\n        This is the analogue of the hat matrix diagonal for general MLE.\n        '
        if hasattr(self, '_hat_matrix_diag'):
            return self._hat_matrix_diag
        try:
            dsdy = self.results.model._deriv_score_obs_dendog(self.results.params)
        except NotImplementedError:
            dsdy = None
        if dsdy is None:
            warnings.warn('hat matrix is not available, missing derivatives', UserWarning)
            return None
        dmu_dp = self.results.model._deriv_mean_dparams(self.results.params)
        h = (dmu_dp * np.linalg.solve(-self.hessian, dsdy.T).T).sum(1)
        return h

    @cache_readonly
    def hat_matrix_exog_diag(self):
        if False:
            for i in range(10):
                print('nop')
        'Diagonal of the hat_matrix using only exog as in OLS\n\n        '
        get_exogs = getattr(self.results.model, '_get_exogs', None)
        if get_exogs is not None:
            exog = np.column_stack(get_exogs())
        else:
            exog = self.exog
        return (exog * np.linalg.pinv(exog).T).sum(1)

    @cache_readonly
    def d_params(self):
        if False:
            for i in range(10):
                print('nop')
        'Approximate change in parameter estimates when dropping observation.\n\n        This uses one-step approximation of the parameter change to deleting\n        one observation.\n        '
        so_noti = self.score_obs.sum(0) - self.score_obs
        beta_i = np.linalg.solve(self.hessian, so_noti.T).T
        if self.hat_matrix_diag is not None:
            beta_i /= (1 - self.hat_matrix_diag)[:, None]
        return beta_i

    @cache_readonly
    def dfbetas(self):
        if False:
            return 10
        'Scaled change in parameter estimates.\n\n        The one-step change of parameters in d_params is rescaled by dividing\n        by the standard error of the parameter estimate given by results.bse.\n        '
        beta_i = self.d_params / self.results.bse
        return beta_i

    @cache_readonly
    def params_one(self):
        if False:
            while True:
                i = 10
        'Parameter estimate based on one-step approximation.\n\n        This the one step parameter estimate computed as\n        ``params`` from the full sample minus ``d_params``.\n        '
        return self.results.params - self.d_params

    @cache_readonly
    def cooks_distance(self):
        if False:
            i = 10
            return i + 15
        "Cook's distance and p-values.\n\n        Based on one step approximation d_params and on results.cov_params\n        Cook's distance divides by the number of explanatory variables.\n\n        p-values are based on the F-distribution which are only approximate\n        outside of linear Gaussian models.\n\n        Warning: The definition of p-values might change if we switch to using\n        chi-square distribution instead of F-distribution, or if we make it\n        dependent on the fit keyword use_t.\n        "
        cooks_d2 = (self.d_params * np.linalg.solve(self.cov_params, self.d_params.T).T).sum(1)
        cooks_d2 /= self.k_params
        from scipy import stats
        pvals = stats.f.sf(cooks_d2, self.k_params, self.results.df_resid)
        return (cooks_d2, pvals)

    @cache_readonly
    def resid_studentized(self):
        if False:
            while True:
                i = 10
        'studentized default residuals.\n\n        This uses the residual in `resid` attribute, which is by default\n        resid_pearson and studentizes is using the generalized leverage.\n\n        self.resid / np.sqrt(1 - self.hat_matrix_diag)\n\n        Studentized residuals are not available if hat_matrix_diag is None.\n\n        '
        return self.resid / np.sqrt(1 - self.hat_matrix_diag)

    def resid_score_factor(self):
        if False:
            for i in range(10):
                print('nop')
        'Score residual divided by sqrt of hessian factor.\n\n        experimental, agrees with GLMInfluence for Binomial and Gaussian.\n        This corresponds to considering the linear predictors as parameters\n        of the model.\n\n        Note: Nhis might have nan values if second derivative, hessian_factor,\n        is positive, i.e. loglikelihood is not globally concave w.r.t. linear\n        predictor. (This occured in an example for GeneralizedPoisson)\n        '
        from statsmodels.genmod.generalized_linear_model import GLM
        sf = self.results.model.score_factor(self.results.params)
        hf = self.results.model.hessian_factor(self.results.params)
        if isinstance(sf, tuple):
            sf = sf[0]
        if isinstance(hf, tuple):
            hf = hf[0]
        if not isinstance(self.results.model, GLM):
            hf = -hf
        return sf / np.sqrt(hf) / np.sqrt(1 - self.hat_matrix_diag)

    def resid_score(self, joint=True, index=None, studentize=False):
        if False:
            print('Hello World!')
        'Score observations scaled by inverse hessian.\n\n        Score residual in resid_score are defined in analogy to a score test\n        statistic for each observation.\n\n        Parameters\n        ----------\n        joint : bool\n            If joint is true, then a quadratic form similar to score_test is\n            returned for each observation.\n            If joint is false, then standardized score_obs are returned. The\n            returned array is two-dimensional\n        index : ndarray (optional)\n            Optional index to select a subset of score_obs columns.\n            By default, all columns of score_obs will be used.\n        studentize : bool\n            If studentize is true, the the scaled residuals are also\n            studentized using the generalized leverage.\n\n        Returns\n        -------\n        array :  1-D or 2-D residuals\n\n        Notes\n        -----\n        Status: experimental\n\n        Because of the one srep approacimation of d_params, score residuals\n        are identical to cooks_distance, except for\n\n        - cooks_distance is normalized by the number of parameters\n        - cooks_distance uses cov_params, resid_score is based on Hessian.\n          This will make them differ in the case of robust cov_params.\n\n        '
        score_obs = self.results.model.score_obs(self.results.params)
        hess = self.results.model.hessian(self.results.params)
        if index is not None:
            score_obs = score_obs[:, index]
            hess = hess[index[:, None], index]
        if joint:
            resid = (score_obs.T * np.linalg.solve(-hess, score_obs.T)).sum(0)
        else:
            resid = score_obs / np.sqrt(np.diag(-hess))
        if studentize:
            if joint:
                resid /= np.sqrt(1 - self.hat_matrix_diag)
            else:
                resid /= np.sqrt(1 - self.hat_matrix_diag[:, None])
        return resid

    @cache_readonly
    def _get_prediction(self):
        if False:
            print('Hello World!')
        with warnings.catch_warnings():
            msg = 'linear keyword is deprecated, use which="linear"'
            warnings.filterwarnings('ignore', message=msg, category=FutureWarning)
            pred = self.results.get_prediction()
        return pred

    @cache_readonly
    def d_fittedvalues(self):
        if False:
            i = 10
            return i + 15
        'Change in expected response, fittedvalues.\n\n        Local change of expected mean given the change in the parameters as\n        computed in d_params.\n\n        Notes\n        -----\n        This uses the one-step approximation of the parameter change to\n        deleting one observation ``d_params``.\n        '
        params = np.asarray(self.results.params)
        deriv = self.results.model._deriv_mean_dparams(params)
        return (deriv * self.d_params).sum(1)

    @property
    def d_fittedvalues_scaled(self):
        if False:
            print('Hello World!')
        '\n        Change in fittedvalues scaled by standard errors.\n\n        This uses one-step approximation of the parameter change to deleting\n        one observation ``d_params``, and divides by the standard errors\n        for the predicted mean provided by results.get_prediction.\n        '
        return self.d_fittedvalues / self._get_prediction.se

    def summary_frame(self):
        if False:
            print('Hello World!')
        "\n        Creates a DataFrame with influence results.\n\n        Returns\n        -------\n        frame : pandas DataFrame\n            A DataFrame with selected results for each observation.\n            The index will be the same as provided to the model.\n\n        Notes\n        -----\n        The resultant DataFrame contains six variables in addition to the\n        ``dfbetas``. These are:\n\n        * cooks_d : Cook's Distance defined in ``cooks_distance``\n        * standard_resid : Standardized residuals defined in\n          `resid_studentizedl`\n        * hat_diag : The diagonal of the projection, or hat, matrix defined in\n          `hat_matrix_diag`. Not included if None.\n        * dffits_internal : DFFITS statistics using internally Studentized\n          residuals defined in `d_fittedvalues_scaled`\n        "
        from pandas import DataFrame
        data = self.results.model.data
        row_labels = data.row_labels
        beta_labels = ['dfb_' + i for i in data.xnames]
        if self.hat_matrix_diag is not None:
            summary_data = DataFrame(dict(cooks_d=self.cooks_distance[0], standard_resid=self.resid_studentized, hat_diag=self.hat_matrix_diag, dffits_internal=self.d_fittedvalues_scaled), index=row_labels)
        else:
            summary_data = DataFrame(dict(cooks_d=self.cooks_distance[0], dffits_internal=self.d_fittedvalues_scaled), index=row_labels)
        dfbeta = DataFrame(self.dfbetas, columns=beta_labels, index=row_labels)
        return dfbeta.join(summary_data)

class OLSInfluence(_BaseInfluenceMixin):
    """class to calculate outlier and influence measures for OLS result

    Parameters
    ----------
    results : RegressionResults
        currently assumes the results are from an OLS regression

    Notes
    -----
    One part of the results can be calculated without any auxiliary regression
    (some of which have the `_internal` postfix in the name. Other statistics
    require leave-one-observation-out (LOOO) auxiliary regression, and will be
    slower (mainly results with `_external` postfix in the name).
    The auxiliary LOOO regression only the required results are stored.

    Using the LOO measures is currently only recommended if the data set
    is not too large. One possible approach for LOOO measures would be to
    identify possible problem observations with the _internal measures, and
    then run the leave-one-observation-out only with observations that are
    possible outliers. (However, this is not yet available in an automated way.)

    This should be extended to general least squares.

    The leave-one-variable-out (LOVO) auxiliary regression are currently not
    used.
    """

    def __init__(self, results):
        if False:
            print('Hello World!')
        self.results = maybe_unwrap_results(results)
        (self.nobs, self.k_vars) = results.model.exog.shape
        self.endog = results.model.endog
        self.exog = results.model.exog
        self.resid = results.resid
        self.model_class = results.model.__class__
        self.scale = results.mse_resid
        self.aux_regression_exog = {}
        self.aux_regression_endog = {}

    @cache_readonly
    def hat_matrix_diag(self):
        if False:
            return 10
        'Diagonal of the hat_matrix for OLS\n\n        Notes\n        -----\n        temporarily calculated here, this should go to model class\n        '
        return (self.exog * self.results.model.pinv_wexog.T).sum(1)

    @cache_readonly
    def resid_press(self):
        if False:
            return 10
        'PRESS residuals\n        '
        hii = self.hat_matrix_diag
        return self.resid / (1 - hii)

    @cache_readonly
    def influence(self):
        if False:
            i = 10
            return i + 15
        'Influence measure\n\n        matches the influence measure that gretl reports\n        u * h / (1 - h)\n        where u are the residuals and h is the diagonal of the hat_matrix\n        '
        hii = self.hat_matrix_diag
        return self.resid * hii / (1 - hii)

    @cache_readonly
    def hat_diag_factor(self):
        if False:
            while True:
                i = 10
        'Factor of diagonal of hat_matrix used in influence\n\n        this might be useful for internal reuse\n        h / (1 - h)\n        '
        hii = self.hat_matrix_diag
        return hii / (1 - hii)

    @cache_readonly
    def ess_press(self):
        if False:
            i = 10
            return i + 15
        'Error sum of squares of PRESS residuals\n        '
        return np.dot(self.resid_press, self.resid_press)

    @cache_readonly
    def resid_studentized(self):
        if False:
            for i in range(10):
                print('nop')
        'Studentized residuals using variance from OLS\n\n        alias for resid_studentized_internal for compatibility with\n        MLEInfluence this uses sigma from original estimate and does\n        not require leave one out loop\n        '
        return self.resid_studentized_internal

    @cache_readonly
    def resid_studentized_internal(self):
        if False:
            while True:
                i = 10
        'Studentized residuals using variance from OLS\n\n        this uses sigma from original estimate\n        does not require leave one out loop\n        '
        return self.get_resid_studentized_external(sigma=None)

    @cache_readonly
    def resid_studentized_external(self):
        if False:
            while True:
                i = 10
        'Studentized residuals using LOOO variance\n\n        this uses sigma from leave-one-out estimates\n\n        requires leave one out loop for observations\n        '
        sigma_looo = np.sqrt(self.sigma2_not_obsi)
        return self.get_resid_studentized_external(sigma=sigma_looo)

    def get_resid_studentized_external(self, sigma=None):
        if False:
            return 10
        'calculate studentized residuals\n\n        Parameters\n        ----------\n        sigma : None or float\n            estimate of the standard deviation of the residuals. If None, then\n            the estimate from the regression results is used.\n\n        Returns\n        -------\n        stzd_resid : ndarray\n            studentized residuals\n\n        Notes\n        -----\n        studentized residuals are defined as ::\n\n           resid / sigma / np.sqrt(1 - hii)\n\n        where resid are the residuals from the regression, sigma is an\n        estimate of the standard deviation of the residuals, and hii is the\n        diagonal of the hat_matrix.\n        '
        hii = self.hat_matrix_diag
        if sigma is None:
            sigma2_est = self.scale
            sigma = np.sqrt(sigma2_est)
        return self.resid / sigma / np.sqrt(1 - hii)

    @cache_readonly
    def cooks_distance(self):
        if False:
            return 10
        "\n        Cooks distance\n\n        Uses original results, no nobs loop\n\n        References\n        ----------\n        .. [*] Eubank, R. L. (1999). Nonparametric regression and spline\n            smoothing. CRC press.\n        .. [*] Cook's distance. (n.d.). In Wikipedia. July 2019, from\n            https://en.wikipedia.org/wiki/Cook%27s_distance\n        "
        hii = self.hat_matrix_diag
        cooks_d2 = self.resid_studentized ** 2 / self.k_vars
        cooks_d2 *= hii / (1 - hii)
        from scipy import stats
        pvals = stats.f.sf(cooks_d2, self.k_vars, self.results.df_resid)
        return (cooks_d2, pvals)

    @cache_readonly
    def dffits_internal(self):
        if False:
            i = 10
            return i + 15
        'dffits measure for influence of an observation\n\n        based on resid_studentized_internal\n        uses original results, no nobs loop\n        '
        hii = self.hat_matrix_diag
        dffits_ = self.resid_studentized_internal * np.sqrt(hii / (1 - hii))
        dffits_threshold = 2 * np.sqrt(self.k_vars * 1.0 / self.nobs)
        return (dffits_, dffits_threshold)

    @cache_readonly
    def dffits(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        dffits measure for influence of an observation\n\n        based on resid_studentized_external,\n        uses results from leave-one-observation-out loop\n\n        It is recommended that observations with dffits large than a\n        threshold of 2 sqrt{k / n} where k is the number of parameters, should\n        be investigated.\n\n        Returns\n        -------\n        dffits : float\n        dffits_threshold : float\n\n        References\n        ----------\n        `Wikipedia <https://en.wikipedia.org/wiki/DFFITS>`_\n        '
        hii = self.hat_matrix_diag
        dffits_ = self.resid_studentized_external * np.sqrt(hii / (1 - hii))
        dffits_threshold = 2 * np.sqrt(self.k_vars * 1.0 / self.nobs)
        return (dffits_, dffits_threshold)

    @cache_readonly
    def dfbetas(self):
        if False:
            for i in range(10):
                print('nop')
        'dfbetas\n\n        uses results from leave-one-observation-out loop\n        '
        dfbetas = self.results.params - self.params_not_obsi
        dfbetas /= np.sqrt(self.sigma2_not_obsi[:, None])
        dfbetas /= np.sqrt(np.diag(self.results.normalized_cov_params))
        return dfbetas

    @cache_readonly
    def dfbeta(self):
        if False:
            while True:
                i = 10
        'dfbetas\n\n        uses results from leave-one-observation-out loop\n        '
        dfbeta = self.results.params - self.params_not_obsi
        return dfbeta

    @cache_readonly
    def sigma2_not_obsi(self):
        if False:
            for i in range(10):
                print('nop')
        "error variance for all LOOO regressions\n\n        This is 'mse_resid' from each auxiliary regression.\n\n        uses results from leave-one-observation-out loop\n        "
        return np.asarray(self._res_looo['mse_resid'])

    @property
    def params_not_obsi(self):
        if False:
            print('Hello World!')
        'parameter estimates for all LOOO regressions\n\n        uses results from leave-one-observation-out loop\n        '
        return np.asarray(self._res_looo['params'])

    @property
    def det_cov_params_not_obsi(self):
        if False:
            print('Hello World!')
        'determinant of cov_params of all LOOO regressions\n\n        uses results from leave-one-observation-out loop\n        '
        return np.asarray(self._res_looo['det_cov_params'])

    @cache_readonly
    def cov_ratio(self):
        if False:
            print('Hello World!')
        'covariance ratio between LOOO and original\n\n        This uses determinant of the estimate of the parameter covariance\n        from leave-one-out estimates.\n        requires leave one out loop for observations\n        '
        cov_ratio = self.det_cov_params_not_obsi / np.linalg.det(self.results.cov_params())
        return cov_ratio

    @cache_readonly
    def resid_var(self):
        if False:
            i = 10
            return i + 15
        'estimate of variance of the residuals\n\n        ::\n\n           sigma2 = sigma2_OLS * (1 - hii)\n\n        where hii is the diagonal of the hat matrix\n        '
        return self.scale * (1 - self.hat_matrix_diag)

    @cache_readonly
    def resid_std(self):
        if False:
            print('Hello World!')
        'estimate of standard deviation of the residuals\n\n        See Also\n        --------\n        resid_var\n        '
        return np.sqrt(self.resid_var)

    def _ols_xnoti(self, drop_idx, endog_idx='endog', store=True):
        if False:
            i = 10
            return i + 15
        "regression results from LOVO auxiliary regression with cache\n\n\n        The result instances are stored, which could use a large amount of\n        memory if the datasets are large. There are too many combinations to\n        store them all, except for small problems.\n\n        Parameters\n        ----------\n        drop_idx : int\n            index of exog that is dropped from the regression\n        endog_idx : 'endog' or int\n            If 'endog', then the endogenous variable of the result instance\n            is regressed on the exogenous variables, excluding the one at\n            drop_idx. If endog_idx is an integer, then the exog with that\n            index is regressed with OLS on all other exogenous variables.\n            (The latter is the auxiliary regression for the variance inflation\n            factor.)\n\n        this needs more thought, memory versus speed\n        not yet used in any other parts, not sufficiently tested\n        "
        if endog_idx == 'endog':
            stored = self.aux_regression_endog
            if hasattr(stored, drop_idx):
                return stored[drop_idx]
            x_i = self.results.model.endog
        else:
            try:
                self.aux_regression_exog[endog_idx][drop_idx]
            except KeyError:
                pass
            stored = self.aux_regression_exog[endog_idx]
            stored = {}
            x_i = self.exog[:, endog_idx]
        k_vars = self.exog.shape[1]
        mask = np.arange(k_vars) != drop_idx
        x_noti = self.exog[:, mask]
        res = OLS(x_i, x_noti).fit()
        if store:
            stored[drop_idx] = res
        return res

    def _get_drop_vari(self, attributes):
        if False:
            i = 10
            return i + 15
        '\n        regress endog on exog without one of the variables\n\n        This uses a k_vars loop, only attributes of the OLS instance are\n        stored.\n\n        Parameters\n        ----------\n        attributes : list[str]\n           These are the names of the attributes of the auxiliary OLS results\n           instance that are stored and returned.\n\n        not yet used\n        '
        from statsmodels.sandbox.tools.cross_val import LeaveOneOut
        endog = self.results.model.endog
        exog = self.exog
        cv_iter = LeaveOneOut(self.k_vars)
        res_loo = defaultdict(list)
        for (inidx, outidx) in cv_iter:
            for att in attributes:
                res_i = self.model_class(endog, exog[:, inidx]).fit()
                res_loo[att].append(getattr(res_i, att))
        return res_loo

    @cache_readonly
    def _res_looo(self):
        if False:
            i = 10
            return i + 15
        "collect required results from the LOOO loop\n\n        all results will be attached.\n        currently only 'params', 'mse_resid', 'det_cov_params' are stored\n\n        regresses endog on exog dropping one observation at a time\n\n        this uses a nobs loop, only attributes of the OLS instance are stored.\n        "
        from statsmodels.sandbox.tools.cross_val import LeaveOneOut

        def get_det_cov_params(res):
            if False:
                while True:
                    i = 10
            return np.linalg.det(res.cov_params())
        endog = self.results.model.endog
        exog = self.results.model.exog
        params = np.zeros(exog.shape, dtype=float)
        mse_resid = np.zeros(endog.shape, dtype=float)
        det_cov_params = np.zeros(endog.shape, dtype=float)
        cv_iter = LeaveOneOut(self.nobs)
        for (inidx, outidx) in cv_iter:
            res_i = self.model_class(endog[inidx], exog[inidx]).fit()
            params[outidx] = res_i.params
            mse_resid[outidx] = res_i.mse_resid
            det_cov_params[outidx] = get_det_cov_params(res_i)
        return dict(params=params, mse_resid=mse_resid, det_cov_params=det_cov_params)

    def summary_frame(self):
        if False:
            print('Hello World!')
        "\n        Creates a DataFrame with all available influence results.\n\n        Returns\n        -------\n        frame : DataFrame\n            A DataFrame with all results.\n\n        Notes\n        -----\n        The resultant DataFrame contains six variables in addition to the\n        DFBETAS. These are:\n\n        * cooks_d : Cook's Distance defined in `Influence.cooks_distance`\n        * standard_resid : Standardized residuals defined in\n          `Influence.resid_studentized_internal`\n        * hat_diag : The diagonal of the projection, or hat, matrix defined in\n          `Influence.hat_matrix_diag`\n        * dffits_internal : DFFITS statistics using internally Studentized\n          residuals defined in `Influence.dffits_internal`\n        * dffits : DFFITS statistics using externally Studentized residuals\n          defined in `Influence.dffits`\n        * student_resid : Externally Studentized residuals defined in\n          `Influence.resid_studentized_external`\n        "
        from pandas import DataFrame
        data = self.results.model.data
        row_labels = data.row_labels
        beta_labels = ['dfb_' + i for i in data.xnames]
        summary_data = DataFrame(dict(cooks_d=self.cooks_distance[0], standard_resid=self.resid_studentized_internal, hat_diag=self.hat_matrix_diag, dffits_internal=self.dffits_internal[0], student_resid=self.resid_studentized_external, dffits=self.dffits[0]), index=row_labels)
        dfbeta = DataFrame(self.dfbetas, columns=beta_labels, index=row_labels)
        return dfbeta.join(summary_data)

    def summary_table(self, float_fmt='%6.3f'):
        if False:
            i = 10
            return i + 15
        'create a summary table with all influence and outlier measures\n\n        This does currently not distinguish between statistics that can be\n        calculated from the original regression results and for which a\n        leave-one-observation-out loop is needed\n\n        Returns\n        -------\n        res : SimpleTable\n           SimpleTable instance with the results, can be printed\n\n        Notes\n        -----\n        This also attaches table_data to the instance.\n        '
        table_raw = [('obs', np.arange(self.nobs)), ('endog', self.endog), ('fitted\nvalue', self.results.fittedvalues), ("Cook's\nd", self.cooks_distance[0]), ('student.\nresidual', self.resid_studentized_internal), ('hat diag', self.hat_matrix_diag), ('dffits \ninternal', self.dffits_internal[0]), ('ext.stud.\nresidual', self.resid_studentized_external), ('dffits', self.dffits[0])]
        (colnames, data) = lzip(*table_raw)
        data = np.column_stack(data)
        self.table_data = data
        from copy import deepcopy
        from statsmodels.iolib.table import SimpleTable, default_html_fmt
        from statsmodels.iolib.tableformatting import fmt_base
        fmt = deepcopy(fmt_base)
        fmt_html = deepcopy(default_html_fmt)
        fmt['data_fmts'] = ['%4d'] + [float_fmt] * (data.shape[1] - 1)
        return SimpleTable(data, headers=colnames, txt_fmt=fmt, html_fmt=fmt_html)

def summary_table(res, alpha=0.05):
    if False:
        while True:
            i = 10
    '\n    Generate summary table of outlier and influence similar to SAS\n\n    Parameters\n    ----------\n    alpha : float\n       significance level for confidence interval\n\n    Returns\n    -------\n    st : SimpleTable\n       table with results that can be printed\n    data : ndarray\n       calculated measures and statistics for the table\n    ss2 : list[str]\n       column_names for table (Note: rows of table are observations)\n    '
    from scipy import stats
    from statsmodels.sandbox.regression.predstd import wls_prediction_std
    infl = OLSInfluence(res)
    predict_mean_se = np.sqrt(infl.hat_matrix_diag * res.mse_resid)
    tppf = stats.t.isf(alpha / 2.0, res.df_resid)
    predict_mean_ci = np.column_stack([res.fittedvalues - tppf * predict_mean_se, res.fittedvalues + tppf * predict_mean_se])
    tmp = wls_prediction_std(res, alpha=alpha)
    (predict_se, predict_ci_low, predict_ci_upp) = tmp
    predict_ci = np.column_stack((predict_ci_low, predict_ci_upp))
    resid_se = np.sqrt(res.mse_resid * (1 - infl.hat_matrix_diag))
    table_sm = np.column_stack([np.arange(res.nobs) + 1, res.model.endog, res.fittedvalues, predict_mean_se, predict_mean_ci[:, 0], predict_mean_ci[:, 1], predict_ci[:, 0], predict_ci[:, 1], res.resid, resid_se, infl.resid_studentized_internal, infl.cooks_distance[0]])
    data = table_sm
    ss2 = ['Obs', 'Dep Var\nPopulation', 'Predicted\nValue', 'Std Error\nMean Predict', 'Mean ci\n95% low', 'Mean ci\n95% upp', 'Predict ci\n95% low', 'Predict ci\n95% upp', 'Residual', 'Std Error\nResidual', 'Student\nResidual', "Cook's\nD"]
    colnames = ss2
    from copy import deepcopy
    from statsmodels.iolib.table import SimpleTable, default_html_fmt
    from statsmodels.iolib.tableformatting import fmt_base
    fmt = deepcopy(fmt_base)
    fmt_html = deepcopy(default_html_fmt)
    fmt['data_fmts'] = ['%4d'] + ['%6.3f'] * (data.shape[1] - 1)
    st = SimpleTable(data, headers=colnames, txt_fmt=fmt, html_fmt=fmt_html)
    return (st, data, ss2)

class GLMInfluence(MLEInfluence):
    """Influence and outlier measures (experimental)

    This uses partly formulas specific to GLM, specifically cooks_distance
    is based on the hessian, i.e. observed or expected information matrix and
    not on cov_params, in contrast to MLEInfluence.
    Standardization for changes in parameters, in fittedvalues and in
    the linear predictor are based on cov_params.

    Parameters
    ----------
    results : instance of results class
        This only works for model and results classes that have the necessary
        helper methods.
    other arguments are only to override default behavior and are used instead
    of the corresponding attribute of the results class.
    By default resid_pearson is used as resid.

    Attributes
    ----------
    dbetas
        change in parameters divided by the standard error of parameters from
        the full model results, ``bse``.
    d_fittedvalues_scaled
        same as d_fittedvalues but scaled by the standard errors of a
        predicted mean of the response.
    d_linpred
        local change in linear prediction.
    d_linpred_scale
        local change in linear prediction scaled by the standard errors for
        the prediction based on cov_params.

    Notes
    -----
    This has not yet been tested for correctness when offset or exposure
    are used, although they should be supported by the code.

    Some GLM specific measures like d_deviance are still missing.

    Computing an explicit leave-one-observation-out (LOOO) loop is included
    but no influence measures are currently computed from it.
    """

    @cache_readonly
    def hat_matrix_diag(self):
        if False:
            i = 10
            return i + 15
        '\n        Diagonal of the hat_matrix for GLM\n\n        Notes\n        -----\n        This returns the diagonal of the hat matrix that was provided as\n        argument to GLMInfluence or computes it using the results method\n        `get_hat_matrix`.\n        '
        if hasattr(self, '_hat_matrix_diag'):
            return self._hat_matrix_diag
        else:
            return self.results.get_hat_matrix()

    @cache_readonly
    def d_params(self):
        if False:
            return 10
        'Change in parameter estimates\n\n        Notes\n        -----\n        This uses one-step approximation of the parameter change to deleting\n        one observation.\n        '
        beta_i = np.linalg.pinv(self.exog) * self.resid_studentized
        beta_i /= np.sqrt(1 - self.hat_matrix_diag)
        return beta_i.T

    @cache_readonly
    def resid_studentized(self):
        if False:
            print('Hello World!')
        '\n        Internally studentized pearson residuals\n\n        Notes\n        -----\n        residuals / sqrt( scale * (1 - hii))\n\n        where residuals are those provided to GLMInfluence which are\n        pearson residuals by default, and\n        hii is the diagonal of the hat matrix.\n        '
        return super().resid_studentized

    @cache_readonly
    def cooks_distance(self):
        if False:
            return 10
        "Cook's distance\n\n        Notes\n        -----\n        Based on one step approximation using resid_studentized and\n        hat_matrix_diag for the computation.\n\n        Cook's distance divides by the number of explanatory variables.\n\n        Computed using formulas for GLM and does not use results.cov_params.\n        It includes p-values based on the F-distribution which are only\n        approximate outside of linear Gaussian models.\n        "
        hii = self.hat_matrix_diag
        cooks_d2 = self.resid_studentized ** 2 / self.k_vars
        cooks_d2 *= hii / (1 - hii)
        from scipy import stats
        pvals = stats.f.sf(cooks_d2, self.k_vars, self.results.df_resid)
        return (cooks_d2, pvals)

    @property
    def d_linpred(self):
        if False:
            return 10
        '\n        Change in linear prediction\n\n        This uses one-step approximation of the parameter change to deleting\n        one observation ``d_params``.\n        '
        exog = self.results.model.exog
        return (exog * self.d_params).sum(1)

    @property
    def d_linpred_scaled(self):
        if False:
            return 10
        '\n        Change in linpred scaled by standard errors\n\n        This uses one-step approximation of the parameter change to deleting\n        one observation ``d_params``, and divides by the standard errors\n        for linpred provided by results.get_prediction.\n        '
        return self.d_linpred / self._get_prediction.linpred.se

    @property
    def _fittedvalues_one(self):
        if False:
            i = 10
            return i + 15
        'experimental code\n        '
        warnings.warn('this ignores offset and exposure', UserWarning)
        exog = self.results.model.exog
        fitted = np.array([self.results.model.predict(pi, exog[i]) for (i, pi) in enumerate(self.params_one)])
        return fitted.squeeze()

    @property
    def _diff_fittedvalues_one(self):
        if False:
            i = 10
            return i + 15
        'experimental code\n        '
        return self.results.predict() - self._fittedvalues_one

    @cache_readonly
    def _res_looo(self):
        if False:
            i = 10
            return i + 15
        "collect required results from the LOOO loop\n\n        all results will be attached.\n        currently only 'params', 'mse_resid', 'det_cov_params' are stored\n\n        Reestimates the model with endog and exog dropping one observation\n        at a time\n\n        This uses a nobs loop, only attributes of the results instance are\n        stored.\n\n        Warning: This will need refactoring and API changes to be able to\n        add options.\n        "
        from statsmodels.sandbox.tools.cross_val import LeaveOneOut
        get_det_cov_params = lambda res: np.linalg.det(res.cov_params())
        endog = self.results.model.endog
        exog = self.results.model.exog
        init_kwds = self.results.model._get_init_kwds()
        freq_weights = init_kwds.pop('freq_weights')
        var_weights = init_kwds.pop('var_weights')
        offset = offset_ = init_kwds.pop('offset')
        exposure = exposure_ = init_kwds.pop('exposure')
        n_trials = init_kwds.pop('n_trials', None)
        if hasattr(init_kwds['family'], 'initialize'):
            is_binomial = True
        else:
            is_binomial = False
        params = np.zeros(exog.shape, dtype=float)
        scale = np.zeros(endog.shape, dtype=float)
        det_cov_params = np.zeros(endog.shape, dtype=float)
        cv_iter = LeaveOneOut(self.nobs)
        for (inidx, outidx) in cv_iter:
            if offset is not None:
                offset_ = offset[inidx]
            if exposure is not None:
                exposure_ = exposure[inidx]
            if n_trials is not None:
                init_kwds['n_trials'] = n_trials[inidx]
            mod_i = self.model_class(endog[inidx], exog[inidx], offset=offset_, exposure=exposure_, freq_weights=freq_weights[inidx], var_weights=var_weights[inidx], **init_kwds)
            if is_binomial:
                mod_i.family.n = init_kwds['n_trials']
            res_i = mod_i.fit(start_params=self.results.params, method='newton')
            params[outidx] = res_i.params.copy()
            scale[outidx] = res_i.scale
            det_cov_params[outidx] = get_det_cov_params(res_i)
        return dict(params=params, scale=scale, mse_resid=scale, det_cov_params=det_cov_params)