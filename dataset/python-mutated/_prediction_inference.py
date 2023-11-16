"""
Created on Fri Dec 19 11:29:18 2014

Author: Josef Perktold
License: BSD-3

"""
import numpy as np
from scipy import stats
import pandas as pd

class PredictionResultsBase:
    """Based class for get_prediction results
    """

    def __init__(self, predicted, var_pred, func=None, deriv=None, df=None, dist=None, row_labels=None, **kwds):
        if False:
            for i in range(10):
                print('nop')
        self.predicted = predicted
        self.var_pred = var_pred
        self.func = func
        self.deriv = deriv
        self.df = df
        self.row_labels = row_labels
        self.__dict__.update(kwds)
        if dist is None or dist == 'norm':
            self.dist = stats.norm
            self.dist_args = ()
        elif dist == 't':
            self.dist = stats.t
            self.dist_args = (self.df,)
        else:
            self.dist = dist
            self.dist_args = ()

    @property
    def se(self):
        if False:
            return 10
        return np.sqrt(self.var_pred)

    @property
    def tvalues(self):
        if False:
            print('Hello World!')
        return self.predicted / self.se

    def t_test(self, value=0, alternative='two-sided'):
        if False:
            return 10
        "z- or t-test for hypothesis that mean is equal to value\n\n        Parameters\n        ----------\n        value : array_like\n            value under the null hypothesis\n        alternative : str\n            'two-sided', 'larger', 'smaller'\n\n        Returns\n        -------\n        stat : ndarray\n            test statistic\n        pvalue : ndarray\n            p-value of the hypothesis test, the distribution is given by\n            the attribute of the instance, specified in `__init__`. Default\n            if not specified is the normal distribution.\n\n        "
        stat = (self.predicted - value) / self.se
        if alternative in ['two-sided', '2-sided', '2s']:
            pvalue = self.dist.sf(np.abs(stat), *self.dist_args) * 2
        elif alternative in ['larger', 'l']:
            pvalue = self.dist.sf(stat, *self.dist_args)
        elif alternative in ['smaller', 's']:
            pvalue = self.dist.cdf(stat, *self.dist_args)
        else:
            raise ValueError('invalid alternative')
        return (stat, pvalue)

    def _conf_int_generic(self, center, se, alpha, dist_args=None):
        if False:
            for i in range(10):
                print('nop')
        'internal function to avoid code duplication\n        '
        if dist_args is None:
            dist_args = ()
        q = self.dist.ppf(1 - alpha / 2.0, *dist_args)
        lower = center - q * se
        upper = center + q * se
        ci = np.column_stack((lower, upper))
        return ci

    def conf_int(self, *, alpha=0.05, **kwds):
        if False:
            i = 10
            return i + 15
        'Confidence interval for the predicted value.\n\n        Parameters\n        ----------\n        alpha : float, optional\n            The significance level for the confidence interval.\n            ie., The default `alpha` = .05 returns a 95% confidence interval.\n\n        kwds : extra keyword arguments\n            Ignored in base class, only for compatibility, consistent signature\n            with subclasses\n\n        Returns\n        -------\n        ci : ndarray, (k_constraints, 2)\n            The array has the lower and the upper limit of the confidence\n            interval in the columns.\n        '
        ci = self._conf_int_generic(self.predicted, self.se, alpha, dist_args=self.dist_args)
        return ci

    def summary_frame(self, alpha=0.05):
        if False:
            for i in range(10):
                print('nop')
        "Summary frame\n\n        Parameters\n        ----------\n        alpha : float, optional\n            The significance level for the confidence interval.\n            ie., The default `alpha` = .05 returns a 95% confidence interval.\n\n        Returns\n        -------\n        pandas DataFrame with columns 'predicted', 'se', 'ci_lower', 'ci_upper'\n        "
        ci = self.conf_int(alpha=alpha)
        to_include = {}
        to_include['predicted'] = self.predicted
        to_include['se'] = self.se
        to_include['ci_lower'] = ci[:, 0]
        to_include['ci_upper'] = ci[:, 1]
        self.table = to_include
        res = pd.DataFrame(to_include, index=self.row_labels, columns=to_include.keys())
        return res

class PredictionResultsMonotonic(PredictionResultsBase):

    def __init__(self, predicted, var_pred, linpred=None, linpred_se=None, func=None, deriv=None, df=None, dist=None, row_labels=None):
        if False:
            for i in range(10):
                print('nop')
        self.predicted = predicted
        self.var_pred = var_pred
        self.linpred = linpred
        self.linpred_se = linpred_se
        self.func = func
        self.deriv = deriv
        self.df = df
        self.row_labels = row_labels
        if dist is None or dist == 'norm':
            self.dist = stats.norm
            self.dist_args = ()
        elif dist == 't':
            self.dist = stats.t
            self.dist_args = (self.df,)
        else:
            self.dist = dist
            self.dist_args = ()

    def _conf_int_generic(self, center, se, alpha, dist_args=None):
        if False:
            i = 10
            return i + 15
        'internal function to avoid code duplication\n        '
        if dist_args is None:
            dist_args = ()
        q = self.dist.ppf(1 - alpha / 2.0, *dist_args)
        lower = center - q * se
        upper = center + q * se
        ci = np.column_stack((lower, upper))
        return ci

    def conf_int(self, method='endpoint', alpha=0.05, **kwds):
        if False:
            for i in range(10):
                print('nop')
        'Confidence interval for the predicted value.\n\n        This is currently only available for t and z tests.\n\n        Parameters\n        ----------\n        method : {"endpoint", "delta"}\n            Method for confidence interval, "m\n            If method is "endpoint", then the confidence interval of the\n            linear predictor is transformed by the prediction function.\n            If method is "delta", then the delta-method is used. The confidence\n            interval in this case might reach outside the range of the\n            prediction, for example probabilities larger than one or smaller\n            than zero.\n        alpha : float, optional\n            The significance level for the confidence interval.\n            ie., The default `alpha` = .05 returns a 95% confidence interval.\n        kwds : extra keyword arguments\n            currently ignored, only for compatibility, consistent signature\n\n        Returns\n        -------\n        ci : ndarray, (k_constraints, 2)\n            The array has the lower and the upper limit of the confidence\n            interval in the columns.\n        '
        tmp = np.linspace(0, 1, 6)
        is_linear = (self.func(tmp) == tmp).all()
        if method == 'endpoint' and (not is_linear):
            ci_linear = self._conf_int_generic(self.linpred, self.linpred_se, alpha, dist_args=self.dist_args)
            ci = self.func(ci_linear)
        elif method == 'delta' or is_linear:
            ci = self._conf_int_generic(self.predicted, self.se, alpha, dist_args=self.dist_args)
        return ci

class PredictionResultsDelta(PredictionResultsBase):
    """Prediction results based on delta method
    """

    def __init__(self, results_delta, **kwds):
        if False:
            return 10
        predicted = results_delta.predicted()
        var_pred = results_delta.var()
        super().__init__(predicted, var_pred, **kwds)

class PredictionResultsMean(PredictionResultsBase):
    """Prediction results for GLM.

    This results class is used for backwards compatibility for
    `get_prediction` with GLM. The new PredictionResults classes dropped the
    `_mean` post fix in the attribute names.
    """

    def __init__(self, predicted_mean, var_pred_mean, var_resid=None, df=None, dist=None, row_labels=None, linpred=None, link=None):
        if False:
            for i in range(10):
                print('nop')
        self.predicted = predicted_mean
        self.var_pred = var_pred_mean
        self.df = df
        self.var_resid = var_resid
        self.row_labels = row_labels
        self.linpred = linpred
        self.link = link
        if dist is None or dist == 'norm':
            self.dist = stats.norm
            self.dist_args = ()
        elif dist == 't':
            self.dist = stats.t
            self.dist_args = (self.df,)
        else:
            self.dist = dist
            self.dist_args = ()

    @property
    def predicted_mean(self):
        if False:
            print('Hello World!')
        return self.predicted

    @property
    def var_pred_mean(self):
        if False:
            i = 10
            return i + 15
        return self.var_pred

    @property
    def se_mean(self):
        if False:
            for i in range(10):
                print('nop')
        return self.se

    def conf_int(self, method='endpoint', alpha=0.05, **kwds):
        if False:
            for i in range(10):
                print('nop')
        'Confidence interval for the predicted value.\n\n        This is currently only available for t and z tests.\n\n        Parameters\n        ----------\n        method : {"endpoint", "delta"}\n            Method for confidence interval, "m\n            If method is "endpoint", then the confidence interval of the\n            linear predictor is transformed by the prediction function.\n            If method is "delta", then the delta-method is used. The confidence\n            interval in this case might reach outside the range of the\n            prediction, for example probabilities larger than one or smaller\n            than zero.\n        alpha : float, optional\n            The significance level for the confidence interval.\n            ie., The default `alpha` = .05 returns a 95% confidence interval.\n        kwds : extra keyword arguments\n            currently ignored, only for compatibility, consistent signature\n\n        Returns\n        -------\n        ci : ndarray, (k_constraints, 2)\n            The array has the lower and the upper limit of the confidence\n            interval in the columns.\n        '
        tmp = np.linspace(0, 1, 6)
        is_linear = (self.link.inverse(tmp) == tmp).all()
        if method == 'endpoint' and (not is_linear):
            ci_linear = self.linpred.conf_int(alpha=alpha, obs=False)
            ci = self.link.inverse(ci_linear)
        elif method == 'delta' or is_linear:
            se = self.se_mean
            q = self.dist.ppf(1 - alpha / 2.0, *self.dist_args)
            lower = self.predicted_mean - q * se
            upper = self.predicted_mean + q * se
            ci = np.column_stack((lower, upper))
        return ci

    def summary_frame(self, alpha=0.05):
        if False:
            i = 10
            return i + 15
        "Summary frame\n\n        Parameters\n        ----------\n        alpha : float, optional\n            The significance level for the confidence interval.\n            ie., The default `alpha` = .05 returns a 95% confidence interval.\n\n        Returns\n        -------\n        pandas DataFrame with columns\n        'mean', 'mean_se', 'mean_ci_lower', 'mean_ci_upper'.\n        "
        ci_mean = self.conf_int(alpha=alpha)
        to_include = {}
        to_include['mean'] = self.predicted_mean
        to_include['mean_se'] = self.se_mean
        to_include['mean_ci_lower'] = ci_mean[:, 0]
        to_include['mean_ci_upper'] = ci_mean[:, 1]
        self.table = to_include
        res = pd.DataFrame(to_include, index=self.row_labels, columns=to_include.keys())
        return res

def _get_exog_predict(self, exog=None, transform=True, row_labels=None):
    if False:
        while True:
            i = 10
    "Prepare or transform exog for prediction\n\n    Parameters\n    ----------\n    exog : array_like, optional\n        The values for which you want to predict.\n    transform : bool, optional\n        If the model was fit via a formula, do you want to pass\n        exog through the formula. Default is True. E.g., if you fit\n        a model y ~ log(x1) + log(x2), and transform is True, then\n        you can pass a data structure that contains x1 and x2 in\n        their original form. Otherwise, you'd need to log the data\n        first.\n    row_labels : list of str or None\n        If row_lables are provided, then they will replace the generated\n        labels.\n\n    Returns\n    -------\n    exog : ndarray\n        Prediction exog\n    row_labels : list of str\n        Labels or pandas index for rows of prediction\n    "
    if transform and hasattr(self.model, 'formula') and (exog is not None):
        from patsy import dmatrix
        if isinstance(exog, pd.Series):
            exog = pd.DataFrame(exog)
        exog = dmatrix(self.model.data.design_info, exog)
    if exog is not None:
        if row_labels is None:
            row_labels = getattr(exog, 'index', None)
            if callable(row_labels):
                row_labels = None
        exog = np.asarray(exog)
        if exog.ndim == 1 and (self.model.exog.ndim == 1 or self.model.exog.shape[1] == 1):
            exog = exog[:, None]
        exog = np.atleast_2d(exog)
    else:
        exog = self.model.exog
        if row_labels is None:
            row_labels = getattr(self.model.data, 'row_labels', None)
    return (exog, row_labels)

def get_prediction_glm(self, exog=None, transform=True, row_labels=None, linpred=None, link=None, pred_kwds=None):
    if False:
        print('Hello World!')
    "\n    Compute prediction results for GLM compatible models.\n\n    Parameters\n    ----------\n    exog : array_like, optional\n        The values for which you want to predict.\n    transform : bool, optional\n        If the model was fit via a formula, do you want to pass\n        exog through the formula. Default is True. E.g., if you fit\n        a model y ~ log(x1) + log(x2), and transform is True, then\n        you can pass a data structure that contains x1 and x2 in\n        their original form. Otherwise, you'd need to log the data\n        first.\n    row_labels : list of str or None\n        If row_lables are provided, then they will replace the generated\n        labels.\n    linpred : linear prediction instance\n        Instance of linear prediction results used for confidence intervals\n        based on endpoint transformation.\n    link : instance of link function\n        If no link function is provided, then the `model.family.link` is used.\n    pred_kwds : dict\n        Some models can take additional keyword arguments, such as offset or\n        additional exog in multi-part models. See the predict method of the\n        model for the details.\n\n    Returns\n    -------\n    prediction_results : generalized_linear_model.PredictionResults\n        The prediction results instance contains prediction and prediction\n        variance and can on demand calculate confidence intervals and summary\n        tables for the prediction of the mean and of new observations.\n    "
    (exog, row_labels) = _get_exog_predict(self, exog=exog, transform=transform, row_labels=row_labels)
    if pred_kwds is None:
        pred_kwds = {}
    predicted_mean = self.model.predict(self.params, exog, **pred_kwds)
    covb = self.cov_params()
    link_deriv = self.model.family.link.inverse_deriv(linpred.predicted_mean)
    var_pred_mean = link_deriv ** 2 * (exog * np.dot(covb, exog.T).T).sum(1)
    var_resid = self.scale
    if self.cov_type == 'fixed scale':
        var_resid = self.cov_kwds['scale']
    dist = ['norm', 't'][self.use_t]
    return PredictionResultsMean(predicted_mean, var_pred_mean, var_resid, df=self.df_resid, dist=dist, row_labels=row_labels, linpred=linpred, link=link)

def get_prediction_linear(self, exog=None, transform=True, row_labels=None, pred_kwds=None, index=None):
    if False:
        return 10
    "\n    Compute prediction results for linear prediction.\n\n    Parameters\n    ----------\n    exog : array_like, optional\n        The values for which you want to predict.\n    transform : bool, optional\n        If the model was fit via a formula, do you want to pass\n        exog through the formula. Default is True. E.g., if you fit\n        a model y ~ log(x1) + log(x2), and transform is True, then\n        you can pass a data structure that contains x1 and x2 in\n        their original form. Otherwise, you'd need to log the data\n        first.\n    row_labels : list of str or None\n        If row_lables are provided, then they will replace the generated\n        labels.\n    pred_kwargs :\n        Some models can take additional keyword arguments, such as offset or\n        additional exog in multi-part models.\n        See the predict method of the model for the details.\n    index : slice or array-index\n        Is used to select rows and columns of cov_params, if the prediction\n        function only depends on a subset of parameters.\n\n    Returns\n    -------\n    prediction_results : PredictionResults\n        The prediction results instance contains prediction and prediction\n        variance and can on demand calculate confidence intervals and summary\n        tables for the prediction.\n    "
    (exog, row_labels) = _get_exog_predict(self, exog=exog, transform=transform, row_labels=row_labels)
    if pred_kwds is None:
        pred_kwds = {}
    k1 = exog.shape[1]
    if len(self.params > k1):
        index = np.arange(k1)
    else:
        index = None
    covb = self.cov_params(column=index)
    var_pred = (exog * np.dot(covb, exog.T).T).sum(1)
    pred_kwds_linear = pred_kwds.copy()
    pred_kwds_linear['which'] = 'linear'
    predicted = self.model.predict(self.params, exog, **pred_kwds_linear)
    dist = ['norm', 't'][self.use_t]
    res = PredictionResultsBase(predicted, var_pred, df=self.df_resid, dist=dist, row_labels=row_labels)
    return res

def get_prediction_monotonic(self, exog=None, transform=True, row_labels=None, link=None, pred_kwds=None, index=None):
    if False:
        while True:
            i = 10
    "\n    Compute prediction results when endpoint transformation is valid.\n\n    Parameters\n    ----------\n    exog : array_like, optional\n        The values for which you want to predict.\n    transform : bool, optional\n        If the model was fit via a formula, do you want to pass\n        exog through the formula. Default is True. E.g., if you fit\n        a model y ~ log(x1) + log(x2), and transform is True, then\n        you can pass a data structure that contains x1 and x2 in\n        their original form. Otherwise, you'd need to log the data\n        first.\n    row_labels : list of str or None\n        If row_lables are provided, then they will replace the generated\n        labels.\n    link : instance of link function\n        If no link function is provided, then the ``mmodel.family.link` is\n        used.\n    pred_kwargs :\n        Some models can take additional keyword arguments, such as offset or\n        additional exog in multi-part models.\n        See the predict method of the model for the details.\n    index : slice or array-index\n        Is used to select rows and columns of cov_params, if the prediction\n        function only depends on a subset of parameters.\n\n    Returns\n    -------\n    prediction_results : PredictionResults\n        The prediction results instance contains prediction and prediction\n        variance and can on demand calculate confidence intervals and summary\n        tables for the prediction.\n    "
    (exog, row_labels) = _get_exog_predict(self, exog=exog, transform=transform, row_labels=row_labels)
    if pred_kwds is None:
        pred_kwds = {}
    if link is None:
        link = self.model.family.link
    func_deriv = link.inverse_deriv
    covb = self.cov_params(column=index)
    linpred_var = (exog * np.dot(covb, exog.T).T).sum(1)
    pred_kwds_linear = pred_kwds.copy()
    pred_kwds_linear['which'] = 'linear'
    linpred = self.model.predict(self.params, exog, **pred_kwds_linear)
    predicted = self.model.predict(self.params, exog, **pred_kwds)
    link_deriv = func_deriv(linpred)
    var_pred = link_deriv ** 2 * linpred_var
    dist = ['norm', 't'][self.use_t]
    res = PredictionResultsMonotonic(predicted, var_pred, df=self.df_resid, dist=dist, row_labels=row_labels, linpred=linpred, linpred_se=np.sqrt(linpred_var), func=link.inverse, deriv=func_deriv)
    return res

def get_prediction_delta(self, exog=None, which='mean', average=False, agg_weights=None, transform=True, row_labels=None, pred_kwds=None):
    if False:
        while True:
            i = 10
    "\n    compute prediction results\n\n    Parameters\n    ----------\n    exog : array_like, optional\n        The values for which you want to predict.\n    which : str\n        The statistic that is prediction. Which statistics are available\n        depends on the model.predict method.\n    average : bool\n        If average is True, then the mean prediction is computed, that is,\n        predictions are computed for individual exog and then them mean over\n        observation is used.\n        If average is False, then the results are the predictions for all\n        observations, i.e. same length as ``exog``.\n    agg_weights : ndarray, optional\n        Aggregation weights, only used if average is True.\n        The weights are not normalized.\n    transform : bool, optional\n        If the model was fit via a formula, do you want to pass\n        exog through the formula. Default is True. E.g., if you fit\n        a model y ~ log(x1) + log(x2), and transform is True, then\n        you can pass a data structure that contains x1 and x2 in\n        their original form. Otherwise, you'd need to log the data\n        first.\n    row_labels : list of str or None\n        If row_lables are provided, then they will replace the generated\n        labels.\n    pred_kwargs :\n        Some models can take additional keyword arguments, such as offset or\n        additional exog in multi-part models.\n        See the predict method of the model for the details.\n\n    Returns\n    -------\n    prediction_results : generalized_linear_model.PredictionResults\n        The prediction results instance contains prediction and prediction\n        variance and can on demand calculate confidence intervals and summary\n        tables for the prediction of the mean and of new observations.\n    "
    (exog, row_labels) = _get_exog_predict(self, exog=exog, transform=transform, row_labels=row_labels)
    if agg_weights is None:
        agg_weights = np.array(1.0)

    def f_pred(p):
        if False:
            return 10
        'Prediction function as function of params\n        '
        pred = self.model.predict(p, exog, which=which, **pred_kwds)
        if average:
            pred = (pred.T * agg_weights.T).mean(-1).T
        return pred
    nlpm = self._get_wald_nonlinear(f_pred)
    res = PredictionResultsDelta(nlpm)
    return res

def get_prediction(self, exog=None, transform=True, which='mean', row_labels=None, average=False, agg_weights=None, pred_kwds=None):
    if False:
        while True:
            i = 10
    '\n    Compute prediction results when endpoint transformation is valid.\n\n    Parameters\n    ----------\n    exog : array_like, optional\n        The values for which you want to predict.\n    transform : bool, optional\n        If the model was fit via a formula, do you want to pass\n        exog through the formula. Default is True. E.g., if you fit\n        a model y ~ log(x1) + log(x2), and transform is True, then\n        you can pass a data structure that contains x1 and x2 in\n        their original form. Otherwise, you\'d need to log the data\n        first.\n    which : str\n        Which statistic is to be predicted. Default is "mean".\n        The available statistics and options depend on the model.\n        see the model.predict docstring\n    linear : bool\n        Linear has been replaced by the `which` keyword and will be\n        deprecated.\n        If linear is True, then `which` is ignored and the linear\n        prediction is returned.\n    row_labels : list of str or None\n        If row_lables are provided, then they will replace the generated\n        labels.\n    average : bool\n        If average is True, then the mean prediction is computed, that is,\n        predictions are computed for individual exog and then the average\n        over observation is used.\n        If average is False, then the results are the predictions for all\n        observations, i.e. same length as ``exog``.\n    agg_weights : ndarray, optional\n        Aggregation weights, only used if average is True.\n        The weights are not normalized.\n    **kwargs :\n        Some models can take additional keyword arguments, such as offset,\n        exposure or additional exog in multi-part models like zero inflated\n        models.\n        See the predict method of the model for the details.\n\n    Returns\n    -------\n    prediction_results : PredictionResults\n        The prediction results instance contains prediction and prediction\n        variance and can on demand calculate confidence intervals and\n        summary dataframe for the prediction.\n\n    Notes\n    -----\n    Status: new in 0.14, experimental\n    '
    use_endpoint = getattr(self.model, '_use_endpoint', True)
    if which == 'linear':
        res = get_prediction_linear(self, exog=exog, transform=transform, row_labels=row_labels, pred_kwds=pred_kwds)
    elif which == 'mean' and use_endpoint is True and (average is False):
        k1 = self.model.exog.shape[1]
        if len(self.params > k1):
            index = np.arange(k1)
        else:
            index = None
        pred_kwds['which'] = which
        link = getattr(self.model, 'link', None)
        if link is None:
            if hasattr(self.model, 'family'):
                link = getattr(self.model.family, 'link', None)
        if link is None:
            import warnings
            warnings.warn('using default log-link in get_prediction')
            from statsmodels.genmod.families import links
            link = links.Log()
        res = get_prediction_monotonic(self, exog=exog, transform=transform, row_labels=row_labels, link=link, pred_kwds=pred_kwds, index=index)
    else:
        res = get_prediction_delta(self, exog=exog, which=which, average=average, agg_weights=agg_weights, pred_kwds=pred_kwds)
    return res

def params_transform_univariate(params, cov_params, link=None, transform=None, row_labels=None):
    if False:
        return 10
    '\n    results for univariate, nonlinear, monotonicaly transformed parameters\n\n    This provides transformed values, standard errors and confidence interval\n    for transformations of parameters, for example in calculating rates with\n    `exp(params)` in the case of Poisson or other models with exponential\n    mean function.\n    '
    from statsmodels.genmod.families import links
    if link is None and transform is None:
        link = links.Log()
    if row_labels is None and hasattr(params, 'index'):
        row_labels = params.index
    params = np.asarray(params)
    predicted_mean = link.inverse(params)
    link_deriv = link.inverse_deriv(params)
    var_pred_mean = link_deriv ** 2 * np.diag(cov_params)
    dist = stats.norm
    linpred = PredictionResultsMean(params, np.diag(cov_params), dist=dist, row_labels=row_labels, link=links.Identity())
    res = PredictionResultsMean(predicted_mean, var_pred_mean, dist=dist, row_labels=row_labels, linpred=linpred, link=link)
    return res