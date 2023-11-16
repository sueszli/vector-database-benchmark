"""
A predict-like function that constructs means and pointwise or
simultaneous confidence bands for the function f(x) = E[Y | X*=x,
X1=x1, ...], where X* is the focus variable and X1, X2, ... are
non-focus variables.  This is especially useful when conducting a
functional regression in which the role of x is modeled with b-splines
or other basis functions.
"""
import pandas as pd
import patsy
import numpy as np
import warnings
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.compat.pandas import Appender
_predict_functional_doc = "\n    Predictions and contrasts of a fitted model as a function of a given covariate.\n\n    The value of the focus variable varies along a sequence of its\n    quantiles, calculated from the data used to fit the model.  The\n    other variables are held constant either at given values, or at\n    values obtained by applying given summary functions to the data\n    used to fit the model.  Optionally, a second specification of the\n    non-focus variables is provided and the contrast between the two\n    specifications is returned.\n\n    Parameters\n    ----------\n    result : statsmodels result object\n        A results object for the fitted model.\n    focus_var : str\n        The name of the 'focus variable'.\n    summaries : dict-like\n        A map from names of non-focus variables to summary functions.\n        Each summary function is applied to the data used to fit the\n        model, to obtain a value at which the variable is held fixed.\n    values : dict-like\n        Values at which a given non-focus variable is held fixed.\n    summaries2 : dict-like\n        A second set of summary functions used to define a contrast.\n    values2 : dict-like\n        A second set of fixed values used to define a contrast.\n    alpha : float\n        `1 - alpha` is the coverage probability.\n    ci_method : str\n        The method for constructing the confidence band, one of\n        'pointwise', 'scheffe', and 'simultaneous'.\n    num_points : int\n        The number of equally-spaced quantile points where the\n        prediction is made.\n    exog : array_like\n        Explicitly provide points to cover with the confidence band.\n    exog2 : array_like\n        Explicitly provide points to contrast to `exog` in a functional\n        confidence band.\n    kwargs :\n        Arguments passed to the `predict` method.\n\n    Returns\n    -------\n    pred : array_like\n        The predicted mean values.\n    cb : array_like\n        An array with two columns, containing respectively the lower\n        and upper limits of a confidence band.\n    fvals : array_like\n        The values of the focus variable at which the prediction is\n        made.\n\n    Notes\n    -----\n    All variables in the model except for the focus variable should be\n    included as a key in either `summaries` or `values` (unless `exog`\n    is provided).\n\n    If `summaries2` and `values2` are not provided, the returned value\n    contains predicted conditional means for the outcome as the focus\n    variable varies, with the other variables fixed as specified.\n\n    If `summaries2` and/or `values2` is provided, two sets of\n    predicted conditional means are calculated, and the returned value\n    is the contrast between them.\n\n    If `exog` is provided, then the rows should contain a sequence of\n    values approximating a continuous path through the domain of the\n    covariates.  For example, if Z(s) is the covariate expressed as a\n    function of s, then the rows of exog may approximate Z(g(s)) for\n    some continuous function g.  If `exog` is provided then neither of\n    the summaries or values arguments should be provided.  If `exog2`\n    is also provided, then the returned value is a contrast between\n    the functionas defined by `exog` and `exog2`.\n\n    Examples\n    --------\n    Fit a model using a formula in which the predictors are age\n    (modeled with splines), ethnicity (which is categorical), gender,\n    and income.  Then we obtain the fitted mean values as a function\n    of age for females with mean income and the most common\n    ethnicity.\n\n    >>> model = sm.OLS.from_formula('y ~ bs(age, df=4) + C(ethnicity) + gender + income', data)\n    >>> result = model.fit()\n    >>> mode = lambda x : x.value_counts().argmax()\n    >>> summaries = {'income': np.mean, ethnicity=mode}\n    >>> values = {'gender': 'female'}\n    >>> pr, cb, x = predict_functional(result, 'age', summaries, values)\n\n    Fit a model using arrays.  Plot the means as a function of x3,\n    holding x1 fixed at its mean value in the data used to fit the\n    model, and holding x2 fixed at 1.\n\n    >>> model = sm.OLS(y ,x)\n    >>> result = model.fit()\n    >>> summaries = {'x1': np.mean}\n    >>> values = {'x2': 1}\n    >>> pr, cb, x = predict_functional(result, 'x3', summaries, values)\n\n    Fit a model usng a formula and construct a contrast comparing the\n    female and male predicted mean functions.\n\n    >>> model = sm.OLS.from_formula('y ~ bs(age, df=4) + gender', data)\n    >>> result = model.fit()\n    >>> values = {'gender': 'female'}\n    >>> values2 = {'gender': 'male'}\n    >>> pr, cb, x = predict_functional(result, 'age', values=values, values2=values2)\n    "

def _make_exog_from_formula(result, focus_var, summaries, values, num_points):
    if False:
        i = 10
        return i + 15
    '\n    Create dataframes for exploring a fitted model as a function of one variable.\n\n    This works for models fit with a formula.\n\n    Returns\n    -------\n    dexog : data frame\n        A data frame in which the focus variable varies and the other variables\n        are fixed at specified or computed values.\n    fexog : data frame\n        The data frame `dexog` processed through the model formula.\n    '
    model = result.model
    exog = model.data.frame
    if summaries is None:
        summaries = {}
    if values is None:
        values = {}
    if exog[focus_var].dtype is np.dtype('O'):
        raise ValueError('focus variable may not have object type')
    colnames = list(summaries.keys()) + list(values.keys()) + [focus_var]
    dtypes = [exog[x].dtype for x in colnames]
    varl = set(exog.columns.tolist()) - set([model.endog_names])
    unmatched = varl - set(colnames)
    unmatched = list(unmatched)
    if len(unmatched) > 0:
        warnings.warn('%s in data frame but not in summaries or values.' % ', '.join(["'%s'" % x for x in unmatched]), ValueWarning)
    ix = range(num_points)
    fexog = pd.DataFrame(index=ix, columns=colnames)
    for (d, x) in zip(dtypes, colnames):
        fexog[x] = pd.Series(index=ix, dtype=d)
    pctls = np.linspace(0, 100, num_points).tolist()
    fvals = np.percentile(exog[focus_var], pctls)
    fvals = np.asarray(fvals)
    fexog.loc[:, focus_var] = fvals
    for ky in summaries.keys():
        fexog.loc[:, ky] = summaries[ky](exog.loc[:, ky])
    for ky in values.keys():
        fexog[ky] = values[ky]
    dexog = patsy.dmatrix(model.data.design_info, fexog, return_type='dataframe')
    return (dexog, fexog, fvals)

def _make_exog_from_arrays(result, focus_var, summaries, values, num_points):
    if False:
        while True:
            i = 10
    '\n    Create dataframes for exploring a fitted model as a function of one variable.\n\n    This works for models fit without a formula.\n\n    Returns\n    -------\n    exog : data frame\n        A data frame in which the focus variable varies and the other variables\n        are fixed at specified or computed values.\n    '
    model = result.model
    model_exog = model.exog
    exog_names = model.exog_names
    if summaries is None:
        summaries = {}
    if values is None:
        values = {}
    exog = np.zeros((num_points, model_exog.shape[1]))
    colnames = list(values.keys()) + list(summaries.keys()) + [focus_var]
    unmatched = set(exog_names) - set(colnames)
    unmatched = list(unmatched)
    if len(unmatched) > 0:
        warnings.warn('%s in model but not in `summaries` or `values`.' % ', '.join(["'%s'" % x for x in unmatched]), ValueWarning)
    pctls = np.linspace(0, 100, num_points).tolist()
    ix = exog_names.index(focus_var)
    fvals = np.percentile(model_exog[:, ix], pctls)
    exog[:, ix] = fvals
    for ky in summaries.keys():
        ix = exog_names.index(ky)
        exog[:, ix] = summaries[ky](model_exog[:, ix])
    for ky in values.keys():
        ix = exog_names.index(ky)
        exog[:, ix] = values[ky]
    return (exog, fvals)

def _make_exog(result, focus_var, summaries, values, num_points):
    if False:
        return 10
    if hasattr(result.model.data, 'frame'):
        (dexog, fexog, fvals) = _make_exog_from_formula(result, focus_var, summaries, values, num_points)
    else:
        (exog, fvals) = _make_exog_from_arrays(result, focus_var, summaries, values, num_points)
        (dexog, fexog) = (exog, exog)
    return (dexog, fexog, fvals)

def _check_args(values, summaries, values2, summaries2):
    if False:
        return 10
    if values is None:
        values = {}
    if values2 is None:
        values2 = {}
    if summaries is None:
        summaries = {}
    if summaries2 is None:
        summaries2 = {}
    for (s, v) in ((summaries, values), (summaries2, values2)):
        ky = set(v.keys()) & set(s.keys())
        ky = list(ky)
        if len(ky) > 0:
            raise ValueError('One or more variable names are contained in both `summaries` and `values`:' + ', '.join(ky))
    return (values, summaries, values2, summaries2)

@Appender(_predict_functional_doc)
def predict_functional(result, focus_var, summaries=None, values=None, summaries2=None, values2=None, alpha=0.05, ci_method='pointwise', linear=True, num_points=10, exog=None, exog2=None, **kwargs):
    if False:
        return 10
    if ci_method not in ('pointwise', 'scheffe', 'simultaneous'):
        raise ValueError('confidence band method must be one of `pointwise`, `scheffe`, and `simultaneous`.')
    contrast = values2 is not None or summaries2 is not None
    if contrast and (not linear):
        raise ValueError('`linear` must be True for computing contrasts')
    model = result.model
    if exog is not None:
        if any((x is not None for x in [summaries, summaries2, values, values2])):
            raise ValueError('if `exog` is provided then do not provide `summaries` or `values`')
        fexog = exog
        dexog = patsy.dmatrix(model.data.design_info, fexog, return_type='dataframe')
        fvals = exog[focus_var]
        if exog2 is not None:
            fexog2 = exog
            dexog2 = patsy.dmatrix(model.data.design_info, fexog2, return_type='dataframe')
            fvals2 = fvals
    else:
        (values, summaries, values2, summaries2) = _check_args(values, summaries, values2, summaries2)
        (dexog, fexog, fvals) = _make_exog(result, focus_var, summaries, values, num_points)
        if len(summaries2) + len(values2) > 0:
            (dexog2, fexog2, fvals2) = _make_exog(result, focus_var, summaries2, values2, num_points)
    from statsmodels.genmod.generalized_linear_model import GLM
    from statsmodels.genmod.generalized_estimating_equations import GEE
    if isinstance(result.model, (GLM, GEE)):
        kwargs_pred = kwargs.copy()
        kwargs_pred.update({'which': 'linear'})
    else:
        kwargs_pred = kwargs
    pred = result.predict(exog=fexog, **kwargs_pred)
    if contrast:
        pred2 = result.predict(exog=fexog2, **kwargs_pred)
        pred = pred - pred2
        dexog = dexog - dexog2
    if ci_method == 'pointwise':
        t_test = result.t_test(dexog)
        cb = t_test.conf_int(alpha=alpha)
    elif ci_method == 'scheffe':
        t_test = result.t_test(dexog)
        sd = t_test.sd
        cb = np.zeros((num_points, 2))
        from scipy.stats.distributions import f as fdist
        df1 = result.model.exog.shape[1]
        df2 = result.model.exog.shape[0] - df1
        qf = fdist.cdf(1 - alpha, df1, df2)
        fx = sd * np.sqrt(df1 * qf)
        cb[:, 0] = pred - fx
        cb[:, 1] = pred + fx
    elif ci_method == 'simultaneous':
        (sigma, c) = _glm_basic_scr(result, dexog, alpha)
        cb = np.zeros((dexog.shape[0], 2))
        cb[:, 0] = pred - c * sigma
        cb[:, 1] = pred + c * sigma
    if not linear:
        link = result.family.link
        pred = link.inverse(pred)
        cb = link.inverse(cb)
    return (pred, cb, fvals)

def _glm_basic_scr(result, exog, alpha):
    if False:
        return 10
    "\n    The basic SCR from (Sun et al. Annals of Statistics 2000).\n\n    Computes simultaneous confidence regions (SCR).\n\n    Parameters\n    ----------\n    result : results instance\n        The fitted GLM results instance\n    exog : array_like\n        The exog values spanning the interval\n    alpha : float\n        `1 - alpha` is the coverage probability.\n\n    Returns\n    -------\n    An array with two columns, containing the lower and upper\n    confidence bounds, respectively.\n\n    Notes\n    -----\n    The rows of `exog` should be a sequence of covariate values\n    obtained by taking one 'free variable' x and varying it over an\n    interval.  The matrix `exog` is thus the basis functions and any\n    other covariates evaluated as x varies.\n    "
    model = result.model
    n = model.exog.shape[0]
    cov = result.cov_params()
    hess = np.linalg.inv(cov)
    A = hess / n
    B = np.linalg.cholesky(A).T
    sigma2 = (np.dot(exog, cov) * exog).sum(1)
    sigma = np.asarray(np.sqrt(sigma2))
    bz = np.linalg.solve(B.T, exog.T).T
    bz /= np.sqrt(n)
    bz /= sigma[:, None]
    bzd = np.diff(bz, 1, axis=0)
    bzdn = (bzd ** 2).sum(1)
    kappa_0 = np.sqrt(bzdn).sum()
    from scipy.stats.distributions import norm

    def func(c):
        if False:
            i = 10
            return i + 15
        return kappa_0 * np.exp(-c ** 2 / 2) / np.pi + 2 * (1 - norm.cdf(c)) - alpha
    from scipy.optimize import brentq
    (c, rslt) = brentq(func, 1, 10, full_output=True)
    if not rslt.converged:
        raise ValueError('Root finding error in basic SCR')
    return (sigma, c)