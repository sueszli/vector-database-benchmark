import numpy as np
from scipy import stats
import pandas as pd
from pandas import DataFrame, Index
import patsy
from statsmodels.regression.linear_model import OLS
from statsmodels.compat.python import lrange
from statsmodels.formula.formulatools import _remove_intercept_patsy, _has_intercept, _intercept_idx
from statsmodels.iolib import summary2

def _get_covariance(model, robust):
    if False:
        print('Hello World!')
    if robust is None:
        return model.cov_params()
    elif robust == 'hc0':
        return model.cov_HC0
    elif robust == 'hc1':
        return model.cov_HC1
    elif robust == 'hc2':
        return model.cov_HC2
    elif robust == 'hc3':
        return model.cov_HC3
    else:
        raise ValueError('robust options %s not understood' % robust)

def anova_single(model, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Anova table for one fitted linear model.\n\n    Parameters\n    ----------\n    model : fitted linear model results instance\n        A fitted linear model\n    typ : int or str {1,2,3} or {"I","II","III"}\n        Type of sum of squares to use.\n\n    **kwargs**\n\n    scale : float\n        Estimate of variance, If None, will be estimated from the largest\n    model. Default is None.\n        test : str {"F", "Chisq", "Cp"} or None\n        Test statistics to provide. Default is "F".\n\n    Notes\n    -----\n    Use of this function is discouraged. Use anova_lm instead.\n    '
    test = kwargs.get('test', 'F')
    scale = kwargs.get('scale', None)
    typ = kwargs.get('typ', 1)
    robust = kwargs.get('robust', None)
    if robust:
        robust = robust.lower()
    endog = model.model.endog
    exog = model.model.exog
    nobs = exog.shape[0]
    response_name = model.model.endog_names
    design_info = model.model.data.design_info
    exog_names = model.model.exog_names
    n_rows = len(design_info.terms) - _has_intercept(design_info) + 1
    pr_test = 'PR(>%s)' % test
    names = ['df', 'sum_sq', 'mean_sq', test, pr_test]
    table = DataFrame(np.zeros((n_rows, 5)), columns=names)
    if typ in [1, 'I']:
        return anova1_lm_single(model, endog, exog, nobs, design_info, table, n_rows, test, pr_test, robust)
    elif typ in [2, 'II']:
        return anova2_lm_single(model, design_info, n_rows, test, pr_test, robust)
    elif typ in [3, 'III']:
        return anova3_lm_single(model, design_info, n_rows, test, pr_test, robust)
    elif typ in [4, 'IV']:
        raise NotImplementedError('Type IV not yet implemented')
    else:
        raise ValueError('Type %s not understood' % str(typ))

def anova1_lm_single(model, endog, exog, nobs, design_info, table, n_rows, test, pr_test, robust):
    if False:
        while True:
            i = 10
    '\n    Anova table for one fitted linear model.\n\n    Parameters\n    ----------\n    model : fitted linear model results instance\n        A fitted linear model\n\n    **kwargs**\n\n    scale : float\n        Estimate of variance, If None, will be estimated from the largest\n    model. Default is None.\n        test : str {"F", "Chisq", "Cp"} or None\n        Test statistics to provide. Default is "F".\n\n    Notes\n    -----\n    Use of this function is discouraged. Use anova_lm instead.\n    '
    effects = getattr(model, 'effects', None)
    if effects is None:
        (q, r) = np.linalg.qr(exog)
        effects = np.dot(q.T, endog)
    arr = np.zeros((len(design_info.terms), len(design_info.column_names)))
    slices = [design_info.slice(name) for name in design_info.term_names]
    for (i, slice_) in enumerate(slices):
        arr[i, slice_] = 1
    sum_sq = np.dot(arr, effects ** 2)
    idx = _intercept_idx(design_info)
    sum_sq = sum_sq[~idx]
    term_names = np.array(design_info.term_names)
    term_names = term_names[~idx]
    index = term_names.tolist()
    table.index = Index(index + ['Residual'])
    table.loc[index, ['df', 'sum_sq']] = np.c_[arr[~idx].sum(1), sum_sq]
    table.loc['Residual', ['sum_sq', 'df']] = (model.ssr, model.df_resid)
    if test == 'F':
        table[test] = table['sum_sq'] / table['df'] / (model.ssr / model.df_resid)
        table[pr_test] = stats.f.sf(table['F'], table['df'], model.df_resid)
        table.loc['Residual', [test, pr_test]] = (np.nan, np.nan)
    table['mean_sq'] = table['sum_sq'] / table['df']
    return table

def anova2_lm_single(model, design_info, n_rows, test, pr_test, robust):
    if False:
        while True:
            i = 10
    '\n    Anova type II table for one fitted linear model.\n\n    Parameters\n    ----------\n    model : fitted linear model results instance\n        A fitted linear model\n\n    **kwargs**\n\n    scale : float\n        Estimate of variance, If None, will be estimated from the largest\n    model. Default is None.\n        test : str {"F", "Chisq", "Cp"} or None\n        Test statistics to provide. Default is "F".\n\n    Notes\n    -----\n    Use of this function is discouraged. Use anova_lm instead.\n\n    Type II\n    Sum of Squares compares marginal contribution of terms. Thus, it is\n    not particularly useful for models with significant interaction terms.\n    '
    terms_info = design_info.terms[:]
    terms_info = _remove_intercept_patsy(terms_info)
    names = ['sum_sq', 'df', test, pr_test]
    table = DataFrame(np.zeros((n_rows, 4)), columns=names)
    cov = _get_covariance(model, None)
    robust_cov = _get_covariance(model, robust)
    col_order = []
    index = []
    for (i, term) in enumerate(terms_info):
        cols = design_info.slice(term)
        L1 = lrange(cols.start, cols.stop)
        L2 = []
        term_set = set(term.factors)
        for t in terms_info:
            other_set = set(t.factors)
            if term_set.issubset(other_set) and (not term_set == other_set):
                col = design_info.slice(t)
                L1.extend(lrange(col.start, col.stop))
                L2.extend(lrange(col.start, col.stop))
        L1 = np.eye(model.model.exog.shape[1])[L1]
        L2 = np.eye(model.model.exog.shape[1])[L2]
        if L2.size:
            LVL = np.dot(np.dot(L1, robust_cov), L2.T)
            from scipy import linalg
            (orth_compl, _) = linalg.qr(LVL)
            r = L1.shape[0] - L2.shape[0]
            L12 = np.dot(orth_compl[:, -r:].T, L1)
        else:
            L12 = L1
            r = L1.shape[0]
        if test == 'F':
            f = model.f_test(L12, cov_p=robust_cov)
            table.loc[table.index[i], test] = test_value = f.fvalue
            table.loc[table.index[i], pr_test] = f.pvalue
        table.loc[table.index[i], 'df'] = r
        col_order.append(cols.start)
        index.append(term.name())
    table.index = Index(index + ['Residual'])
    table = table.iloc[np.argsort(col_order + [model.model.exog.shape[1] + 1])]
    ssr = table[test] * table['df'] * model.ssr / model.df_resid
    table['sum_sq'] = ssr
    table.loc['Residual', ['sum_sq', 'df', test, pr_test]] = (model.ssr, model.df_resid, np.nan, np.nan)
    return table

def anova3_lm_single(model, design_info, n_rows, test, pr_test, robust):
    if False:
        return 10
    n_rows += _has_intercept(design_info)
    terms_info = design_info.terms
    names = ['sum_sq', 'df', test, pr_test]
    table = DataFrame(np.zeros((n_rows, 4)), columns=names)
    cov = _get_covariance(model, robust)
    col_order = []
    index = []
    for (i, term) in enumerate(terms_info):
        cols = design_info.slice(term)
        L1 = np.eye(model.model.exog.shape[1])[cols]
        L12 = L1
        r = L1.shape[0]
        if test == 'F':
            f = model.f_test(L12, cov_p=cov)
            table.loc[table.index[i], test] = test_value = f.fvalue
            table.loc[table.index[i], pr_test] = f.pvalue
        table.loc[table.index[i], 'df'] = r
        index.append(term.name())
    table.index = Index(index + ['Residual'])
    ssr = table[test] * table['df'] * model.ssr / model.df_resid
    table['sum_sq'] = ssr
    table.loc['Residual', ['sum_sq', 'df', test, pr_test]] = (model.ssr, model.df_resid, np.nan, np.nan)
    return table

def anova_lm(*args, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Anova table for one or more fitted linear models.\n\n    Parameters\n    ----------\n    args : fitted linear model results instance\n        One or more fitted linear models\n    scale : float\n        Estimate of variance, If None, will be estimated from the largest\n        model. Default is None.\n    test : str {"F", "Chisq", "Cp"} or None\n        Test statistics to provide. Default is "F".\n    typ : str or int {"I","II","III"} or {1,2,3}\n        The type of Anova test to perform. See notes.\n    robust : {None, "hc0", "hc1", "hc2", "hc3"}\n        Use heteroscedasticity-corrected coefficient covariance matrix.\n        If robust covariance is desired, it is recommended to use `hc3`.\n\n    Returns\n    -------\n    anova : DataFrame\n        When args is a single model, return is DataFrame with columns:\n\n        sum_sq : float64\n            Sum of squares for model terms.\n        df : float64\n            Degrees of freedom for model terms.\n        F : float64\n            F statistic value for significance of adding model terms.\n        PR(>F) : float64\n            P-value for significance of adding model terms.\n\n        When args is multiple models, return is DataFrame with columns:\n\n        df_resid : float64\n            Degrees of freedom of residuals in models.\n        ssr : float64\n            Sum of squares of residuals in models.\n        df_diff : float64\n            Degrees of freedom difference from previous model in args\n        ss_dff : float64\n            Difference in ssr from previous model in args\n        F : float64\n            F statistic comparing to previous model in args\n        PR(>F): float64\n            P-value for significance comparing to previous model in args\n\n    Notes\n    -----\n    Model statistics are given in the order of args. Models must have been fit\n    using the formula api.\n\n    See Also\n    --------\n    model_results.compare_f_test, model_results.compare_lm_test\n\n    Examples\n    --------\n    >>> import statsmodels.api as sm\n    >>> from statsmodels.formula.api import ols\n    >>> moore = sm.datasets.get_rdataset("Moore", "carData", cache=True) # load\n    >>> data = moore.data\n    >>> data = data.rename(columns={"partner.status" :\n    ...                             "partner_status"}) # make name pythonic\n    >>> moore_lm = ols(\'conformity ~ C(fcategory, Sum)*C(partner_status, Sum)\',\n    ...                 data=data).fit()\n    >>> table = sm.stats.anova_lm(moore_lm, typ=2) # Type 2 Anova DataFrame\n    >>> print(table)\n    '
    typ = kwargs.get('typ', 1)
    if len(args) == 1:
        model = args[0]
        return anova_single(model, **kwargs)
    if typ not in [1, 'I']:
        raise ValueError('Multiple models only supported for type I. Got type %s' % str(typ))
    test = kwargs.get('test', 'F')
    scale = kwargs.get('scale', None)
    n_models = len(args)
    pr_test = 'Pr(>%s)' % test
    names = ['df_resid', 'ssr', 'df_diff', 'ss_diff', test, pr_test]
    table = DataFrame(np.zeros((n_models, 6)), columns=names)
    if not scale:
        scale = args[-1].scale
    table['ssr'] = [mdl.ssr for mdl in args]
    table['df_resid'] = [mdl.df_resid for mdl in args]
    table.loc[table.index[1:], 'df_diff'] = -np.diff(table['df_resid'].values)
    table['ss_diff'] = -table['ssr'].diff()
    if test == 'F':
        table['F'] = table['ss_diff'] / table['df_diff'] / scale
        table[pr_test] = stats.f.sf(table['F'], table['df_diff'], table['df_resid'])
        table.loc[table['F'].isnull(), pr_test] = np.nan
    return table

def _not_slice(slices, slices_to_exclude, n):
    if False:
        print('Hello World!')
    ind = np.array([True] * n)
    for term in slices_to_exclude:
        s = slices[term]
        ind[s] = False
    return ind

def _ssr_reduced_model(y, x, term_slices, params, keys):
    if False:
        i = 10
        return i + 15
    '\n    Residual sum of squares of OLS model excluding factors in `keys`\n    Assumes x matrix is orthogonal\n\n    Parameters\n    ----------\n    y : array_like\n        dependent variable\n    x : array_like\n        independent variables\n    term_slices : a dict of slices\n        term_slices[key] is a boolean array specifies the parameters\n        associated with the factor `key`\n    params : ndarray\n        OLS solution of y = x * params\n    keys : keys for term_slices\n        factors to be excluded\n\n    Returns\n    -------\n    rss : float\n        residual sum of squares\n    df : int\n        degrees of freedom\n    '
    ind = _not_slice(term_slices, keys, x.shape[1])
    params1 = params[ind]
    ssr = np.subtract(y, x[:, ind].dot(params1))
    ssr = ssr.T.dot(ssr)
    df_resid = len(y) - len(params1)
    return (ssr, df_resid)

class AnovaRM:
    """
    Repeated measures Anova using least squares regression

    The full model regression residual sum of squares is
    used to compare with the reduced model for calculating the
    within-subject effect sum of squares [1].

    Currently, only fully balanced within-subject designs are supported.
    Calculation of between-subject effects and corrections for violation of
    sphericity are not yet implemented.

    Parameters
    ----------
    data : DataFrame
    depvar : str
        The dependent variable in `data`
    subject : str
        Specify the subject id
    within : list[str]
        The within-subject factors
    between : list[str]
        The between-subject factors, this is not yet implemented
    aggregate_func : {None, 'mean', callable}
        If the data set contains more than a single observation per subject
        and cell of the specified model, this function will be used to
        aggregate the data before running the Anova. `None` (the default) will
        not perform any aggregation; 'mean' is s shortcut to `numpy.mean`.
        An exception will be raised if aggregation is required, but no
        aggregation function was specified.

    Returns
    -------
    results : AnovaResults instance

    Raises
    ------
    ValueError
        If the data need to be aggregated, but `aggregate_func` was not
        specified.

    Notes
    -----
    This implementation currently only supports fully balanced designs. If the
    data contain more than one observation per subject and cell of the design,
    these observations need to be aggregated into a single observation
    before the Anova is calculated, either manually or by passing an aggregation
    function via the `aggregate_func` keyword argument.
    Note that if the input data set was not balanced before performing the
    aggregation, the implied heteroscedasticity of the data is ignored.

    References
    ----------
    .. [*] Rutherford, Andrew. Anova and ANCOVA: a GLM approach. John Wiley & Sons, 2011.
    """

    def __init__(self, data, depvar, subject, within=None, between=None, aggregate_func=None):
        if False:
            for i in range(10):
                print('nop')
        self.data = data
        self.depvar = depvar
        self.within = within
        if 'C' in within:
            raise ValueError("Factor name cannot be 'C'! This is in conflict with patsy's contrast function name.")
        self.between = between
        if between is not None:
            raise NotImplementedError('Between subject effect not yet supported!')
        self.subject = subject
        if aggregate_func == 'mean':
            self.aggregate_func = np.mean
        else:
            self.aggregate_func = aggregate_func
        if not data.equals(data.drop_duplicates(subset=[subject] + within)):
            if self.aggregate_func is not None:
                self._aggregate()
            else:
                msg = 'The data set contains more than one observation per subject and cell. Either aggregate the data manually, or pass the `aggregate_func` parameter.'
                raise ValueError(msg)
        self._check_data_balanced()

    def _aggregate(self):
        if False:
            while True:
                i = 10
        self.data = self.data.groupby([self.subject] + self.within, as_index=False)[self.depvar].agg(self.aggregate_func)

    def _check_data_balanced(self):
        if False:
            for i in range(10):
                print('nop')
        'raise if data is not balanced\n\n        This raises a ValueError if the data is not balanced, and\n        returns None if it is balance\n\n        Return might change\n        '
        factor_levels = 1
        for wi in self.within:
            factor_levels *= len(self.data[wi].unique())
        cell_count = {}
        for index in range(self.data.shape[0]):
            key = []
            for col in self.within:
                key.append(self.data[col].iloc[index])
            key = tuple(key)
            if key in cell_count:
                cell_count[key] = cell_count[key] + 1
            else:
                cell_count[key] = 1
        error_message = 'Data is unbalanced.'
        if len(cell_count) != factor_levels:
            raise ValueError(error_message)
        count = cell_count[key]
        for key in cell_count:
            if count != cell_count[key]:
                raise ValueError(error_message)
        if self.data.shape[0] > count * factor_levels:
            raise ValueError('There are more than 1 element in a cell! Missing factors?')

    def fit(self):
        if False:
            return 10
        'estimate the model and compute the Anova table\n\n        Returns\n        -------\n        AnovaResults instance\n        '
        y = self.data[self.depvar].values
        within = ['C(%s, Sum)' % i for i in self.within]
        subject = 'C(%s, Sum)' % self.subject
        factors = within + [subject]
        x = patsy.dmatrix('*'.join(factors), data=self.data)
        term_slices = x.design_info.term_name_slices
        for key in term_slices:
            ind = np.array([False] * x.shape[1])
            ind[term_slices[key]] = True
            term_slices[key] = np.array(ind)
        term_exclude = [':'.join(factors)]
        ind = _not_slice(term_slices, term_exclude, x.shape[1])
        x = x[:, ind]
        model = OLS(y, x)
        results = model.fit()
        if model.rank < x.shape[1]:
            raise ValueError('Independent variables are collinear.')
        for i in term_exclude:
            term_slices.pop(i)
        for key in term_slices:
            term_slices[key] = term_slices[key][ind]
        params = results.params
        df_resid = results.df_resid
        ssr = results.ssr
        columns = ['F Value', 'Num DF', 'Den DF', 'Pr > F']
        anova_table = pd.DataFrame(np.zeros((0, 4)), columns=columns)
        for key in term_slices:
            if self.subject not in key and key != 'Intercept':
                (ssr1, df_resid1) = _ssr_reduced_model(y, x, term_slices, params, [key])
                df1 = df_resid1 - df_resid
                msm = (ssr1 - ssr) / df1
                if key == ':'.join(factors[:-1]) or key + ':' + subject not in term_slices:
                    mse = ssr / df_resid
                    df2 = df_resid
                else:
                    (ssr1, df_resid1) = _ssr_reduced_model(y, x, term_slices, params, [key + ':' + subject])
                    df2 = df_resid1 - df_resid
                    mse = (ssr1 - ssr) / df2
                F = msm / mse
                p = stats.f.sf(F, df1, df2)
                term = key.replace('C(', '').replace(', Sum)', '')
                anova_table.loc[term, 'F Value'] = F
                anova_table.loc[term, 'Num DF'] = df1
                anova_table.loc[term, 'Den DF'] = df2
                anova_table.loc[term, 'Pr > F'] = p
        return AnovaResults(anova_table)

class AnovaResults:
    """
    Anova results class

    Attributes
    ----------
    anova_table : DataFrame
    """

    def __init__(self, anova_table):
        if False:
            i = 10
            return i + 15
        self.anova_table = anova_table

    def __str__(self):
        if False:
            while True:
                i = 10
        return self.summary().__str__()

    def summary(self):
        if False:
            for i in range(10):
                print('nop')
        'create summary results\n\n        Returns\n        -------\n        summary : summary2.Summary instance\n        '
        summ = summary2.Summary()
        summ.add_title('Anova')
        summ.add_df(self.anova_table)
        return summ
if __name__ == '__main__':
    import pandas
    from statsmodels.formula.api import ols
    moore = pandas.read_csv('moore.csv', skiprows=1, names=['partner_status', 'conformity', 'fcategory', 'fscore'])
    moore_lm = ols('conformity ~ C(fcategory, Sum)*C(partner_status, Sum)', data=moore).fit()
    mooreB = ols('conformity ~ C(partner_status, Sum)', data=moore).fit()
    table = anova_lm(moore_lm, typ=2)