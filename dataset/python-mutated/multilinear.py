"""Analyze a set of multiple variables with a linear models

multiOLS:
    take a model and test it on a series of variables defined over a
    pandas dataset, returning a summary for each variable

multigroup:
    take a boolean vector and the definition of several groups of variables
    and test if the group has a fraction of true values higher than the
    rest. It allows to test if the variables in the group are significantly
    more significant than outside the group.
"""
from patsy import dmatrix
import pandas as pd
from statsmodels.api import OLS
from statsmodels.api import stats
import numpy as np
import logging

def _model2dataframe(model_endog, model_exog, model_type=OLS, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    'return a series containing the summary of a linear model\n\n    All the exceding parameters will be redirected to the linear model\n    '
    model_result = model_type(model_endog, model_exog, **kwargs).fit()
    statistics = pd.Series({'r2': model_result.rsquared, 'adj_r2': model_result.rsquared_adj})
    result_df = pd.DataFrame({'params': model_result.params, 'pvals': model_result.pvalues, 'std': model_result.bse, 'statistics': statistics})
    fisher_df = pd.DataFrame({'params': {'_f_test': model_result.fvalue}, 'pvals': {'_f_test': model_result.f_pvalue}})
    res_series = pd.concat([result_df, fisher_df]).unstack()
    return res_series.dropna()

def multiOLS(model, dataframe, column_list=None, method='fdr_bh', alpha=0.05, subset=None, model_type=OLS, **kwargs):
    if False:
        print('Hello World!')
    "apply a linear model to several endogenous variables on a dataframe\n\n    Take a linear model definition via formula and a dataframe that will be\n    the environment of the model, and apply the linear model to a subset\n    (or all) of the columns of the dataframe. It will return a dataframe\n    with part of the information from the linear model summary.\n\n    Parameters\n    ----------\n    model : str\n        formula description of the model\n    dataframe : pandas.dataframe\n        dataframe where the model will be evaluated\n    column_list : list[str], optional\n        Names of the columns to analyze with the model.\n        If None (Default) it will perform the function on all the\n        eligible columns (numerical type and not in the model definition)\n    model_type : model class, optional\n        The type of model to be used. The default is the linear model.\n        Can be any linear model (OLS, WLS, GLS, etc..)\n    method : str, optional\n        the method used to perform the pvalue correction for multiple testing.\n        default is the Benjamini/Hochberg, other available methods are:\n\n            `bonferroni` : one-step correction\n            `sidak` : on-step correction\n            `holm-sidak` :\n            `holm` :\n            `simes-hochberg` :\n            `hommel` :\n            `fdr_bh` : Benjamini/Hochberg\n            `fdr_by` : Benjamini/Yekutieli\n\n    alpha : float, optional\n        the significance level used for the pvalue correction (default 0.05)\n    subset : bool array\n        the selected rows to be used in the regression\n\n    all the other parameters will be directed to the model creation.\n\n    Returns\n    -------\n    summary : pandas.DataFrame\n        a dataframe containing an extract from the summary of the model\n        obtained for each columns. It will give the model complexive f test\n        result and p-value, and the regression value and standard deviarion\n        for each of the regressors. The DataFrame has a hierachical column\n        structure, divided as:\n\n            - params: contains the parameters resulting from the models. Has\n            an additional column named _f_test containing the result of the\n            F test.\n            - pval: the pvalue results of the models. Has the _f_test column\n            for the significativity of the whole test.\n            - adj_pval: the corrected pvalues via the multitest function.\n            - std: uncertainties of the model parameters\n            - statistics: contains the r squared statistics and the adjusted\n            r squared.\n\n    Notes\n    -----\n    The main application of this function is on system biology to perform\n    a linear model testing of a lot of different parameters, like the\n    different genetic expression of several genes.\n\n    See Also\n    --------\n    statsmodels.stats.multitest\n        contains several functions to perform the multiple p-value correction\n\n    Examples\n    --------\n    Using the longley data as dataframe example\n\n    >>> import statsmodels.api as sm\n    >>> data = sm.datasets.longley.load_pandas()\n    >>> df = data.exog\n    >>> df['TOTEMP'] = data.endog\n\n    This will perform the specified linear model on all the\n    other columns of the dataframe\n    >>> multiOLS('GNP + 1', df)\n\n    This select only a certain subset of the columns\n    >>> multiOLS('GNP + 0', df, ['GNPDEFL', 'TOTEMP', 'POP'])\n\n    It is possible to specify a trasformation also on the target column,\n    conforming to the patsy formula specification\n    >>> multiOLS('GNP + 0', df, ['I(GNPDEFL**2)', 'center(TOTEMP)'])\n\n    It is possible to specify the subset of the dataframe\n    on which perform the analysis\n    >> multiOLS('GNP + 1', df, subset=df.GNPDEFL > 90)\n\n    Even a single column name can be given without enclosing it in a list\n    >>> multiOLS('GNP + 0', df, 'GNPDEFL')\n    "
    if column_list is None:
        column_list = [name for name in dataframe.columns if dataframe[name].dtype != object and name not in model]
    if isinstance(column_list, str):
        column_list = [column_list]
    if subset is not None:
        dataframe = dataframe.loc[subset]
    col_results = {}
    model_exog = dmatrix(model, data=dataframe, return_type='dataframe')
    for col_name in column_list:
        try:
            model_endog = dataframe[col_name]
        except KeyError:
            model_endog = dmatrix(col_name + ' + 0', data=dataframe)
        res = _model2dataframe(model_endog, model_exog, model_type, **kwargs)
        col_results[col_name] = res
    summary = pd.DataFrame(col_results)
    summary = summary.T.sort_values([('pvals', '_f_test')])
    summary.index.name = 'endogenous vars'
    smt = stats.multipletests
    for (key1, key2) in summary:
        if key1 != 'pvals':
            continue
        p_values = summary[key1, key2]
        corrected = smt(p_values, method=method, alpha=alpha)[1]
        summary['adj_' + key1, key2] = corrected
    return summary

def _test_group(pvalues, group_name, group, exact=True):
    if False:
        i = 10
        return i + 15
    'test if the objects in the group are different from the general set.\n\n    The test is performed on the pvalues set (ad a pandas series) over\n    the group specified via a fisher exact test.\n    '
    from scipy.stats import fisher_exact, chi2_contingency
    totals = 1.0 * len(pvalues)
    total_significant = 1.0 * np.sum(pvalues)
    cross_index = [c for c in group if c in pvalues.index]
    missing = [c for c in group if c not in pvalues.index]
    if missing:
        s = 'the test is not well defined if the group has elements not presents in the significativity array. group name: {}, missing elements: {}'
        logging.warning(s.format(group_name, missing))
    group_total = 1.0 * len(cross_index)
    group_sign = 1.0 * len([c for c in cross_index if pvalues[c]])
    group_nonsign = 1.0 * (group_total - group_sign)
    extern_sign = 1.0 * (total_significant - group_sign)
    extern_nonsign = 1.0 * (totals - total_significant - group_nonsign)
    test = fisher_exact if exact else chi2_contingency
    table = [[extern_nonsign, extern_sign], [group_nonsign, group_sign]]
    pvalue = test(np.array(table))[1]
    part = (group_sign, group_nonsign, extern_sign, extern_nonsign)
    increase = np.log(totals * group_sign / (total_significant * group_total))
    return (pvalue, increase, part)

def multigroup(pvals, groups, exact=True, keep_all=True, alpha=0.05):
    if False:
        i = 10
        return i + 15
    'Test if the given groups are different from the total partition.\n\n    Given a boolean array test if each group has a proportion of positives\n    different than the complexive proportion.\n    The test can be done as an exact Fisher test or approximated as a\n    Chi squared test for more speed.\n\n    Parameters\n    ----------\n    pvals : pandas series of boolean\n        the significativity of the variables under analysis\n    groups : dict of list\n        the name of each category of variables under exam.\n        each one is a list of the variables included\n    exact : bool, optional\n        If True (default) use the fisher exact test, otherwise\n        use the chi squared test for contingencies tables.\n        For high number of elements in the array the fisher test can\n        be significantly slower than the chi squared.\n    keep_all : bool, optional\n        if False it will drop those groups where the fraction\n        of positive is below the expected result. If True (default)\n         it will keep all the significant results.\n    alpha : float, optional\n        the significativity level for the pvalue correction\n        on the whole set of groups (not inside the groups themselves).\n\n    Returns\n    -------\n    result_df: pandas dataframe\n        for each group returns:\n\n            pvals - the fisher p value of the test\n            adj_pvals - the adjusted pvals\n            increase - the log of the odd ratio between the\n                internal significant ratio versus the external one\n            _in_sign - significative elements inside the group\n            _in_non - non significative elements inside the group\n            _out_sign - significative elements outside the group\n            _out_non - non significative elements outside the group\n\n    Notes\n    -----\n    This test allow to see if a category of variables is generally better\n    suited to be described for the model. For example to see if a predictor\n    gives more information on demographic or economical parameters,\n    by creating two groups containing the endogenous variables of each\n    category.\n\n    This function is conceived for medical dataset with a lot of variables\n    that can be easily grouped into functional groups. This is because\n    The significativity of a group require a rather large number of\n    composing elements.\n\n    Examples\n    --------\n    A toy example on a real dataset, the Guerry dataset from R\n    >>> url = "https://raw.githubusercontent.com/vincentarelbundock/"\n    >>> url = url + "Rdatasets/csv/HistData/Guerry.csv"\n    >>> df = pd.read_csv(url, index_col=\'dept\')\n\n    evaluate the relationship between the various paramenters whith the Wealth\n    >>> pvals = multiOLS(\'Wealth\', df)[\'adj_pvals\', \'_f_test\']\n\n    define the groups\n    >>> groups = {}\n    >>> groups[\'crime\'] = [\'Crime_prop\', \'Infanticide\',\n    ...     \'Crime_parents\', \'Desertion\', \'Crime_pers\']\n    >>> groups[\'religion\'] = [\'Donation_clergy\', \'Clergy\', \'Donations\']\n    >>> groups[\'wealth\'] = [\'Commerce\', \'Lottery\', \'Instruction\', \'Literacy\']\n\n    do the analysis of the significativity\n    >>> multigroup(pvals < 0.05, groups)\n    '
    pvals = pd.Series(pvals)
    if not set(pvals.unique()) <= set([False, True]):
        raise ValueError('the series should be binary')
    if hasattr(pvals.index, 'is_unique') and (not pvals.index.is_unique):
        raise ValueError('series with duplicated index is not accepted')
    results = {'pvals': {}, 'increase': {}, '_in_sign': {}, '_in_non': {}, '_out_sign': {}, '_out_non': {}}
    for (group_name, group_list) in groups.items():
        res = _test_group(pvals, group_name, group_list, exact)
        results['pvals'][group_name] = res[0]
        results['increase'][group_name] = res[1]
        results['_in_sign'][group_name] = res[2][0]
        results['_in_non'][group_name] = res[2][1]
        results['_out_sign'][group_name] = res[2][2]
        results['_out_non'][group_name] = res[2][3]
    result_df = pd.DataFrame(results).sort_values('pvals')
    if not keep_all:
        result_df = result_df[result_df.increase]
    smt = stats.multipletests
    corrected = smt(result_df['pvals'], method='fdr_bh', alpha=alpha)[1]
    result_df['adj_pvals'] = corrected
    return result_df