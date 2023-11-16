from statsmodels.compat.python import lzip
import datetime
from functools import reduce
import re
import textwrap
import numpy as np
import pandas as pd
from .table import SimpleTable
from .tableformatting import fmt_latex, fmt_txt

class Summary:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.tables = []
        self.settings = []
        self.extra_txt = []
        self.title = None
        self._merge_latex = False

    def __str__(self):
        if False:
            return 10
        return self.as_text()

    def __repr__(self):
        if False:
            return 10
        return str(type(self)) + '\n"""\n' + self.__str__() + '\n"""'

    def _repr_html_(self):
        if False:
            i = 10
            return i + 15
        'Display as HTML in IPython notebook.'
        return self.as_html()

    def _repr_latex_(self):
        if False:
            while True:
                i = 10
        'Display as LaTeX when converting IPython notebook to PDF.'
        return self.as_latex()

    def add_df(self, df, index=True, header=True, float_format='%.4f', align='r'):
        if False:
            print('Hello World!')
        '\n        Add the contents of a DataFrame to summary table\n\n        Parameters\n        ----------\n        df : DataFrame\n        header : bool\n            Reproduce the DataFrame column labels in summary table\n        index : bool\n            Reproduce the DataFrame row labels in summary table\n        float_format : str\n            Formatting to float data columns\n        align : str\n            Data alignment (l/c/r)\n        '
        settings = {'index': index, 'header': header, 'float_format': float_format, 'align': align}
        self.tables.append(df)
        self.settings.append(settings)

    def add_array(self, array, align='r', float_format='%.4f'):
        if False:
            print('Hello World!')
        'Add the contents of a Numpy array to summary table\n\n        Parameters\n        ----------\n        array : numpy array (2D)\n        float_format : str\n            Formatting to array if type is float\n        align : str\n            Data alignment (l/c/r)\n        '
        table = pd.DataFrame(array)
        self.add_df(table, index=False, header=False, float_format=float_format, align=align)

    def add_dict(self, d, ncols=2, align='l', float_format='%.4f'):
        if False:
            for i in range(10):
                print('nop')
        'Add the contents of a Dict to summary table\n\n        Parameters\n        ----------\n        d : dict\n            Keys and values are automatically coerced to strings with str().\n            Users are encouraged to format them before using add_dict.\n        ncols : int\n            Number of columns of the output table\n        align : str\n            Data alignment (l/c/r)\n        float_format : str\n            Formatting to float data columns\n        '
        keys = [_formatter(x, float_format) for x in d.keys()]
        vals = [_formatter(x, float_format) for x in d.values()]
        data = np.array(lzip(keys, vals))
        if data.shape[0] % ncols != 0:
            pad = ncols - data.shape[0] % ncols
            data = np.vstack([data, np.array(pad * [['', '']])])
        data = np.split(data, ncols)
        data = reduce(lambda x, y: np.hstack([x, y]), data)
        self.add_array(data, align=align)

    def add_text(self, string):
        if False:
            return 10
        'Append a note to the bottom of the summary table. In ASCII tables,\n        the note will be wrapped to table width. Notes are not indented.\n        '
        self.extra_txt.append(string)

    def add_title(self, title=None, results=None):
        if False:
            return 10
        'Insert a title on top of the summary table. If a string is provided\n        in the title argument, that string is printed. If no title string is\n        provided but a results instance is provided, statsmodels attempts\n        to construct a useful title automatically.\n        '
        if isinstance(title, str):
            self.title = title
        elif results is not None:
            model = results.model.__class__.__name__
            if model in _model_types:
                model = _model_types[model]
            self.title = 'Results: ' + model
        else:
            self.title = ''

    def add_base(self, results, alpha=0.05, float_format='%.4f', title=None, xname=None, yname=None):
        if False:
            i = 10
            return i + 15
        'Try to construct a basic summary instance.\n\n        Parameters\n        ----------\n        results : Model results instance\n        alpha : float\n            significance level for the confidence intervals (optional)\n        float_format: str\n            Float formatting for summary of parameters (optional)\n        title : str\n            Title of the summary table (optional)\n        xname : list[str] of length equal to the number of parameters\n            Names of the independent variables (optional)\n        yname : str\n            Name of the dependent variable (optional)\n        '
        param = summary_params(results, alpha=alpha, use_t=results.use_t)
        info = summary_model(results)
        if xname is not None:
            param.index = xname
        if yname is not None:
            info['Dependent Variable:'] = yname
        self.add_dict(info, align='l')
        self.add_df(param, float_format=float_format)
        self.add_title(title=title, results=results)

    def as_text(self):
        if False:
            while True:
                i = 10
        'Generate ASCII Summary Table\n        '
        tables = self.tables
        settings = self.settings
        title = self.title
        extra_txt = self.extra_txt
        (pad_col, pad_index, widest) = _measure_tables(tables, settings)
        rule_equal = widest * '='
        simple_tables = _simple_tables(tables, settings, pad_col, pad_index)
        tab = [x.as_text() for x in simple_tables]
        tab = '\n'.join(tab)
        tab = tab.split('\n')
        tab[0] = rule_equal
        tab.append(rule_equal)
        tab = '\n'.join(tab)
        if title is not None:
            title = title
            if len(title) < widest:
                title = ' ' * int(widest / 2 - len(title) / 2) + title
        else:
            title = ''
        txt = [textwrap.wrap(x, widest) for x in extra_txt]
        txt = ['\n'.join(x) for x in txt]
        txt = '\n'.join(txt)
        out = '\n'.join([title, tab, txt])
        return out

    def as_html(self):
        if False:
            i = 10
            return i + 15
        'Generate HTML Summary Table\n        '
        tables = self.tables
        settings = self.settings
        simple_tables = _simple_tables(tables, settings)
        tab = [x.as_html() for x in simple_tables]
        tab = '\n'.join(tab)
        temp_txt = [st.replace('\n', '<br/>\n') for st in self.extra_txt]
        txt = '<br/>\n'.join(temp_txt)
        out = '<br/>\n'.join([tab, txt])
        return out

    def as_latex(self, label=''):
        if False:
            print('Hello World!')
        'Generate LaTeX Summary Table\n\n        Parameters\n        ----------\n        label : str\n            Label of the summary table that can be referenced\n            in a latex document (optional)\n        '
        tables = self.tables
        settings = self.settings
        title = self.title
        if title is not None:
            title = '\\caption{' + title + '}'
        else:
            title = '\\caption{}'
        label = '\\label{' + label + '}'
        simple_tables = _simple_tables(tables, settings)
        tab = [x.as_latex_tabular() for x in simple_tables]
        tab = '\n\n'.join(tab)
        to_replace = '\\\\hline\\n\\\\hline\\n\\\\end{tabular}\\n\\\\begin{tabular}{.*}\\n'
        if self._merge_latex:
            tab = re.sub(to_replace, '\\\\midrule\\n', tab)
        non_captioned = ('\\begin{table}', title, label, tab, '\\end{table}')
        non_captioned = '\n'.join(non_captioned)
        txt = ' \\newline \n'.join(self.extra_txt)
        out = non_captioned + '\n\\bigskip\n' + txt
        return out

def _measure_tables(tables, settings):
    if False:
        i = 10
        return i + 15
    'Compare width of ascii tables in a list and calculate padding values.\n    We add space to each col_sep to get us as close as possible to the\n    width of the largest table. Then, we add a few spaces to the first\n    column to pad the rest.\n    '
    simple_tables = _simple_tables(tables, settings)
    tab = [x.as_text() for x in simple_tables]
    length = [len(x.splitlines()[0]) for x in tab]
    len_max = max(length)
    pad_sep = []
    pad_index = []
    for i in range(len(tab)):
        nsep = max(tables[i].shape[1] - 1, 1)
        pad = int((len_max - length[i]) / nsep)
        pad_sep.append(pad)
        len_new = length[i] + nsep * pad
        pad_index.append(len_max - len_new)
    return (pad_sep, pad_index, max(length))
_model_types = {'OLS': 'Ordinary least squares', 'GLS': 'Generalized least squares', 'GLSAR': 'Generalized least squares with AR(p)', 'WLS': 'Weighted least squares', 'RLM': 'Robust linear model', 'NBin': 'Negative binomial model', 'GLM': 'Generalized linear model'}

def summary_model(results):
    if False:
        return 10
    '\n    Create a dict with information about the model\n    '

    def time_now(*args, **kwds):
        if False:
            for i in range(10):
                print('nop')
        now = datetime.datetime.now()
        return now.strftime('%Y-%m-%d %H:%M')
    info = {}
    info['Model:'] = lambda x: x.model.__class__.__name__
    info['Model Family:'] = lambda x: x.family.__class.__name__
    info['Link Function:'] = lambda x: x.family.link.__class__.__name__
    info['Dependent Variable:'] = lambda x: x.model.endog_names
    info['Date:'] = time_now
    info['No. Observations:'] = lambda x: '%#6d' % x.nobs
    info['Df Model:'] = lambda x: '%#6d' % x.df_model
    info['Df Residuals:'] = lambda x: '%#6d' % x.df_resid
    info['Converged:'] = lambda x: x.mle_retvals['converged']
    info['No. Iterations:'] = lambda x: x.mle_retvals['iterations']
    info['Method:'] = lambda x: x.method
    info['Norm:'] = lambda x: x.fit_options['norm']
    info['Scale Est.:'] = lambda x: x.fit_options['scale_est']
    info['Cov. Type:'] = lambda x: x.fit_options['cov']
    rsquared_type = '' if results.k_constant else ' (uncentered)'
    info['R-squared' + rsquared_type + ':'] = lambda x: '%#8.3f' % x.rsquared
    info['Adj. R-squared' + rsquared_type + ':'] = lambda x: '%#8.3f' % x.rsquared_adj
    info['Pseudo R-squared:'] = lambda x: '%#8.3f' % x.prsquared
    info['AIC:'] = lambda x: '%8.4f' % x.aic
    info['BIC:'] = lambda x: '%8.4f' % x.bic
    info['Log-Likelihood:'] = lambda x: '%#8.5g' % x.llf
    info['LL-Null:'] = lambda x: '%#8.5g' % x.llnull
    info['LLR p-value:'] = lambda x: '%#8.5g' % x.llr_pvalue
    info['Deviance:'] = lambda x: '%#8.5g' % x.deviance
    info['Pearson chi2:'] = lambda x: '%#6.3g' % x.pearson_chi2
    info['F-statistic:'] = lambda x: '%#8.4g' % x.fvalue
    info['Prob (F-statistic):'] = lambda x: '%#6.3g' % x.f_pvalue
    info['Scale:'] = lambda x: '%#8.5g' % x.scale
    out = {}
    for (key, func) in info.items():
        try:
            out[key] = func(results)
        except (AttributeError, KeyError, NotImplementedError):
            pass
    return out

def summary_params(results, yname=None, xname=None, alpha=0.05, use_t=True, skip_header=False, float_format='%.4f'):
    if False:
        return 10
    'create a summary table of parameters from results instance\n\n    Parameters\n    ----------\n    res : results instance\n        some required information is directly taken from the result\n        instance\n    yname : {str, None}\n        optional name for the endogenous variable, default is "y"\n    xname : {list[str], None}\n        optional names for the exogenous variables, default is "var_xx"\n    alpha : float\n        significance level for the confidence intervals\n    use_t : bool\n        indicator whether the p-values are based on the Student-t\n        distribution (if True) or on the normal distribution (if False)\n    skip_header : bool\n        If false (default), then the header row is added. If true, then no\n        header row is added.\n    float_format : str\n        float formatting options (e.g. ".3g")\n\n    Returns\n    -------\n    params_table : SimpleTable instance\n    '
    if isinstance(results, tuple):
        (results, params, bse, tvalues, pvalues, conf_int) = results
    else:
        params = results.params
        bse = results.bse
        tvalues = results.tvalues
        pvalues = results.pvalues
        conf_int = results.conf_int(alpha)
    data = np.array([params, bse, tvalues, pvalues]).T
    data = np.hstack([data, conf_int])
    data = pd.DataFrame(data)
    if use_t:
        data.columns = ['Coef.', 'Std.Err.', 't', 'P>|t|', '[' + str(alpha / 2), str(1 - alpha / 2) + ']']
    else:
        data.columns = ['Coef.', 'Std.Err.', 'z', 'P>|z|', '[' + str(alpha / 2), str(1 - alpha / 2) + ']']
    if not xname:
        try:
            data.index = results.model.data.param_names
        except AttributeError:
            data.index = results.model.exog_names
    else:
        data.index = xname
    return data

def _col_params(result, float_format='%.4f', stars=True, include_r2=False):
    if False:
        while True:
            i = 10
    'Stack coefficients and standard errors in single column\n    '
    res = summary_params(result)
    for col in res.columns[:2]:
        res[col] = res[col].apply(lambda x: float_format % x)
    res.iloc[:, 1] = '(' + res.iloc[:, 1] + ')'
    if stars:
        idx = res.iloc[:, 3] < 0.1
        res.loc[idx, res.columns[0]] = res.loc[idx, res.columns[0]] + '*'
        idx = res.iloc[:, 3] < 0.05
        res.loc[idx, res.columns[0]] = res.loc[idx, res.columns[0]] + '*'
        idx = res.iloc[:, 3] < 0.01
        res.loc[idx, res.columns[0]] = res.loc[idx, res.columns[0]] + '*'
    res = res.iloc[:, :2]
    res = res.stack()
    if include_r2:
        rsquared = getattr(result, 'rsquared', np.nan)
        rsquared_adj = getattr(result, 'rsquared_adj', np.nan)
        r2 = pd.Series({('R-squared', ''): rsquared, ('R-squared Adj.', ''): rsquared_adj})
        if r2.notnull().any():
            r2 = r2.apply(lambda x: float_format % x)
            res = pd.concat([res, r2], axis=0)
    res = pd.DataFrame(res)
    res.columns = [str(result.model.endog_names)]
    return res

def _col_info(result, info_dict=None):
    if False:
        print('Hello World!')
    'Stack model info in a column\n    '
    if info_dict is None:
        info_dict = {}
    out = []
    index = []
    for i in info_dict:
        if isinstance(info_dict[i], dict):
            continue
        try:
            out.append(info_dict[i](result))
        except AttributeError:
            out.append('')
        index.append(i)
    out = pd.DataFrame({str(result.model.endog_names): out}, index=index)
    return out

def _make_unique(list_of_names):
    if False:
        for i in range(10):
            print('nop')
    if len(set(list_of_names)) == len(list_of_names):
        return list_of_names
    from collections import defaultdict
    name_counter = defaultdict(str)
    header = []
    for _name in list_of_names:
        name_counter[_name] += 'I'
        header.append(_name + ' ' + name_counter[_name])
    return header

def summary_col(results, float_format='%.4f', model_names=(), stars=False, info_dict=None, regressor_order=(), drop_omitted=False, include_r2=True):
    if False:
        return 10
    '\n    Summarize multiple results instances side-by-side (coefs and SEs)\n\n    Parameters\n    ----------\n    results : statsmodels results instance or list of result instances\n    float_format : str, optional\n        float format for coefficients and standard errors\n        Default : \'%.4f\'\n    model_names : list[str], optional\n        Must have same length as the number of results. If the names are not\n        unique, a roman number will be appended to all model names\n    stars : bool\n        print significance stars\n    info_dict : dict, default None\n        dict of functions to be applied to results instances to retrieve\n        model info. To use specific information for different models, add a\n        (nested) info_dict with model name as the key.\n        Example: `info_dict = {"N":lambda x:(x.nobs), "R2": ..., "OLS":{\n        "R2":...}}` would only show `R2` for OLS regression models, but\n        additionally `N` for all other results.\n        Default : None (use the info_dict specified in\n        result.default_model_infos, if this property exists)\n    regressor_order : list[str], optional\n        list of names of the regressors in the desired order. All regressors\n        not specified will be appended to the end of the list.\n    drop_omitted : bool, optional\n        Includes regressors that are not specified in regressor_order. If\n        False, regressors not specified will be appended to end of the list.\n        If True, only regressors in regressor_order will be included.\n    include_r2 : bool, optional\n        Includes R2 and adjusted R2 in the summary table.\n    '
    if not isinstance(results, list):
        results = [results]
    cols = [_col_params(x, stars=stars, float_format=float_format, include_r2=include_r2) for x in results]
    if model_names:
        colnames = _make_unique(model_names)
    else:
        colnames = _make_unique([x.columns[0] for x in cols])
    for i in range(len(cols)):
        cols[i].columns = [colnames[i]]

    def merg(x, y):
        if False:
            for i in range(10):
                print('nop')
        return x.merge(y, how='outer', right_index=True, left_index=True)
    summ = reduce(merg, cols)
    if regressor_order:
        varnames = summ.index.get_level_values(0).tolist()
        vc = pd.Series(varnames).value_counts()
        varnames = vc.loc[vc == 2].index.tolist()
        ordered = [x for x in regressor_order if x in varnames]
        unordered = [x for x in varnames if x not in regressor_order]
        new_order = ordered + unordered
        other = [x for x in summ.index.get_level_values(0) if x not in new_order]
        new_order += other
        if drop_omitted:
            for uo in unordered:
                new_order.remove(uo)
        summ = summ.loc[new_order]
    idx = []
    index = summ.index.get_level_values(0)
    for i in range(0, index.shape[0], 2):
        idx.append(index[i])
        if i + 1 < index.shape[0] and index[i] == index[i + 1]:
            idx.append('')
        else:
            idx.append(index[i + 1])
    summ.index = idx
    if info_dict:
        cols = [_col_info(x, info_dict.get(x.model.__class__.__name__, info_dict)) for x in results]
    else:
        cols = [_col_info(x, getattr(x, 'default_model_infos', None)) for x in results]
    for (df, name) in zip(cols, _make_unique([df.columns[0] for df in cols])):
        df.columns = [name]

    def merg(x, y):
        if False:
            print('Hello World!')
        return x.merge(y, how='outer', right_index=True, left_index=True)
    info = reduce(merg, cols)
    dat = pd.DataFrame(np.vstack([summ, info]))
    dat.columns = summ.columns
    dat.index = pd.Index(summ.index.tolist() + info.index.tolist())
    summ = dat
    summ = summ.fillna('')
    smry = Summary()
    smry._merge_latex = True
    smry.add_df(summ, header=True, align='l')
    smry.add_text('Standard errors in parentheses.')
    if stars:
        smry.add_text('* p<.1, ** p<.05, ***p<.01')
    return smry

def _formatter(element, float_format='%.4f'):
    if False:
        i = 10
        return i + 15
    try:
        out = float_format % element
    except (ValueError, TypeError):
        out = str(element)
    return out.strip()

def _df_to_simpletable(df, align='r', float_format='%.4f', header=True, index=True, table_dec_above='-', table_dec_below=None, header_dec_below='-', pad_col=0, pad_index=0):
    if False:
        print('Hello World!')
    dat = df.copy()
    try:
        dat = dat.map(lambda x: _formatter(x, float_format))
    except AttributeError:
        dat = dat.applymap(lambda x: _formatter(x, float_format))
    if header:
        headers = [str(x) for x in dat.columns.tolist()]
    else:
        headers = None
    if index:
        stubs = [str(x) + int(pad_index) * ' ' for x in dat.index.tolist()]
    else:
        dat.iloc[:, 0] = [str(x) + int(pad_index) * ' ' for x in dat.iloc[:, 0]]
        stubs = None
    st = SimpleTable(np.array(dat), headers=headers, stubs=stubs, ltx_fmt=fmt_latex, txt_fmt=fmt_txt)
    st.output_formats['latex']['data_aligns'] = align
    st.output_formats['latex']['header_align'] = align
    st.output_formats['txt']['data_aligns'] = align
    st.output_formats['txt']['table_dec_above'] = table_dec_above
    st.output_formats['txt']['table_dec_below'] = table_dec_below
    st.output_formats['txt']['header_dec_below'] = header_dec_below
    st.output_formats['txt']['colsep'] = ' ' * int(pad_col + 1)
    return st

def _simple_tables(tables, settings, pad_col=None, pad_index=None):
    if False:
        print('Hello World!')
    simple_tables = []
    float_format = settings[0]['float_format'] if settings else '%.4f'
    if pad_col is None:
        pad_col = [0] * len(tables)
    if pad_index is None:
        pad_index = [0] * len(tables)
    for (i, v) in enumerate(tables):
        index = settings[i]['index']
        header = settings[i]['header']
        align = settings[i]['align']
        simple_tables.append(_df_to_simpletable(v, align=align, float_format=float_format, header=header, index=index, pad_col=pad_col[i], pad_index=pad_index[i]))
    return simple_tables