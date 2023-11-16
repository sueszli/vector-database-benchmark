""" convenience functions for ANOVA type analysis with OLS

Note: statistical results of ANOVA are not checked, OLS is
checked but not whether the reported results are the ones used
in ANOVA

includes form2design for creating dummy variables

TODO:
 * ...
 *

"""
from statsmodels.compat.python import lmap
import numpy as np
import statsmodels.api as sm

def data2dummy(x, returnall=False):
    if False:
        for i in range(10):
            print('nop')
    'convert array of categories to dummy variables\n    by default drops dummy variable for last category\n    uses ravel, 1d only'
    x = x.ravel()
    groups = np.unique(x)
    if returnall:
        return (x[:, None] == groups).astype(int)
    else:
        return (x[:, None] == groups).astype(int)[:, :-1]

def data2proddummy(x):
    if False:
        i = 10
        return i + 15
    'creates product dummy variables from 2 columns of 2d array\n\n    drops last dummy variable, but not from each category\n    singular with simple dummy variable but not with constant\n\n    quickly written, no safeguards\n\n    '
    groups = np.unique(lmap(tuple, x.tolist()))
    return (x == groups[:, None, :]).all(-1).T.astype(int)[:, :-1]

def data2groupcont(x1, x2):
    if False:
        i = 10
        return i + 15
    'create dummy continuous variable\n\n    Parameters\n    ----------\n    x1 : 1d array\n        label or group array\n    x2 : 1d array (float)\n        continuous variable\n\n    Notes\n    -----\n    useful for group specific slope coefficients in regression\n    '
    if x2.ndim == 1:
        x2 = x2[:, None]
    dummy = data2dummy(x1, returnall=True)
    return dummy * x2
anova_str0 = '\nANOVA statistics (model sum of squares excludes constant)\nSource    DF  Sum Squares   Mean Square    F Value    Pr > F\nModel     %(df_model)i        %(ess)f       %(mse_model)f   %(fvalue)f %(f_pvalue)f\nError     %(df_resid)i     %(ssr)f       %(mse_resid)f\nCTotal    %(nobs)i    %(uncentered_tss)f     %(mse_total)f\n\nR squared  %(rsquared)f\n'
anova_str = '\nANOVA statistics (model sum of squares includes constant)\nSource    DF  Sum Squares   Mean Square    F Value    Pr > F\nModel     %(df_model)i      %(ssmwithmean)f       %(mse_model)f   %(fvalue)f %(f_pvalue)f\nError     %(df_resid)i     %(ssr)f       %(mse_resid)f\nCTotal    %(nobs)i    %(uncentered_tss)f     %(mse_total)f\n\nR squared  %(rsquared)f\n'

def anovadict(res):
    if False:
        for i in range(10):
            print('nop')
    'update regression results dictionary with ANOVA specific statistics\n\n    not checked for completeness\n    '
    ad = {}
    ad.update(res.__dict__)
    anova_attr = ['df_model', 'df_resid', 'ess', 'ssr', 'uncentered_tss', 'mse_model', 'mse_resid', 'mse_total', 'fvalue', 'f_pvalue', 'rsquared']
    for key in anova_attr:
        ad[key] = getattr(res, key)
    ad['nobs'] = res.model.nobs
    ad['ssmwithmean'] = res.uncentered_tss - res.ssr
    return ad

def form2design(ss, data):
    if False:
        i = 10
        return i + 15
    "convert string formula to data dictionary\n\n    ss : str\n     * I : add constant\n     * varname : for simple varnames data is used as is\n     * F:varname : create dummy variables for factor varname\n     * P:varname1*varname2 : create product dummy variables for\n       varnames\n     * G:varname1*varname2 : create product between factor and\n       continuous variable\n    data : dict or structured array\n       data set, access of variables by name as in dictionaries\n\n    Returns\n    -------\n    vars : dictionary\n        dictionary of variables with converted dummy variables\n    names : list\n        list of names, product (P:) and grouped continuous\n        variables (G:) have name by joining individual names\n        sorted according to input\n\n    Examples\n    --------\n    >>> xx, n = form2design('I a F:b P:c*d G:c*f', testdata)\n    >>> xx.keys()\n    ['a', 'b', 'const', 'cf', 'cd']\n    >>> n\n    ['const', 'a', 'b', 'cd', 'cf']\n\n    Notes\n    -----\n\n    with sorted dict, separate name list would not be necessary\n    "
    vars = {}
    names = []
    for item in ss.split():
        if item == 'I':
            vars['const'] = np.ones(data.shape[0])
            names.append('const')
        elif ':' not in item:
            vars[item] = data[item]
            names.append(item)
        elif item[:2] == 'F:':
            v = item.split(':')[1]
            vars[v] = data2dummy(data[v])
            names.append(v)
        elif item[:2] == 'P:':
            v = item.split(':')[1].split('*')
            vars[''.join(v)] = data2proddummy(np.c_[data[v[0]], data[v[1]]])
            names.append(''.join(v))
        elif item[:2] == 'G:':
            v = item.split(':')[1].split('*')
            vars[''.join(v)] = data2groupcont(data[v[0]], data[v[1]])
            names.append(''.join(v))
        else:
            raise ValueError('unknown expression in formula')
    return (vars, names)

def dropname(ss, li):
    if False:
        while True:
            i = 10
    'drop names from a list of strings,\n    names to drop are in space delimited list\n    does not change original list\n    '
    newli = li[:]
    for item in ss.split():
        newli.remove(item)
    return newli
if __name__ == '__main__':
    nobs = 1000
    testdataint = np.random.randint(3, size=(nobs, 4)).view([('a', int), ('b', int), ('c', int), ('d', int)])
    testdatacont = np.random.normal(size=(nobs, 2)).view([('e', float), ('f', float)])
    import numpy.lib.recfunctions
    dt2 = numpy.lib.recfunctions.zip_descr((testdataint, testdatacont), flatten=True)
    testdata = np.empty((nobs, 1), dt2)
    for name in testdataint.dtype.names:
        testdata[name] = testdataint[name]
    for name in testdatacont.dtype.names:
        testdata[name] = testdatacont[name]
    if 0:
        (xx, n) = form2design('F:a', testdata)
        print(xx)
        print(form2design('P:a*b', testdata))
        print(data2proddummy(np.c_[testdata['a'], testdata['b']]))
        (xx, names) = form2design('a F:b P:c*d', testdata)
    (xx, names) = form2design('I a F:b P:c*d', testdata)
    (xx, names) = form2design('I a F:b P:c*d G:a*e f', testdata)
    X = np.column_stack([xx[nn] for nn in names])
    y = X.sum(1) + 0.01 * np.random.normal(size=nobs)
    rest1 = sm.OLS(y, X).fit()
    print(rest1.params)
    print(anova_str % anovadict(rest1))
    X = np.column_stack([xx[nn] for nn in dropname('ae f', names)])
    y = X.sum(1) + 0.01 * np.random.normal(size=nobs)
    rest1 = sm.OLS(y, X).fit()
    print(rest1.params)
    print(anova_str % anovadict(rest1))
    dt_b = np.dtype([('breed', int), ('sex', int), ('litter', int), ('pen', int), ('pig', int), ('age', float), ('bage', float), ('y', float)])
    dta = np.genfromtxt('dftest3.data', dt_b, missing='.', usemask=True)
    print('missing', [dta.mask[k].sum() for k in dta.dtype.names])
    m = dta.mask.view(bool)
    droprows = m.reshape(-1, len(dta.dtype.names)).any(1)
    dta_use_b1 = dta[~droprows, :].data
    print(dta_use_b1.shape)
    print(dta_use_b1.dtype)
    (xx_b1, names_b1) = form2design('I F:sex age', dta_use_b1)
    X_b1 = np.column_stack([xx_b1[nn] for nn in dropname('', names_b1)])
    y_b1 = dta_use_b1['y']
    rest_b1 = sm.OLS(y_b1, X_b1).fit()
    print(rest_b1.params)
    print(anova_str % anovadict(rest_b1))
    allexog = ' '.join(dta.dtype.names[:-1])
    (xx_b1a, names_b1a) = form2design('I F:breed F:sex F:litter F:pen age bage', dta_use_b1)
    X_b1a = np.column_stack([xx_b1a[nn] for nn in dropname('', names_b1a)])
    y_b1a = dta_use_b1['y']
    rest_b1a = sm.OLS(y_b1a, X_b1a).fit()
    print(rest_b1a.params)
    print(anova_str % anovadict(rest_b1a))
    for dropn in names_b1a:
        print(('\nResults dropping', dropn))
        X_b1a_ = np.column_stack([xx_b1a[nn] for nn in dropname(dropn, names_b1a)])
        y_b1a_ = dta_use_b1['y']
        rest_b1a_ = sm.OLS(y_b1a_, X_b1a_).fit()
        print(anova_str % anovadict(rest_b1a_))