"""Example: Test for equality of coefficients across groups/regressions


Created on Sat Mar 27 22:36:51 2010
Author: josef-pktd
"""
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.sandbox.regression.onewaygls import OneWayLS
example = ['null', 'diff'][1]
example_size = [10, 100][0]
example_size = [(10, 2), (100, 2)][0]
example_groups = ['2', '2-2'][1]
np.random.seed(87654589)
(nobs, nvars) = example_size
x1 = np.random.normal(size=(nobs, nvars))
y1 = 10 + np.dot(x1, [15.0] * nvars) + 2 * np.random.normal(size=nobs)
x1 = sm.add_constant(x1, prepend=False)
x2 = np.random.normal(size=(nobs, nvars))
if example == 'null':
    y2 = 10 + np.dot(x2, [15.0] * nvars) + 2 * np.random.normal(size=nobs)
else:
    y2 = 19 + np.dot(x2, [17.0] * nvars) + 2 * np.random.normal(size=nobs)
x2 = sm.add_constant(x2, prepend=False)
x = np.concatenate((x1, x2), 0)
y = np.concatenate((y1, y2))
if example_groups == '2':
    groupind = (np.arange(2 * nobs) > nobs - 1).astype(int)
else:
    groupind = np.mod(np.arange(2 * nobs), 4)
    groupind.sort()

def print_results(res):
    if False:
        while True:
            i = 10
    groupind = res.groups
    ft = res.ftest_summary()
    print('\nTable of F-tests for overall or pairwise equality of coefficients')
    from statsmodels.iolib import SimpleTable
    print(SimpleTable([['%r' % (row[0],)] + list(row[1]) + ['*'] * (row[1][1] > 0.5).item() for row in ft[1]], headers=['pair', 'F-statistic', 'p-value', 'df_denom', 'df_num']))
    print('Notes: p-values are not corrected for many tests')
    print('       (no Bonferroni correction)')
    print('       * : reject at 5% uncorrected confidence level')
    print('Null hypothesis: all or pairwise coefficient are the same')
    print('Alternative hypothesis: all coefficients are different')
    print('\nComparison with stats.f_oneway')
    print(stats.f_oneway(*[y[groupind == gr] for gr in res.unique]))
    print('\nLikelihood Ratio Test')
    print('likelihood ratio    p-value       df')
    print(res.lr_test())
    print('Null model: pooled all coefficients are the same across groups,')
    print('Alternative model: all coefficients are allowed to be different')
    print('not verified but looks close to f-test result')
    print('\nOLS parameters by group from individual, separate ols regressions')
    for group in sorted(res.olsbygroup):
        r = res.olsbygroup[group]
        print(group, r.params)
    print('\nCheck for heteroscedasticity, ')
    print('variance and standard deviation for individual regressions')
    print(' ' * 12, ' '.join(('group %-10s' % gr for gr in res.unique)))
    print('variance    ', res.sigmabygroup)
    print('standard dev', np.sqrt(res.sigmabygroup))

def print_results2(res):
    if False:
        return 10
    groupind = res.groups
    ft = res.ftest_summary()
    txt = ''
    templ = "Table of F-tests for overall or pairwise equality of coefficients'\n%(tab)s\n\n\nNotes: p-values are not corrected for many tests\n       (no Bonferroni correction)\n       * : reject at 5%% uncorrected confidence level\nNull hypothesis: all or pairwise coefficient are the same'\nAlternative hypothesis: all coefficients are different'\n\n\nComparison with stats.f_oneway\n%(statsfow)s\n\n\nLikelihood Ratio Test\n%(lrtest)s\nNull model: pooled all coefficients are the same across groups,'\nAlternative model: all coefficients are allowed to be different'\nnot verified but looks close to f-test result'\n\n\nOLS parameters by group from individual, separate ols regressions'\n%(olsbg)s\nfor group in sorted(res.olsbygroup):\n    r = res.olsbygroup[group]\n    print group, r.params\n\n\nCheck for heteroscedasticity, '\nvariance and standard deviation for individual regressions'\n%(grh)s\nvariance    ', res.sigmabygroup\nstandard dev', np.sqrt(res.sigmabygroup)\n"
    from statsmodels.iolib import SimpleTable
    resvals = {}
    resvals['tab'] = str(SimpleTable([['%r' % (row[0],)] + list(row[1]) + ['*'] * (row[1][1] > 0.5).item() for row in ft[1]], headers=['pair', 'F-statistic', 'p-value', 'df_denom', 'df_num']))
    resvals['statsfow'] = str(stats.f_oneway(*[y[groupind == gr] for gr in res.unique]))
    resvals['lrtest'] = str(SimpleTable([res.lr_test()], headers=['likelihood ratio', 'p-value', 'df']))
    resvals['olsbg'] = str(SimpleTable([[group] + res.olsbygroup[group].params.tolist() for group in sorted(res.olsbygroup)]))
    resvals['grh'] = str(SimpleTable(np.vstack([res.sigmabygroup, np.sqrt(res.sigmabygroup)]), headers=res.unique.tolist()))
    return templ % resvals
print('\nTest for equality of coefficients for all exogenous variables')
print('-------------------------------------------------------------')
res = OneWayLS(y, x, groups=groupind.astype(int))
print_results(res)
print('\n\nOne way ANOVA, constant is the only regressor')
print('---------------------------------------------')
print('this is the same as scipy.stats.f_oneway')
res = OneWayLS(y, np.ones(len(y)), groups=groupind)
print_results(res)
print('\n\nOne way ANOVA, constant is the only regressor with het is true')
print('--------------------------------------------------------------')
print('this is the similar to scipy.stats.f_oneway,')
print('but variance is not assumed to be the same across groups')
res = OneWayLS(y, np.ones(len(y)), groups=groupind.astype(str), het=True)
print_results(res)
print(res.print_summary())