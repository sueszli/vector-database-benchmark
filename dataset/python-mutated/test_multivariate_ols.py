import numpy as np
import pandas as pd
from statsmodels.multivariate.multivariate_ols import _MultivariateOLS
from numpy.testing import assert_array_almost_equal, assert_raises
import patsy
data = pd.DataFrame([['Morphine', 'N', 0.04, 0.2, 0.1, 0.08], ['Morphine', 'N', 0.02, 0.06, 0.02, 0.02], ['Morphine', 'N', 0.07, 1.4, 0.48, 0.24], ['Morphine', 'N', 0.17, 0.57, 0.35, 0.24], ['Morphine', 'Y', 0.1, 0.09, 0.13, 0.14], ['placebo', 'Y', 0.07, 0.07, 0.06, 0.07], ['placebo', 'Y', 0.05, 0.07, 0.06, 0.07], ['placebo', 'N', 0.03, 0.62, 0.31, 0.22], ['placebo', 'N', 0.03, 1.05, 0.73, 0.6], ['placebo', 'N', 0.07, 0.83, 1.07, 0.8], ['Trimethaphan', 'N', 0.09, 3.13, 2.06, 1.23], ['Trimethaphan', 'Y', 0.1, 0.09, 0.09, 0.08], ['Trimethaphan', 'Y', 0.08, 0.09, 0.09, 0.1], ['Trimethaphan', 'Y', 0.13, 0.1, 0.12, 0.12], ['Trimethaphan', 'Y', 0.06, 0.05, 0.05, 0.05]], columns=['Drug', 'Depleted', 'Histamine0', 'Histamine1', 'Histamine3', 'Histamine5'])
for i in range(2, 6):
    data.iloc[:, i] = np.log(data.iloc[:, i])

def compare_r_output_dogs_data(method):
    if False:
        print('Hello World!')
    ' Testing within-subject effect interact with 2 between-subject effect\n    Compares with R car library Anova(, type=3) output\n\n    Note: The test statistis Phillai, Wilks, Hotelling-Lawley\n          and Roy are the same as R output but the approximate F and degree\n          of freedoms can be different. This is due to the fact that this\n          implementation is based on SAS formula [1]\n\n    .. [*] https://support.sas.com/documentation/cdl/en/statug/63033/HTML/default/viewer.htm#statug_introreg_sect012.htm\n    '
    mod = _MultivariateOLS.from_formula('Histamine0 + Histamine1 + Histamine3 + Histamine5 ~ Drug * Depleted', data)
    r = mod.fit(method=method)
    r = r.mv_test()
    a = [[0.026860766, 4, 6, 54.3435304, 7.5958561e-05], [0.973139234, 4, 6, 54.3435304, 7.5958561e-05], [36.2290202, 4, 6, 54.3435304, 7.5958561e-05], [36.2290202, 4, 6, 54.3435304, 7.5958561e-05]]
    assert_array_almost_equal(r['Intercept']['stat'].values, a, decimal=6)
    a = [[0.0839646619, 8, 12.0, 3.67658068, 0.0212614444], [1.18605382, 8, 14.0, 2.55003861, 0.0601270701], [7.69391362, 8, 6.63157895, 5.5081427, 0.020739226], [7.25036952, 4, 7.0, 12.6881467, 0.00252669877]]
    assert_array_almost_equal(r['Drug']['stat'].values, a, decimal=6)
    a = [[0.32048892, 4.0, 6.0, 3.18034906, 0.10002373], [0.67951108, 4.0, 6.0, 3.18034906, 0.10002373], [2.12023271, 4.0, 6.0, 3.18034906, 0.10002373], [2.12023271, 4.0, 6.0, 3.18034906, 0.10002373]]
    assert_array_almost_equal(r['Depleted']['stat'].values, a, decimal=6)
    a = [[0.15234366, 8.0, 12.0, 2.34307678, 0.08894239], [1.13013353, 8.0, 14.0, 2.27360606, 0.08553213], [3.70989596, 8.0, 6.63157895, 2.65594824, 0.11370285], [3.1145597, 4.0, 7.0, 5.45047947, 0.02582767]]
    assert_array_almost_equal(r['Drug:Depleted']['stat'].values, a, decimal=6)

def test_glm_dogs_example():
    if False:
        i = 10
        return i + 15
    compare_r_output_dogs_data(method='svd')
    compare_r_output_dogs_data(method='pinv')

def test_specify_L_M_by_string():
    if False:
        return 10
    mod = _MultivariateOLS.from_formula('Histamine0 + Histamine1 + Histamine3 + Histamine5 ~ Drug * Depleted', data)
    r = mod.fit()
    r1 = r.mv_test(hypotheses=[['Intercept', ['Intercept'], None]])
    a = [[0.026860766, 4, 6, 54.3435304, 7.5958561e-05], [0.973139234, 4, 6, 54.3435304, 7.5958561e-05], [36.2290202, 4, 6, 54.3435304, 7.5958561e-05], [36.2290202, 4, 6, 54.3435304, 7.5958561e-05]]
    assert_array_almost_equal(r1['Intercept']['stat'].values, a, decimal=6)
    L = ['Intercept', 'Drug[T.Trimethaphan]', 'Drug[T.placebo]']
    M = ['Histamine1', 'Histamine3', 'Histamine5']
    r1 = r.mv_test(hypotheses=[['a', L, M]])
    a = [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]]
    assert_array_almost_equal(r1['a']['contrast_L'], a, decimal=10)
    a = [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    assert_array_almost_equal(r1['a']['transform_M'].T, a, decimal=10)

def test_independent_variable_singular():
    if False:
        for i in range(10):
            print('nop')
    data1 = data.copy()
    data1['dup'] = data1['Drug']
    mod = _MultivariateOLS.from_formula('Histamine0 + Histamine1 + Histamine3 + Histamine5 ~ Drug * dup', data1)
    assert_raises(ValueError, mod.fit)
    mod = _MultivariateOLS.from_formula('Histamine0 + Histamine1 + Histamine3 + Histamine5 ~ Drug * dup', data1)
    assert_raises(ValueError, mod.fit)

def test_from_formula_vs_no_formula():
    if False:
        while True:
            i = 10
    mod = _MultivariateOLS.from_formula('Histamine0 + Histamine1 + Histamine3 + Histamine5 ~ Drug * Depleted', data)
    r = mod.fit(method='svd')
    r0 = r.mv_test()
    (endog, exog) = patsy.dmatrices('Histamine0 + Histamine1 + Histamine3 + Histamine5 ~ Drug * Depleted', data, return_type='dataframe')
    L = np.array([[1, 0, 0, 0, 0, 0]])
    r = _MultivariateOLS(endog, exog).fit(method='svd')
    r1 = r.mv_test(hypotheses=[['Intercept', L, None]])
    assert_array_almost_equal(r1['Intercept']['stat'].values, r0['Intercept']['stat'].values, decimal=6)
    r = _MultivariateOLS(endog.values, exog.values).fit(method='svd')
    r1 = r.mv_test(hypotheses=[['Intercept', L, None]])
    assert_array_almost_equal(r1['Intercept']['stat'].values, r0['Intercept']['stat'].values, decimal=6)
    L = np.array([[0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]])
    r1 = r.mv_test(hypotheses=[['Drug', L, None]])
    r = _MultivariateOLS(endog, exog).fit(method='svd')
    r1 = r.mv_test(hypotheses=[['Drug', L, None]])
    assert_array_almost_equal(r1['Drug']['stat'].values, r0['Drug']['stat'].values, decimal=6)
    r = _MultivariateOLS(endog.values, exog.values).fit(method='svd')
    r1 = r.mv_test(hypotheses=[['Drug', L, None]])
    assert_array_almost_equal(r1['Drug']['stat'].values, r0['Drug']['stat'].values, decimal=6)

def test_L_M_matrices_1D_array():
    if False:
        i = 10
        return i + 15
    mod = _MultivariateOLS.from_formula('Histamine0 + Histamine1 + Histamine3 + Histamine5 ~ Drug * Depleted', data)
    r = mod.fit(method='svd')
    L = np.array([1, 0, 0, 0, 0, 0])
    assert_raises(ValueError, r.mv_test, hypotheses=[['Drug', L, None]])
    L = np.array([[1, 0, 0, 0, 0, 0]])
    M = np.array([1, 0, 0, 0, 0, 0])
    assert_raises(ValueError, r.mv_test, hypotheses=[['Drug', L, M]])

def test_exog_1D_array():
    if False:
        while True:
            i = 10
    mod = _MultivariateOLS.from_formula('Histamine0 + Histamine1 + Histamine3 + Histamine5 ~ 0 + Depleted', data)
    r = mod.fit(method='svd')
    r0 = r.mv_test()
    a = [[0.0019, 8.0, 20.0, 55.0013, 0.0], [1.8112, 8.0, 22.0, 26.3796, 0.0], [97.8858, 8.0, 12.1818, 117.1133, 0.0], [93.2742, 4.0, 11.0, 256.5041, 0.0]]
    assert_array_almost_equal(r0['Depleted']['stat'].values, a, decimal=4)

def test_endog_1D_array():
    if False:
        while True:
            i = 10
    assert_raises(ValueError, _MultivariateOLS.from_formula, 'Histamine0 ~ 0 + Depleted', data)

def test_affine_hypothesis():
    if False:
        i = 10
        return i + 15
    mod = _MultivariateOLS.from_formula('Histamine0 + Histamine1 + Histamine3 + Histamine5 ~ Drug * Depleted', data)
    r = mod.fit(method='svd')
    L = np.array([[0, 1.2, 1.1, 1.3, 1.5, 1.4], [0, 3.2, 2.1, 3.3, 5.5, 4.4]])
    M = None
    C = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    r0 = r.mv_test(hypotheses=[('test1', L, M, C)])
    a = [[0.0269, 8.0, 12.0, 7.6441, 0.001], [1.4277, 8.0, 14.0, 4.3657, 0.008], [19.2678, 8.0, 6.6316, 13.794, 0.0016], [18.347, 4.0, 7.0, 32.1072, 0.0001]]
    assert_array_almost_equal(r0['test1']['stat'].values, a, decimal=4)
    r0.summary(show_contrast_L=True, show_transform_M=True, show_constant_C=True)