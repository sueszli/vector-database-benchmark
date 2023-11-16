"""

Created on Mon Dec 10 09:18:14 2012

Author: Josef Perktold
"""
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal, assert_allclose
from statsmodels.stats.inter_rater import fleiss_kappa, cohens_kappa, to_table, aggregate_raters
from statsmodels.tools.testing import Holder
table0 = np.asarray('1 \t0 \t0 \t0 \t0 \t14 \t1.000\n2 \t0 \t2 \t6 \t4 \t2 \t0.253\n3 \t0 \t0 \t3 \t5 \t6 \t0.308\n4 \t0 \t3 \t9 \t2 \t0 \t0.440\n5 \t2 \t2 \t8 \t1 \t1 \t0.330\n6 \t7 \t7 \t0 \t0 \t0 \t0.462\n7 \t3 \t2 \t6 \t3 \t0 \t0.242\n8 \t2 \t5 \t3 \t2 \t2 \t0.176\n9 \t6 \t5 \t2 \t1 \t0 \t0.286\n10 \t0 \t2 \t2 \t3 \t7 \t0.286'.split(), float).reshape(10, -1)
table1 = table0[:, 1:-1]
table10 = [[0, 4, 1], [0, 8, 0], [0, 1, 5]]
diagnoses = np.array([[4, 4, 4, 4, 4, 4], [2, 2, 2, 5, 5, 5], [2, 3, 3, 3, 3, 5], [5, 5, 5, 5, 5, 5], [2, 2, 2, 4, 4, 4], [1, 1, 3, 3, 3, 3], [3, 3, 3, 3, 5, 5], [1, 1, 3, 3, 3, 4], [1, 1, 4, 4, 4, 4], [5, 5, 5, 5, 5, 5], [1, 4, 4, 4, 4, 4], [1, 2, 4, 4, 4, 4], [2, 2, 2, 3, 3, 3], [1, 4, 4, 4, 4, 4], [2, 2, 4, 4, 4, 5], [3, 3, 3, 3, 3, 5], [1, 1, 1, 4, 5, 5], [1, 1, 1, 1, 1, 2], [2, 2, 4, 4, 4, 4], [1, 3, 3, 5, 5, 5], [5, 5, 5, 5, 5, 5], [2, 4, 4, 4, 4, 4], [2, 2, 4, 5, 5, 5], [1, 1, 4, 4, 4, 4], [1, 4, 4, 4, 4, 5], [2, 2, 2, 2, 2, 4], [1, 1, 1, 1, 5, 5], [2, 2, 4, 4, 4, 4], [1, 3, 3, 3, 3, 3], [5, 5, 5, 5, 5, 5]])
diagnoses_rownames = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30']
diagnoses_colnames = ['rater1', 'rater2', 'rater3', 'rater4', 'rater5', 'rater6']

def test_fleiss_kappa():
    if False:
        print('Hello World!')
    kappa_wp = 0.21
    assert_almost_equal(fleiss_kappa(table1), kappa_wp, decimal=3)

def test_fleis_randolph():
    if False:
        i = 10
        return i + 15
    table = [[7, 0], [7, 0]]
    assert_equal(fleiss_kappa(table, method='unif'), 1)
    table = [[6.99, 0.01], [6.99, 0.01]]
    assert_allclose(fleiss_kappa(table), -0.166667, atol=6e-06)
    assert_allclose(fleiss_kappa(table, method='unif'), 0.993343, atol=6e-06)
    table = [[7, 1], [3, 5]]
    assert_allclose(fleiss_kappa(table, method='fleiss'), 0.161905, atol=6e-06)
    assert_allclose(fleiss_kappa(table, method='randolph'), 0.214286, atol=6e-06)
    table = [[7, 0], [0, 7]]
    assert_allclose(fleiss_kappa(table), 1)
    assert_allclose(fleiss_kappa(table, method='uniform'), 1)
    table = [[6, 1, 0], [0, 7, 0]]
    assert_allclose(fleiss_kappa(table), 0.708333, atol=6e-06)
    assert_allclose(fleiss_kappa(table, method='rand'), 0.785714, atol=6e-06)

class CheckCohens:

    def test_results(self):
        if False:
            while True:
                i = 10
        res = self.res
        res2 = self.res2
        res_ = [res.kappa, res.std_kappa, res.kappa_low, res.kappa_upp, res.std_kappa0, res.z_value, res.pvalue_one_sided, res.pvalue_two_sided]
        assert_almost_equal(res_, res2, decimal=4)
        assert_equal(str(res), self.res_string)

class TestUnweightedCohens(CheckCohens):

    @classmethod
    def setup_class(cls):
        if False:
            for i in range(10):
                print('nop')
        cls.res = cohens_kappa(table10)
        res10_sas = [0.4842, 0.138, 0.2137, 0.7547]
        res10_sash0 = [0.1484, 3.2626, 0.0006, 0.0011]
        cls.res2 = res10_sas + res10_sash0
        cls.res_string = '                  Simple Kappa Coefficient\n              --------------------------------\n              Kappa                     0.4842\n              ASE                       0.1380\n              95% Lower Conf Limit      0.2137\n              95% Upper Conf Limit      0.7547\n\n                 Test of H0: Simple Kappa = 0\n\n              ASE under H0              0.1484\n              Z                         3.2626\n              One-sided Pr >  Z         0.0006\n              Two-sided Pr > |Z|        0.0011' + '\n'

    def test_option(self):
        if False:
            i = 10
            return i + 15
        kappa = cohens_kappa(table10, return_results=False)
        assert_almost_equal(kappa, self.res2[0], decimal=4)

class TestWeightedCohens(CheckCohens):

    @classmethod
    def setup_class(cls):
        if False:
            for i in range(10):
                print('nop')
        cls.res = cohens_kappa(table10, weights=[0, 1, 2])
        res10w_sas = [0.4701, 0.1457, 0.1845, 0.7558]
        res10w_sash0 = [0.1426, 3.2971, 0.0005, 0.001]
        cls.res2 = res10w_sas + res10w_sash0
        cls.res_string = '                  Weighted Kappa Coefficient\n              --------------------------------\n              Kappa                     0.4701\n              ASE                       0.1457\n              95% Lower Conf Limit      0.1845\n              95% Upper Conf Limit      0.7558\n\n                 Test of H0: Weighted Kappa = 0\n\n              ASE under H0              0.1426\n              Z                         3.2971\n              One-sided Pr >  Z         0.0005\n              Two-sided Pr > |Z|        0.0010' + '\n'

    def test_option(self):
        if False:
            while True:
                i = 10
        kappa = cohens_kappa(table10, weights=[0, 1, 2], return_results=False)
        assert_almost_equal(kappa, self.res2[0], decimal=4)

def test_cohenskappa_weights():
    if False:
        return 10
    np.random.seed(9743678)
    table = np.random.randint(0, 10, size=(5, 5)) + 5 * np.eye(5)
    mat = np.array([[1, 1, 1, 0, 0], [0, 0, 0, 1, 1]])
    table_agg = np.dot(np.dot(mat, table), mat.T)
    res1 = cohens_kappa(table, weights=np.arange(5) > 2, wt='linear')
    res2 = cohens_kappa(table_agg, weights=np.arange(2), wt='linear')
    assert_almost_equal(res1.kappa, res2.kappa, decimal=14)
    assert_almost_equal(res1.var_kappa, res2.var_kappa, decimal=14)
    res1 = cohens_kappa(table, weights=2 * np.arange(5), wt='linear')
    res2 = cohens_kappa(table, weights=2 * np.arange(5), wt='toeplitz')
    res3 = cohens_kappa(table, weights=res1.weights[0], wt='toeplitz')
    res4 = cohens_kappa(table, weights=res1.weights)
    assert_almost_equal(res1.kappa, res2.kappa, decimal=14)
    assert_almost_equal(res1.var_kappa, res2.var_kappa, decimal=14)
    assert_almost_equal(res1.kappa, res3.kappa, decimal=14)
    assert_almost_equal(res1.var_kappa, res3.var_kappa, decimal=14)
    assert_almost_equal(res1.kappa, res4.kappa, decimal=14)
    assert_almost_equal(res1.var_kappa, res4.var_kappa, decimal=14)
    res1 = cohens_kappa(table, weights=5 * np.arange(5) ** 2, wt='toeplitz')
    res2 = cohens_kappa(table, weights=5 * np.arange(5), wt='quadratic')
    assert_almost_equal(res1.kappa, res2.kappa, decimal=14)
    assert_almost_equal(res1.var_kappa, res2.var_kappa, decimal=14)
anxiety = np.array([3, 3, 3, 4, 5, 5, 2, 3, 5, 2, 2, 6, 1, 5, 2, 2, 1, 2, 4, 3, 3, 6, 4, 6, 2, 4, 2, 4, 3, 3, 2, 3, 3, 3, 2, 2, 1, 3, 3, 4, 2, 1, 4, 4, 3, 2, 1, 6, 1, 1, 1, 2, 3, 3, 1, 1, 3, 3, 2, 2]).reshape(20, 3, order='F')
anxiety_rownames = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']
anxiety_colnames = ['rater1', 'rater2', 'rater3']

def test_cohens_kappa_irr():
    if False:
        for i in range(10):
            print('nop')
    ck_w3 = Holder()
    ck_w4 = Holder()
    ck_w3.method = "Cohen's Kappa for 2 Raters (Weights: 0,0,0,1,1,1)"
    ck_w3.irr_name = 'Kappa'
    ck_w3.value = 0.1891892
    ck_w3.stat_name = 'z'
    ck_w3.statistic = 0.5079002
    ck_w3.p_value = 0.6115233
    ck_w4.method = "Cohen's Kappa for 2 Raters (Weights: 0,0,1,1,2,2)"
    ck_w4.irr_name = 'Kappa'
    ck_w4.value = 0.2820513
    ck_w4.stat_name = 'z'
    ck_w4.statistic = 1.25741
    ck_w4.p_value = 0.2086053
    ck_w1 = Holder()
    ck_w2 = Holder()
    ck_w3 = Holder()
    ck_w4 = Holder()
    ck_w1.method = "Cohen's Kappa for 2 Raters (Weights: unweighted)"
    ck_w1.irr_name = 'Kappa'
    ck_w1.value = -0.006289308
    ck_w1.stat_name = 'z'
    ck_w1.statistic = -0.0604067
    ck_w1.p_value = 0.9518317
    ck_w2.method = "Cohen's Kappa for 2 Raters (Weights: equal)"
    ck_w2.irr_name = 'Kappa'
    ck_w2.value = 0.1459075
    ck_w2.stat_name = 'z'
    ck_w2.statistic = 1.282472
    ck_w2.p_value = 0.1996772
    ck_w3.method = "Cohen's Kappa for 2 Raters (Weights: squared)"
    ck_w3.irr_name = 'Kappa'
    ck_w3.value = 0.2520325
    ck_w3.stat_name = 'z'
    ck_w3.statistic = 1.437451
    ck_w3.p_value = 0.1505898
    ck_w4.method = "Cohen's Kappa for 2 Raters (Weights: 0,0,1,1,2)"
    ck_w4.irr_name = 'Kappa'
    ck_w4.value = 0.2391304
    ck_w4.stat_name = 'z'
    ck_w4.statistic = 1.223734
    ck_w4.p_value = 0.2210526
    all_cases = [(ck_w1, None, None), (ck_w2, None, 'linear'), (ck_w2, np.arange(5), None), (ck_w2, np.arange(5), 'toeplitz'), (ck_w3, None, 'quadratic'), (ck_w3, np.arange(5) ** 2, 'toeplitz'), (ck_w3, 4 * np.arange(5) ** 2, 'toeplitz'), (ck_w4, [0, 0, 1, 1, 2], 'toeplitz')]
    r = np.histogramdd(anxiety[:, 1:], ([1, 2, 3, 4, 6, 7], [1, 2, 3, 4, 6, 7]))
    for (res2, w, wt) in all_cases:
        msg = repr(w) + repr(wt)
        res1 = cohens_kappa(r[0], weights=w, wt=wt)
        assert_almost_equal(res1.kappa, res2.value, decimal=6, err_msg=msg)
        assert_almost_equal(res1.z_value, res2.statistic, decimal=5, err_msg=msg)
        assert_almost_equal(res1.pvalue_two_sided, res2.p_value, decimal=6, err_msg=msg)

def test_fleiss_kappa_irr():
    if False:
        while True:
            i = 10
    fleiss = Holder()
    fleiss.method = "Fleiss' Kappa for m Raters"
    fleiss.irr_name = 'Kappa'
    fleiss.value = 0.4302445
    fleiss.stat_name = 'z'
    fleiss.statistic = 17.65183
    fleiss.p_value = 0
    (data_, _) = aggregate_raters(diagnoses)
    res1_kappa = fleiss_kappa(data_)
    assert_almost_equal(res1_kappa, fleiss.value, decimal=7)

def test_to_table():
    if False:
        return 10
    data = diagnoses
    res1 = to_table(data[:, :2] - 1, 5)
    res0 = np.asarray([[(data[:, :2] - 1 == [i, j]).all(1).sum() for j in range(5)] for i in range(5)])
    assert_equal(res1[0], res0)
    res2 = to_table(data[:, :2])
    assert_equal(res2[0], res0)
    bins = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    res3 = to_table(data[:, :2], bins)
    assert_equal(res3[0], res0)
    res4 = to_table(data[:, :3] - 1, bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5])
    res5 = to_table(data[:, :3] - 1, bins=5)
    assert_equal(res4[0].sum(-1), res0)
    assert_equal(res5[0].sum(-1), res0)

def test_aggregate_raters():
    if False:
        i = 10
        return i + 15
    data = diagnoses
    (data_, categories) = aggregate_raters(data)
    colsum = np.array([26, 26, 30, 55, 43])
    assert_equal(data_.sum(0), colsum)
    assert_equal(np.unique(diagnoses), categories)