from __future__ import division
from __future__ import print_function
import os
import sys
import unittest
import numpy as np
from numpy.testing import assert_allclose
from numpy.testing import assert_equal
from numpy.testing import assert_raises
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pyod.utils.stat_models import pairwise_distances_no_broadcast
from pyod.utils.stat_models import wpearsonr
from pyod.utils.stat_models import pearsonr_mat
from pyod.utils.stat_models import column_ecdf
import statsmodels.distributions
import time

class TestStatModels(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.a = [1, 2, 3, 2, 3, 1, 0, 5]
        self.b = [1, 2, 1, 2, 2, 1, 0, 2]
        self.w = [2, 2, 1, 2, 4, 1, 0, 2]
        self.mat = np.random.rand(10, 20)
        self.w_mat = np.random.rand(10, 1)
        self.X = np.array([[1, 2, 3], [3, 4, 5], [3, 6, 7], [4, 1, 1]])
        self.Y = np.array([[2, 2, 2], [3, 3, 3], [4, 4, 3], [0, 1, 2]])

    def test_pairwise_distances_no_broadcast(self):
        if False:
            print('Hello World!')
        assert_allclose(pairwise_distances_no_broadcast(self.X, self.Y), [1.41421356, 2.23606798, 4.58257569, 4.12310563])
        with assert_raises(ValueError):
            pairwise_distances_no_broadcast([1, 2, 3], [6])

    def test_wpearsonr(self):
        if False:
            while True:
                i = 10
        wp = wpearsonr(self.a, self.b)
        assert_allclose(wp[0], 0.6956083, atol=0.01)
        wp = wpearsonr(self.a, self.b, w=self.w)
        assert_allclose(wp, 0.5477226, atol=0.01)

    def test_pearsonr_mat(self):
        if False:
            return 10
        pear_mat = pearsonr_mat(self.mat)
        assert_equal(pear_mat.shape, (10, 10))
        pear_mat = pearsonr_mat(self.mat, self.w_mat)
        assert_equal(pear_mat.shape, (10, 10))
        assert np.min(pear_mat) >= -1
        assert np.max(pear_mat) <= 1

    def test_njit_probability_reordering(self):
        if False:
            while True:
                i = 10
        column_ecdf(self.mat)

    def test_column_ecdf(self):
        if False:
            return 10

        def ecdf(X):
            if False:
                return 10
            'Calculated the empirical CDF of a given dataset using the statsmodels function.\n            Parameters\n            ----------\n            X : numpy array of shape (n_samples, n_features)\n                The training dataset.\n            Returns\n            -------\n            ecdf(X) : float\n                Empirical CDF of X\n            '
            ecdf = statsmodels.distributions.ECDF(X)
            return ecdf(X)
        mat = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1], [2, 2, 2]])
        assert_equal(column_ecdf(mat), np.apply_along_axis(ecdf, 0, mat))
        new = []
        old = []
        for _ in range(50):
            mat = np.random.rand(1000, 100)
            t = time.time()
            result = column_ecdf(mat)
            new.append(time.time() - t)
            t = time.time()
            expected = np.apply_along_axis(ecdf, 0, mat)
            old.append(time.time() - t)
            assert_equal(result, expected)

    def tearDown(self):
        if False:
            print('Hello World!')
        pass
if __name__ == '__main__':
    unittest.main()