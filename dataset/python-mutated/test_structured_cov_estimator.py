import unittest
import numpy as np
from scipy.linalg import sqrtm
from qlib.model.riskmodel import StructuredCovEstimator

class TestStructuredCovEstimator(unittest.TestCase):

    def test_random_covariance(self):
        if False:
            return 10
        NUM_VARIABLE = 10
        NUM_OBSERVATION = 200
        EPS = 1e-06
        estimator = StructuredCovEstimator(scale_return=False, assume_centered=True)
        X = np.random.rand(NUM_OBSERVATION, NUM_VARIABLE)
        est_cov = estimator.predict(X, is_price=False)
        np_cov = np.cov(X.T)
        delta = abs(est_cov - np_cov)
        if_identical = (delta < EPS).all()
        self.assertTrue(if_identical)

    def test_nan_option_covariance(self):
        if False:
            return 10
        NUM_VARIABLE = 10
        NUM_OBSERVATION = 200
        EPS = 1e-06
        estimator = StructuredCovEstimator(scale_return=False, assume_centered=True, nan_option='fill')
        X = np.random.rand(NUM_OBSERVATION, NUM_VARIABLE)
        est_cov = estimator.predict(X, is_price=False)
        np_cov = np.cov(X.T)
        delta = abs(est_cov - np_cov)
        if_identical = (delta < EPS).all()
        self.assertTrue(if_identical)

    def test_decompose_covariance(self):
        if False:
            print('Hello World!')
        NUM_VARIABLE = 10
        NUM_OBSERVATION = 200
        estimator = StructuredCovEstimator(scale_return=False, assume_centered=True, nan_option='fill')
        X = np.random.rand(NUM_OBSERVATION, NUM_VARIABLE)
        (F, cov_b, var_u) = estimator.predict(X, is_price=False, return_decomposed_components=True)
        self.assertTrue(F is not None and cov_b is not None and (var_u is not None))

    def test_constructed_covariance(self):
        if False:
            print('Hello World!')
        NUM_VARIABLE = 7
        NUM_OBSERVATION = 500
        EPS = 0.1
        estimator = StructuredCovEstimator(scale_return=False, assume_centered=True, num_factors=NUM_VARIABLE - 1)
        sqrt_cov = None
        while sqrt_cov is None or np.iscomplex(sqrt_cov).any():
            cov = np.random.rand(NUM_VARIABLE, NUM_VARIABLE)
            for i in range(NUM_VARIABLE):
                cov[i][i] = 1
            sqrt_cov = sqrtm(cov)
        X = np.random.rand(NUM_OBSERVATION, NUM_VARIABLE) @ sqrt_cov
        est_cov = estimator.predict(X, is_price=False)
        np_cov = np.cov(X.T)
        delta = abs(est_cov - np_cov)
        if_identical = (delta < EPS).all()
        self.assertTrue(if_identical)

    def test_decomposition(self):
        if False:
            return 10
        NUM_VARIABLE = 30
        NUM_OBSERVATION = 100
        NUM_FACTOR = 10
        EPS = 0.1
        estimator = StructuredCovEstimator(scale_return=False, assume_centered=True, num_factors=NUM_FACTOR)
        F = np.random.rand(NUM_VARIABLE, NUM_FACTOR)
        B = np.random.rand(NUM_FACTOR, NUM_OBSERVATION)
        U = np.random.rand(NUM_OBSERVATION, NUM_VARIABLE)
        X = (F @ B).T + U
        est_cov = estimator.predict(X, is_price=False)
        np_cov = np.cov(X.T)
        delta = abs(est_cov - np_cov)
        if_identical = (delta < EPS).all()
        self.assertTrue(if_identical)
if __name__ == '__main__':
    unittest.main()