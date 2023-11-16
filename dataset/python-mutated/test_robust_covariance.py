import itertools
import numpy as np
import pytest
from sklearn import datasets
from sklearn.covariance import MinCovDet, empirical_covariance, fast_mcd
from sklearn.utils._testing import assert_array_almost_equal
X = datasets.load_iris().data
X_1d = X[:, 0]
(n_samples, n_features) = X.shape

def test_mcd(global_random_seed):
    if False:
        i = 10
        return i + 15
    launch_mcd_on_dataset(100, 5, 0, 0.02, 0.1, 75, global_random_seed)
    launch_mcd_on_dataset(100, 5, 20, 0.3, 0.3, 65, global_random_seed)
    launch_mcd_on_dataset(100, 5, 40, 0.1, 0.1, 50, global_random_seed)
    launch_mcd_on_dataset(1000, 5, 450, 0.1, 0.1, 540, global_random_seed)
    launch_mcd_on_dataset(1700, 5, 800, 0.1, 0.1, 870, global_random_seed)
    launch_mcd_on_dataset(500, 1, 100, 0.02, 0.02, 350, global_random_seed)

def test_fast_mcd_on_invalid_input():
    if False:
        for i in range(10):
            print('nop')
    X = np.arange(100)
    msg = 'Expected 2D array, got 1D array instead'
    with pytest.raises(ValueError, match=msg):
        fast_mcd(X)

def test_mcd_class_on_invalid_input():
    if False:
        for i in range(10):
            print('nop')
    X = np.arange(100)
    mcd = MinCovDet()
    msg = 'Expected 2D array, got 1D array instead'
    with pytest.raises(ValueError, match=msg):
        mcd.fit(X)

def launch_mcd_on_dataset(n_samples, n_features, n_outliers, tol_loc, tol_cov, tol_support, seed):
    if False:
        while True:
            i = 10
    rand_gen = np.random.RandomState(seed)
    data = rand_gen.randn(n_samples, n_features)
    outliers_index = rand_gen.permutation(n_samples)[:n_outliers]
    outliers_offset = 10.0 * (rand_gen.randint(2, size=(n_outliers, n_features)) - 0.5)
    data[outliers_index] += outliers_offset
    inliers_mask = np.ones(n_samples).astype(bool)
    inliers_mask[outliers_index] = False
    pure_data = data[inliers_mask]
    mcd_fit = MinCovDet(random_state=seed).fit(data)
    T = mcd_fit.location_
    S = mcd_fit.covariance_
    H = mcd_fit.support_
    error_location = np.mean((pure_data.mean(0) - T) ** 2)
    assert error_location < tol_loc
    error_cov = np.mean((empirical_covariance(pure_data) - S) ** 2)
    assert error_cov < tol_cov
    assert np.sum(H) >= tol_support
    assert_array_almost_equal(mcd_fit.mahalanobis(data), mcd_fit.dist_)

def test_mcd_issue1127():
    if False:
        while True:
            i = 10
    rnd = np.random.RandomState(0)
    X = rnd.normal(size=(3, 1))
    mcd = MinCovDet()
    mcd.fit(X)

def test_mcd_issue3367(global_random_seed):
    if False:
        print('Hello World!')
    rand_gen = np.random.RandomState(global_random_seed)
    data_values = np.linspace(-5, 5, 10).tolist()
    data = np.array(list(itertools.product(data_values, data_values)))
    data = np.hstack((data, np.zeros((data.shape[0], 1))))
    MinCovDet(random_state=rand_gen).fit(data)

def test_mcd_support_covariance_is_zero():
    if False:
        for i in range(10):
            print('nop')
    X_1 = np.array([0.5, 0.1, 0.1, 0.1, 0.957, 0.1, 0.1, 0.1, 0.4285, 0.1])
    X_1 = X_1.reshape(-1, 1)
    X_2 = np.array([0.5, 0.3, 0.3, 0.3, 0.957, 0.3, 0.3, 0.3, 0.4285, 0.3])
    X_2 = X_2.reshape(-1, 1)
    msg = 'The covariance matrix of the support data is equal to 0, try to increase support_fraction'
    for X in [X_1, X_2]:
        with pytest.raises(ValueError, match=msg):
            MinCovDet().fit(X)

def test_mcd_increasing_det_warning(global_random_seed):
    if False:
        while True:
            i = 10
    X = [[5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2], [4.7, 3.2, 1.3, 0.2], [4.6, 3.1, 1.5, 0.2], [5.0, 3.6, 1.4, 0.2], [4.6, 3.4, 1.4, 0.3], [5.0, 3.4, 1.5, 0.2], [4.4, 2.9, 1.4, 0.2], [4.9, 3.1, 1.5, 0.1], [5.4, 3.7, 1.5, 0.2], [4.8, 3.4, 1.6, 0.2], [4.8, 3.0, 1.4, 0.1], [4.3, 3.0, 1.1, 0.1], [5.1, 3.5, 1.4, 0.3], [5.7, 3.8, 1.7, 0.3], [5.4, 3.4, 1.7, 0.2], [4.6, 3.6, 1.0, 0.2], [5.0, 3.0, 1.6, 0.2], [5.2, 3.5, 1.5, 0.2]]
    mcd = MinCovDet(support_fraction=0.5, random_state=global_random_seed)
    warn_msg = 'Determinant has increased'
    with pytest.warns(RuntimeWarning, match=warn_msg):
        mcd.fit(X)