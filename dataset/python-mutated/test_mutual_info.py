import numpy as np
import pytest
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.feature_selection._mutual_info import _compute_mi
from sklearn.utils import check_random_state
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.fixes import CSR_CONTAINERS

def test_compute_mi_dd():
    if False:
        print('Hello World!')
    x = np.array([0, 1, 1, 0, 0])
    y = np.array([1, 0, 0, 0, 1])
    H_x = H_y = -(3 / 5) * np.log(3 / 5) - 2 / 5 * np.log(2 / 5)
    H_xy = -1 / 5 * np.log(1 / 5) - 2 / 5 * np.log(2 / 5) - 2 / 5 * np.log(2 / 5)
    I_xy = H_x + H_y - H_xy
    assert_allclose(_compute_mi(x, y, x_discrete=True, y_discrete=True), I_xy)

def test_compute_mi_cc(global_dtype):
    if False:
        print('Hello World!')
    mean = np.zeros(2)
    sigma_1 = 1
    sigma_2 = 10
    corr = 0.5
    cov = np.array([[sigma_1 ** 2, corr * sigma_1 * sigma_2], [corr * sigma_1 * sigma_2, sigma_2 ** 2]])
    I_theory = np.log(sigma_1) + np.log(sigma_2) - 0.5 * np.log(np.linalg.det(cov))
    rng = check_random_state(0)
    Z = rng.multivariate_normal(mean, cov, size=1000).astype(global_dtype, copy=False)
    (x, y) = (Z[:, 0], Z[:, 1])
    for n_neighbors in [3, 5, 7]:
        I_computed = _compute_mi(x, y, x_discrete=False, y_discrete=False, n_neighbors=n_neighbors)
        assert_allclose(I_computed, I_theory, rtol=0.1)

def test_compute_mi_cd(global_dtype):
    if False:
        i = 10
        return i + 15
    n_samples = 1000
    rng = check_random_state(0)
    for p in [0.3, 0.5, 0.7]:
        x = rng.uniform(size=n_samples) > p
        y = np.empty(n_samples, global_dtype)
        mask = x == 0
        y[mask] = rng.uniform(-1, 1, size=np.sum(mask))
        y[~mask] = rng.uniform(0, 2, size=np.sum(~mask))
        I_theory = -0.5 * ((1 - p) * np.log(0.5 * (1 - p)) + p * np.log(0.5 * p) + np.log(0.5)) - np.log(2)
        for n_neighbors in [3, 5, 7]:
            I_computed = _compute_mi(x, y, x_discrete=True, y_discrete=False, n_neighbors=n_neighbors)
            assert_allclose(I_computed, I_theory, rtol=0.1)

def test_compute_mi_cd_unique_label(global_dtype):
    if False:
        return 10
    n_samples = 100
    x = np.random.uniform(size=n_samples) > 0.5
    y = np.empty(n_samples, global_dtype)
    mask = x == 0
    y[mask] = np.random.uniform(-1, 1, size=np.sum(mask))
    y[~mask] = np.random.uniform(0, 2, size=np.sum(~mask))
    mi_1 = _compute_mi(x, y, x_discrete=True, y_discrete=False)
    x = np.hstack((x, 2))
    y = np.hstack((y, 10))
    mi_2 = _compute_mi(x, y, x_discrete=True, y_discrete=False)
    assert_allclose(mi_1, mi_2)

def test_mutual_info_classif_discrete(global_dtype):
    if False:
        for i in range(10):
            print('nop')
    X = np.array([[0, 0, 0], [1, 1, 0], [2, 0, 1], [2, 0, 1], [2, 0, 1]], dtype=global_dtype)
    y = np.array([0, 1, 2, 2, 1])
    mi = mutual_info_classif(X, y, discrete_features=True)
    assert_array_equal(np.argsort(-mi), np.array([0, 2, 1]))

def test_mutual_info_regression(global_dtype):
    if False:
        print('Hello World!')
    T = np.array([[1, 0.5, 2, 1], [0, 1, 0.1, 0.0], [0, 0.1, 1, 0.1], [0, 0.1, 0.1, 1]])
    cov = T.dot(T.T)
    mean = np.zeros(4)
    rng = check_random_state(0)
    Z = rng.multivariate_normal(mean, cov, size=1000).astype(global_dtype, copy=False)
    X = Z[:, 1:]
    y = Z[:, 0]
    mi = mutual_info_regression(X, y, random_state=0)
    assert_array_equal(np.argsort(-mi), np.array([1, 2, 0]))
    assert mi.dtype == np.float64

def test_mutual_info_classif_mixed(global_dtype):
    if False:
        for i in range(10):
            print('nop')
    rng = check_random_state(0)
    X = rng.rand(1000, 3).astype(global_dtype, copy=False)
    X[:, 1] += X[:, 0]
    y = (0.5 * X[:, 0] + X[:, 2] > 0.5).astype(int)
    X[:, 2] = X[:, 2] > 0.5
    mi = mutual_info_classif(X, y, discrete_features=[2], n_neighbors=3, random_state=0)
    assert_array_equal(np.argsort(-mi), [2, 0, 1])
    for n_neighbors in [5, 7, 9]:
        mi_nn = mutual_info_classif(X, y, discrete_features=[2], n_neighbors=n_neighbors, random_state=0)
        assert mi_nn[0] > mi[0]
        assert mi_nn[1] > mi[1]
        assert mi_nn[2] == mi[2]

@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_mutual_info_options(global_dtype, csr_container):
    if False:
        print('Hello World!')
    X = np.array([[0, 0, 0], [1, 1, 0], [2, 0, 1], [2, 0, 1], [2, 0, 1]], dtype=global_dtype)
    y = np.array([0, 1, 2, 2, 1], dtype=global_dtype)
    X_csr = csr_container(X)
    for mutual_info in (mutual_info_regression, mutual_info_classif):
        with pytest.raises(ValueError):
            mutual_info(X_csr, y, discrete_features=False)
        with pytest.raises(ValueError):
            mutual_info(X, y, discrete_features='manual')
        with pytest.raises(ValueError):
            mutual_info(X_csr, y, discrete_features=[True, False, True])
        with pytest.raises(IndexError):
            mutual_info(X, y, discrete_features=[True, False, True, False])
        with pytest.raises(IndexError):
            mutual_info(X, y, discrete_features=[1, 4])
        mi_1 = mutual_info(X, y, discrete_features='auto', random_state=0)
        mi_2 = mutual_info(X, y, discrete_features=False, random_state=0)
        mi_3 = mutual_info(X_csr, y, discrete_features='auto', random_state=0)
        mi_4 = mutual_info(X_csr, y, discrete_features=True, random_state=0)
        mi_5 = mutual_info(X, y, discrete_features=[True, False, True], random_state=0)
        mi_6 = mutual_info(X, y, discrete_features=[0, 2], random_state=0)
        assert_allclose(mi_1, mi_2)
        assert_allclose(mi_3, mi_4)
        assert_allclose(mi_5, mi_6)
        assert not np.allclose(mi_1, mi_3)

@pytest.mark.parametrize('correlated', [True, False])
def test_mutual_information_symmetry_classif_regression(correlated, global_random_seed):
    if False:
        i = 10
        return i + 15
    'Check that `mutual_info_classif` and `mutual_info_regression` are\n    symmetric by switching the target `y` as `feature` in `X` and vice\n    versa.\n\n    Non-regression test for:\n    https://github.com/scikit-learn/scikit-learn/issues/23720\n    '
    rng = np.random.RandomState(global_random_seed)
    n = 100
    d = rng.randint(10, size=n)
    if correlated:
        c = d.astype(np.float64)
    else:
        c = rng.normal(0, 1, size=n)
    mi_classif = mutual_info_classif(c[:, None], d, discrete_features=[False], random_state=global_random_seed)
    mi_regression = mutual_info_regression(d[:, None], c, discrete_features=[True], random_state=global_random_seed)
    assert mi_classif == pytest.approx(mi_regression)

def test_mutual_info_regression_X_int_dtype(global_random_seed):
    if False:
        print('Hello World!')
    'Check that results agree when X is integer dtype and float dtype.\n\n    Non-regression test for Issue #26696.\n    '
    rng = np.random.RandomState(global_random_seed)
    X = rng.randint(100, size=(100, 10))
    X_float = X.astype(np.float64, copy=True)
    y = rng.randint(100, size=100)
    expected = mutual_info_regression(X_float, y, random_state=global_random_seed)
    result = mutual_info_regression(X, y, random_state=global_random_seed)
    assert_allclose(result, expected)