"""Tests for Incremental PCA."""
import warnings
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn import datasets
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.utils._testing import assert_allclose_dense_sparse, assert_almost_equal, assert_array_almost_equal
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS, LIL_CONTAINERS
iris = datasets.load_iris()

def test_incremental_pca():
    if False:
        print('Hello World!')
    X = iris.data
    batch_size = X.shape[0] // 3
    ipca = IncrementalPCA(n_components=2, batch_size=batch_size)
    pca = PCA(n_components=2)
    pca.fit_transform(X)
    X_transformed = ipca.fit_transform(X)
    assert X_transformed.shape == (X.shape[0], 2)
    np.testing.assert_allclose(ipca.explained_variance_ratio_.sum(), pca.explained_variance_ratio_.sum(), rtol=0.001)
    for n_components in [1, 2, X.shape[1]]:
        ipca = IncrementalPCA(n_components, batch_size=batch_size)
        ipca.fit(X)
        cov = ipca.get_covariance()
        precision = ipca.get_precision()
        np.testing.assert_allclose(np.dot(cov, precision), np.eye(X.shape[1]), atol=1e-13)

@pytest.mark.parametrize('sparse_container', CSC_CONTAINERS + CSR_CONTAINERS + LIL_CONTAINERS)
def test_incremental_pca_sparse(sparse_container):
    if False:
        i = 10
        return i + 15
    X = iris.data
    pca = PCA(n_components=2)
    pca.fit_transform(X)
    X_sparse = sparse_container(X)
    batch_size = X_sparse.shape[0] // 3
    ipca = IncrementalPCA(n_components=2, batch_size=batch_size)
    X_transformed = ipca.fit_transform(X_sparse)
    assert X_transformed.shape == (X_sparse.shape[0], 2)
    np.testing.assert_allclose(ipca.explained_variance_ratio_.sum(), pca.explained_variance_ratio_.sum(), rtol=0.001)
    for n_components in [1, 2, X.shape[1]]:
        ipca = IncrementalPCA(n_components, batch_size=batch_size)
        ipca.fit(X_sparse)
        cov = ipca.get_covariance()
        precision = ipca.get_precision()
        np.testing.assert_allclose(np.dot(cov, precision), np.eye(X_sparse.shape[1]), atol=1e-13)
    with pytest.raises(TypeError, match='IncrementalPCA.partial_fit does not support sparse input. Either convert data to dense or use IncrementalPCA.fit to do so in batches.'):
        ipca.partial_fit(X_sparse)

def test_incremental_pca_check_projection():
    if False:
        i = 10
        return i + 15
    rng = np.random.RandomState(1999)
    (n, p) = (100, 3)
    X = rng.randn(n, p) * 0.1
    X[:10] += np.array([3, 4, 5])
    Xt = 0.1 * rng.randn(1, p) + np.array([3, 4, 5])
    Yt = IncrementalPCA(n_components=2).fit(X).transform(Xt)
    Yt /= np.sqrt((Yt ** 2).sum())
    assert_almost_equal(np.abs(Yt[0][0]), 1.0, 1)

def test_incremental_pca_inverse():
    if False:
        for i in range(10):
            print('nop')
    rng = np.random.RandomState(1999)
    (n, p) = (50, 3)
    X = rng.randn(n, p)
    X[:, 1] *= 1e-05
    X += [5, 4, 3]
    ipca = IncrementalPCA(n_components=2, batch_size=10).fit(X)
    Y = ipca.transform(X)
    Y_inverse = ipca.inverse_transform(Y)
    assert_almost_equal(X, Y_inverse, decimal=3)

def test_incremental_pca_validation():
    if False:
        return 10
    X = np.array([[0, 1, 0], [1, 0, 0]])
    (n_samples, n_features) = X.shape
    n_components = 4
    with pytest.raises(ValueError, match='n_components={} invalid for n_features={}, need more rows than columns for IncrementalPCA processing'.format(n_components, n_features)):
        IncrementalPCA(n_components, batch_size=10).fit(X)
    n_components = 3
    with pytest.raises(ValueError, match='n_components={} must be less or equal to the batch number of samples {}'.format(n_components, n_samples)):
        IncrementalPCA(n_components=n_components).partial_fit(X)

def test_n_samples_equal_n_components():
    if False:
        print('Hello World!')
    ipca = IncrementalPCA(n_components=5)
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        ipca.partial_fit(np.random.randn(5, 7))
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        ipca.fit(np.random.randn(5, 7))

def test_n_components_none():
    if False:
        print('Hello World!')
    rng = np.random.RandomState(1999)
    for (n_samples, n_features) in [(50, 10), (10, 50)]:
        X = rng.rand(n_samples, n_features)
        ipca = IncrementalPCA(n_components=None)
        ipca.partial_fit(X)
        assert ipca.n_components_ == min(X.shape)
        ipca.partial_fit(X)
        assert ipca.n_components_ == ipca.components_.shape[0]

def test_incremental_pca_set_params():
    if False:
        while True:
            i = 10
    rng = np.random.RandomState(1999)
    n_samples = 100
    n_features = 20
    X = rng.randn(n_samples, n_features)
    X2 = rng.randn(n_samples, n_features)
    X3 = rng.randn(n_samples, n_features)
    ipca = IncrementalPCA(n_components=20)
    ipca.fit(X)
    ipca.set_params(n_components=10)
    with pytest.raises(ValueError):
        ipca.partial_fit(X2)
    ipca.set_params(n_components=15)
    with pytest.raises(ValueError):
        ipca.partial_fit(X3)
    ipca.set_params(n_components=20)
    ipca.partial_fit(X)

def test_incremental_pca_num_features_change():
    if False:
        i = 10
        return i + 15
    rng = np.random.RandomState(1999)
    n_samples = 100
    X = rng.randn(n_samples, 20)
    X2 = rng.randn(n_samples, 50)
    ipca = IncrementalPCA(n_components=None)
    ipca.fit(X)
    with pytest.raises(ValueError):
        ipca.partial_fit(X2)

def test_incremental_pca_batch_signs():
    if False:
        for i in range(10):
            print('nop')
    rng = np.random.RandomState(1999)
    n_samples = 100
    n_features = 3
    X = rng.randn(n_samples, n_features)
    all_components = []
    batch_sizes = np.arange(10, 20)
    for batch_size in batch_sizes:
        ipca = IncrementalPCA(n_components=None, batch_size=batch_size).fit(X)
        all_components.append(ipca.components_)
    for (i, j) in zip(all_components[:-1], all_components[1:]):
        assert_almost_equal(np.sign(i), np.sign(j), decimal=6)

def test_incremental_pca_batch_values():
    if False:
        i = 10
        return i + 15
    rng = np.random.RandomState(1999)
    n_samples = 100
    n_features = 3
    X = rng.randn(n_samples, n_features)
    all_components = []
    batch_sizes = np.arange(20, 40, 3)
    for batch_size in batch_sizes:
        ipca = IncrementalPCA(n_components=None, batch_size=batch_size).fit(X)
        all_components.append(ipca.components_)
    for (i, j) in zip(all_components[:-1], all_components[1:]):
        assert_almost_equal(i, j, decimal=1)

def test_incremental_pca_batch_rank():
    if False:
        print('Hello World!')
    rng = np.random.RandomState(1999)
    n_samples = 100
    n_features = 20
    X = rng.randn(n_samples, n_features)
    all_components = []
    batch_sizes = np.arange(20, 90, 3)
    for batch_size in batch_sizes:
        ipca = IncrementalPCA(n_components=20, batch_size=batch_size).fit(X)
        all_components.append(ipca.components_)
    for (components_i, components_j) in zip(all_components[:-1], all_components[1:]):
        assert_allclose_dense_sparse(components_i, components_j)

def test_incremental_pca_partial_fit():
    if False:
        print('Hello World!')
    rng = np.random.RandomState(1999)
    (n, p) = (50, 3)
    X = rng.randn(n, p)
    X[:, 1] *= 1e-05
    X += [5, 4, 3]
    batch_size = 10
    ipca = IncrementalPCA(n_components=2, batch_size=batch_size).fit(X)
    pipca = IncrementalPCA(n_components=2, batch_size=batch_size)
    batch_itr = np.arange(0, n + 1, batch_size)
    for (i, j) in zip(batch_itr[:-1], batch_itr[1:]):
        pipca.partial_fit(X[i:j, :])
    assert_almost_equal(ipca.components_, pipca.components_, decimal=3)

def test_incremental_pca_against_pca_iris():
    if False:
        return 10
    X = iris.data
    Y_pca = PCA(n_components=2).fit_transform(X)
    Y_ipca = IncrementalPCA(n_components=2, batch_size=25).fit_transform(X)
    assert_almost_equal(np.abs(Y_pca), np.abs(Y_ipca), 1)

def test_incremental_pca_against_pca_random_data():
    if False:
        for i in range(10):
            print('nop')
    rng = np.random.RandomState(1999)
    n_samples = 100
    n_features = 3
    X = rng.randn(n_samples, n_features) + 5 * rng.rand(1, n_features)
    Y_pca = PCA(n_components=3).fit_transform(X)
    Y_ipca = IncrementalPCA(n_components=3, batch_size=25).fit_transform(X)
    assert_almost_equal(np.abs(Y_pca), np.abs(Y_ipca), 1)

def test_explained_variances():
    if False:
        while True:
            i = 10
    X = datasets.make_low_rank_matrix(1000, 100, tail_strength=0.0, effective_rank=10, random_state=1999)
    prec = 3
    (n_samples, n_features) = X.shape
    for nc in [None, 99]:
        pca = PCA(n_components=nc).fit(X)
        ipca = IncrementalPCA(n_components=nc, batch_size=100).fit(X)
        assert_almost_equal(pca.explained_variance_, ipca.explained_variance_, decimal=prec)
        assert_almost_equal(pca.explained_variance_ratio_, ipca.explained_variance_ratio_, decimal=prec)
        assert_almost_equal(pca.noise_variance_, ipca.noise_variance_, decimal=prec)

def test_singular_values():
    if False:
        print('Hello World!')
    rng = np.random.RandomState(0)
    n_samples = 1000
    n_features = 100
    X = datasets.make_low_rank_matrix(n_samples, n_features, tail_strength=0.0, effective_rank=10, random_state=rng)
    pca = PCA(n_components=10, svd_solver='full', random_state=rng).fit(X)
    ipca = IncrementalPCA(n_components=10, batch_size=100).fit(X)
    assert_array_almost_equal(pca.singular_values_, ipca.singular_values_, 2)
    X_pca = pca.transform(X)
    X_ipca = ipca.transform(X)
    assert_array_almost_equal(np.sum(pca.singular_values_ ** 2.0), np.linalg.norm(X_pca, 'fro') ** 2.0, 12)
    assert_array_almost_equal(np.sum(ipca.singular_values_ ** 2.0), np.linalg.norm(X_ipca, 'fro') ** 2.0, 2)
    assert_array_almost_equal(pca.singular_values_, np.sqrt(np.sum(X_pca ** 2.0, axis=0)), 12)
    assert_array_almost_equal(ipca.singular_values_, np.sqrt(np.sum(X_ipca ** 2.0, axis=0)), 2)
    rng = np.random.RandomState(0)
    n_samples = 100
    n_features = 110
    X = datasets.make_low_rank_matrix(n_samples, n_features, tail_strength=0.0, effective_rank=3, random_state=rng)
    pca = PCA(n_components=3, svd_solver='full', random_state=rng)
    ipca = IncrementalPCA(n_components=3, batch_size=100)
    X_pca = pca.fit_transform(X)
    X_pca /= np.sqrt(np.sum(X_pca ** 2.0, axis=0))
    X_pca[:, 0] *= 3.142
    X_pca[:, 1] *= 2.718
    X_hat = np.dot(X_pca, pca.components_)
    pca.fit(X_hat)
    ipca.fit(X_hat)
    assert_array_almost_equal(pca.singular_values_, [3.142, 2.718, 1.0], 14)
    assert_array_almost_equal(ipca.singular_values_, [3.142, 2.718, 1.0], 14)

def test_whitening():
    if False:
        return 10
    X = datasets.make_low_rank_matrix(1000, 10, tail_strength=0.0, effective_rank=2, random_state=1999)
    prec = 3
    (n_samples, n_features) = X.shape
    for nc in [None, 9]:
        pca = PCA(whiten=True, n_components=nc).fit(X)
        ipca = IncrementalPCA(whiten=True, n_components=nc, batch_size=250).fit(X)
        Xt_pca = pca.transform(X)
        Xt_ipca = ipca.transform(X)
        assert_almost_equal(np.abs(Xt_pca), np.abs(Xt_ipca), decimal=prec)
        Xinv_ipca = ipca.inverse_transform(Xt_ipca)
        Xinv_pca = pca.inverse_transform(Xt_pca)
        assert_almost_equal(X, Xinv_ipca, decimal=prec)
        assert_almost_equal(X, Xinv_pca, decimal=prec)
        assert_almost_equal(Xinv_pca, Xinv_ipca, decimal=prec)

def test_incremental_pca_partial_fit_float_division():
    if False:
        print('Hello World!')
    rng = np.random.RandomState(0)
    A = rng.randn(5, 3) + 2
    B = rng.randn(7, 3) + 5
    pca = IncrementalPCA(n_components=2)
    pca.partial_fit(A)
    pca.n_samples_seen_ = float(pca.n_samples_seen_)
    pca.partial_fit(B)
    singular_vals_float_samples_seen = pca.singular_values_
    pca2 = IncrementalPCA(n_components=2)
    pca2.partial_fit(A)
    pca2.partial_fit(B)
    singular_vals_int_samples_seen = pca2.singular_values_
    np.testing.assert_allclose(singular_vals_float_samples_seen, singular_vals_int_samples_seen)

def test_incremental_pca_fit_overflow_error():
    if False:
        return 10
    rng = np.random.RandomState(0)
    A = rng.rand(500000, 2)
    ipca = IncrementalPCA(n_components=2, batch_size=10000)
    ipca.fit(A)
    pca = PCA(n_components=2)
    pca.fit(A)
    np.testing.assert_allclose(ipca.singular_values_, pca.singular_values_)

def test_incremental_pca_feature_names_out():
    if False:
        print('Hello World!')
    'Check feature names out for IncrementalPCA.'
    ipca = IncrementalPCA(n_components=2).fit(iris.data)
    names = ipca.get_feature_names_out()
    assert_array_equal([f'incrementalpca{i}' for i in range(2)], names)