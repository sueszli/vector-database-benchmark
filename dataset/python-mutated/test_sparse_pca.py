import sys
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.decomposition import PCA, MiniBatchSparsePCA, SparsePCA
from sklearn.utils import check_random_state
from sklearn.utils._testing import assert_allclose, assert_array_almost_equal, if_safe_multiprocessing_with_blas

def generate_toy_data(n_components, n_samples, image_size, random_state=None):
    if False:
        for i in range(10):
            print('nop')
    n_features = image_size[0] * image_size[1]
    rng = check_random_state(random_state)
    U = rng.randn(n_samples, n_components)
    V = rng.randn(n_components, n_features)
    centers = [(3, 3), (6, 7), (8, 1)]
    sz = [1, 2, 1]
    for k in range(n_components):
        img = np.zeros(image_size)
        (xmin, xmax) = (centers[k][0] - sz[k], centers[k][0] + sz[k])
        (ymin, ymax) = (centers[k][1] - sz[k], centers[k][1] + sz[k])
        img[xmin:xmax][:, ymin:ymax] = 1.0
        V[k, :] = img.ravel()
    Y = np.dot(U, V)
    Y += 0.1 * rng.randn(Y.shape[0], Y.shape[1])
    return (Y, U, V)

def test_correct_shapes():
    if False:
        i = 10
        return i + 15
    rng = np.random.RandomState(0)
    X = rng.randn(12, 10)
    spca = SparsePCA(n_components=8, random_state=rng)
    U = spca.fit_transform(X)
    assert spca.components_.shape == (8, 10)
    assert U.shape == (12, 8)
    spca = SparsePCA(n_components=13, random_state=rng)
    U = spca.fit_transform(X)
    assert spca.components_.shape == (13, 10)
    assert U.shape == (12, 13)

def test_fit_transform():
    if False:
        return 10
    alpha = 1
    rng = np.random.RandomState(0)
    (Y, _, _) = generate_toy_data(3, 10, (8, 8), random_state=rng)
    spca_lars = SparsePCA(n_components=3, method='lars', alpha=alpha, random_state=0)
    spca_lars.fit(Y)
    spca_lasso = SparsePCA(n_components=3, method='cd', random_state=0, alpha=alpha)
    spca_lasso.fit(Y)
    assert_array_almost_equal(spca_lasso.components_, spca_lars.components_)

@if_safe_multiprocessing_with_blas
def test_fit_transform_parallel():
    if False:
        return 10
    alpha = 1
    rng = np.random.RandomState(0)
    (Y, _, _) = generate_toy_data(3, 10, (8, 8), random_state=rng)
    spca_lars = SparsePCA(n_components=3, method='lars', alpha=alpha, random_state=0)
    spca_lars.fit(Y)
    U1 = spca_lars.transform(Y)
    spca = SparsePCA(n_components=3, n_jobs=2, method='lars', alpha=alpha, random_state=0).fit(Y)
    U2 = spca.transform(Y)
    assert not np.all(spca_lars.components_ == 0)
    assert_array_almost_equal(U1, U2)

def test_transform_nan():
    if False:
        return 10
    rng = np.random.RandomState(0)
    (Y, _, _) = generate_toy_data(3, 10, (8, 8), random_state=rng)
    Y[:, 0] = 0
    estimator = SparsePCA(n_components=8)
    assert not np.any(np.isnan(estimator.fit_transform(Y)))

def test_fit_transform_tall():
    if False:
        print('Hello World!')
    rng = np.random.RandomState(0)
    (Y, _, _) = generate_toy_data(3, 65, (8, 8), random_state=rng)
    spca_lars = SparsePCA(n_components=3, method='lars', random_state=rng)
    U1 = spca_lars.fit_transform(Y)
    spca_lasso = SparsePCA(n_components=3, method='cd', random_state=rng)
    U2 = spca_lasso.fit(Y).transform(Y)
    assert_array_almost_equal(U1, U2)

def test_initialization():
    if False:
        for i in range(10):
            print('nop')
    rng = np.random.RandomState(0)
    U_init = rng.randn(5, 3)
    V_init = rng.randn(3, 4)
    model = SparsePCA(n_components=3, U_init=U_init, V_init=V_init, max_iter=0, random_state=rng)
    model.fit(rng.randn(5, 4))
    assert_allclose(model.components_, V_init / np.linalg.norm(V_init, axis=1)[:, None])

def test_mini_batch_correct_shapes():
    if False:
        return 10
    rng = np.random.RandomState(0)
    X = rng.randn(12, 10)
    pca = MiniBatchSparsePCA(n_components=8, max_iter=1, random_state=rng)
    U = pca.fit_transform(X)
    assert pca.components_.shape == (8, 10)
    assert U.shape == (12, 8)
    pca = MiniBatchSparsePCA(n_components=13, max_iter=1, random_state=rng)
    U = pca.fit_transform(X)
    assert pca.components_.shape == (13, 10)
    assert U.shape == (12, 13)

@pytest.mark.skipif(True, reason='skipping mini_batch_fit_transform.')
def test_mini_batch_fit_transform():
    if False:
        print('Hello World!')
    alpha = 1
    rng = np.random.RandomState(0)
    (Y, _, _) = generate_toy_data(3, 10, (8, 8), random_state=rng)
    spca_lars = MiniBatchSparsePCA(n_components=3, random_state=0, alpha=alpha).fit(Y)
    U1 = spca_lars.transform(Y)
    if sys.platform == 'win32':
        import joblib
        _mp = joblib.parallel.multiprocessing
        joblib.parallel.multiprocessing = None
        try:
            spca = MiniBatchSparsePCA(n_components=3, n_jobs=2, alpha=alpha, random_state=0)
            U2 = spca.fit(Y).transform(Y)
        finally:
            joblib.parallel.multiprocessing = _mp
    else:
        spca = MiniBatchSparsePCA(n_components=3, n_jobs=2, alpha=alpha, random_state=0)
        U2 = spca.fit(Y).transform(Y)
    assert not np.all(spca_lars.components_ == 0)
    assert_array_almost_equal(U1, U2)
    spca_lasso = MiniBatchSparsePCA(n_components=3, method='cd', alpha=alpha, random_state=0).fit(Y)
    assert_array_almost_equal(spca_lasso.components_, spca_lars.components_)

def test_scaling_fit_transform():
    if False:
        return 10
    alpha = 1
    rng = np.random.RandomState(0)
    (Y, _, _) = generate_toy_data(3, 1000, (8, 8), random_state=rng)
    spca_lars = SparsePCA(n_components=3, method='lars', alpha=alpha, random_state=rng)
    results_train = spca_lars.fit_transform(Y)
    results_test = spca_lars.transform(Y[:10])
    assert_allclose(results_train[0], results_test[0])

def test_pca_vs_spca():
    if False:
        while True:
            i = 10
    rng = np.random.RandomState(0)
    (Y, _, _) = generate_toy_data(3, 1000, (8, 8), random_state=rng)
    (Z, _, _) = generate_toy_data(3, 10, (8, 8), random_state=rng)
    spca = SparsePCA(alpha=0, ridge_alpha=0, n_components=2)
    pca = PCA(n_components=2)
    pca.fit(Y)
    spca.fit(Y)
    results_test_pca = pca.transform(Z)
    results_test_spca = spca.transform(Z)
    assert_allclose(np.abs(spca.components_.dot(pca.components_.T)), np.eye(2), atol=1e-05)
    results_test_pca *= np.sign(results_test_pca[0, :])
    results_test_spca *= np.sign(results_test_spca[0, :])
    assert_allclose(results_test_pca, results_test_spca)

@pytest.mark.parametrize('SPCA', [SparsePCA, MiniBatchSparsePCA])
@pytest.mark.parametrize('n_components', [None, 3])
def test_spca_n_components_(SPCA, n_components):
    if False:
        print('Hello World!')
    rng = np.random.RandomState(0)
    (n_samples, n_features) = (12, 10)
    X = rng.randn(n_samples, n_features)
    model = SPCA(n_components=n_components).fit(X)
    if n_components is not None:
        assert model.n_components_ == n_components
    else:
        assert model.n_components_ == n_features

@pytest.mark.parametrize('SPCA', (SparsePCA, MiniBatchSparsePCA))
@pytest.mark.parametrize('method', ('lars', 'cd'))
@pytest.mark.parametrize('data_type, expected_type', ((np.float32, np.float32), (np.float64, np.float64), (np.int32, np.float64), (np.int64, np.float64)))
def test_sparse_pca_dtype_match(SPCA, method, data_type, expected_type):
    if False:
        print('Hello World!')
    (n_samples, n_features, n_components) = (12, 10, 3)
    rng = np.random.RandomState(0)
    input_array = rng.randn(n_samples, n_features).astype(data_type)
    model = SPCA(n_components=n_components, method=method)
    transformed = model.fit_transform(input_array)
    assert transformed.dtype == expected_type
    assert model.components_.dtype == expected_type

@pytest.mark.parametrize('SPCA', (SparsePCA, MiniBatchSparsePCA))
@pytest.mark.parametrize('method', ('lars', 'cd'))
def test_sparse_pca_numerical_consistency(SPCA, method):
    if False:
        return 10
    rtol = 0.001
    alpha = 2
    (n_samples, n_features, n_components) = (12, 10, 3)
    rng = np.random.RandomState(0)
    input_array = rng.randn(n_samples, n_features)
    model_32 = SPCA(n_components=n_components, alpha=alpha, method=method, random_state=0)
    transformed_32 = model_32.fit_transform(input_array.astype(np.float32))
    model_64 = SPCA(n_components=n_components, alpha=alpha, method=method, random_state=0)
    transformed_64 = model_64.fit_transform(input_array.astype(np.float64))
    assert_allclose(transformed_64, transformed_32, rtol=rtol)
    assert_allclose(model_64.components_, model_32.components_, rtol=rtol)

@pytest.mark.parametrize('SPCA', [SparsePCA, MiniBatchSparsePCA])
def test_spca_feature_names_out(SPCA):
    if False:
        return 10
    'Check feature names out for *SparsePCA.'
    rng = np.random.RandomState(0)
    (n_samples, n_features) = (12, 10)
    X = rng.randn(n_samples, n_features)
    model = SPCA(n_components=4).fit(X)
    names = model.get_feature_names_out()
    estimator_name = SPCA.__name__.lower()
    assert_array_equal([f'{estimator_name}{i}' for i in range(4)], names)

def test_spca_n_iter_deprecation():
    if False:
        while True:
            i = 10
    'Check that we raise a warning for the deprecation of `n_iter` and it is ignored\n    when `max_iter` is specified.\n    '
    rng = np.random.RandomState(0)
    (n_samples, n_features) = (12, 10)
    X = rng.randn(n_samples, n_features)
    warn_msg = "'n_iter' is deprecated in version 1.1 and will be removed"
    with pytest.warns(FutureWarning, match=warn_msg):
        MiniBatchSparsePCA(n_iter=2).fit(X)
    (n_iter, max_iter) = (1, 100)
    with pytest.warns(FutureWarning, match=warn_msg):
        model = MiniBatchSparsePCA(n_iter=n_iter, max_iter=max_iter, random_state=0).fit(X)
    assert model.n_iter_ > 1
    assert model.n_iter_ <= max_iter

def test_pca_n_features_deprecation():
    if False:
        return 10
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    pca = PCA(n_components=2).fit(X)
    with pytest.warns(FutureWarning, match='`n_features_` was deprecated'):
        pca.n_features_

def test_spca_early_stopping(global_random_seed):
    if False:
        while True:
            i = 10
    'Check that `tol` and `max_no_improvement` act as early stopping.'
    rng = np.random.RandomState(global_random_seed)
    (n_samples, n_features) = (50, 10)
    X = rng.randn(n_samples, n_features)
    model_early_stopped = MiniBatchSparsePCA(max_iter=100, tol=0.5, random_state=global_random_seed).fit(X)
    model_not_early_stopped = MiniBatchSparsePCA(max_iter=100, tol=0.001, random_state=global_random_seed).fit(X)
    assert model_early_stopped.n_iter_ < model_not_early_stopped.n_iter_
    model_early_stopped = MiniBatchSparsePCA(max_iter=100, tol=1e-06, max_no_improvement=2, random_state=global_random_seed).fit(X)
    model_not_early_stopped = MiniBatchSparsePCA(max_iter=100, tol=1e-06, max_no_improvement=100, random_state=global_random_seed).fit(X)
    assert model_early_stopped.n_iter_ < model_not_early_stopped.n_iter_

def test_equivalence_components_pca_spca(global_random_seed):
    if False:
        while True:
            i = 10
    'Check the equivalence of the components found by PCA and SparsePCA.\n\n    Non-regression test for:\n    https://github.com/scikit-learn/scikit-learn/issues/23932\n    '
    rng = np.random.RandomState(global_random_seed)
    X = rng.randn(50, 4)
    n_components = 2
    pca = PCA(n_components=n_components, svd_solver='randomized', random_state=0).fit(X)
    spca = SparsePCA(n_components=n_components, method='lars', ridge_alpha=0, alpha=0, random_state=0).fit(X)
    assert_allclose(pca.components_, spca.components_)

def test_sparse_pca_inverse_transform():
    if False:
        for i in range(10):
            print('nop')
    'Check that `inverse_transform` in `SparsePCA` and `PCA` are similar.'
    rng = np.random.RandomState(0)
    (n_samples, n_features) = (10, 5)
    X = rng.randn(n_samples, n_features)
    n_components = 2
    spca = SparsePCA(n_components=n_components, alpha=1e-12, ridge_alpha=1e-12, random_state=0)
    pca = PCA(n_components=n_components, random_state=0)
    X_trans_spca = spca.fit_transform(X)
    X_trans_pca = pca.fit_transform(X)
    assert_allclose(spca.inverse_transform(X_trans_spca), pca.inverse_transform(X_trans_pca))

@pytest.mark.parametrize('SPCA', [SparsePCA, MiniBatchSparsePCA])
def test_transform_inverse_transform_round_trip(SPCA):
    if False:
        i = 10
        return i + 15
    'Check the `transform` and `inverse_transform` round trip with no loss of\n    information.\n    '
    rng = np.random.RandomState(0)
    (n_samples, n_features) = (10, 5)
    X = rng.randn(n_samples, n_features)
    n_components = n_features
    spca = SPCA(n_components=n_components, alpha=1e-12, ridge_alpha=1e-12, random_state=0)
    X_trans_spca = spca.fit_transform(X)
    assert_allclose(spca.inverse_transform(X_trans_spca), X)