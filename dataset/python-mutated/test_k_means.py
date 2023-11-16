"""Testing for K-means"""
import re
import sys
import warnings
from io import StringIO
import numpy as np
import pytest
from scipy import sparse as sp
from sklearn.base import clone
from sklearn.cluster import KMeans, MiniBatchKMeans, k_means, kmeans_plusplus
from sklearn.cluster._k_means_common import _euclidean_dense_dense_wrapper, _euclidean_sparse_dense_wrapper, _inertia_dense, _inertia_sparse, _is_same_clustering, _relocate_empty_clusters_dense, _relocate_empty_clusters_sparse
from sklearn.cluster._kmeans import _labels_inertia, _mini_batch_step
from sklearn.datasets import make_blobs
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin
from sklearn.metrics.cluster import v_measure_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils._testing import assert_allclose, assert_array_equal, create_memmap_backed_data
from sklearn.utils.extmath import row_norms
from sklearn.utils.fixes import CSR_CONTAINERS, threadpool_limits
msg = "The default value of `n_init` will change from \\d* to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning:FutureWarning"
pytestmark = pytest.mark.filterwarnings('ignore:' + msg)
centers = np.array([[0.0, 5.0, 0.0, 0.0, 0.0], [1.0, 1.0, 4.0, 0.0, 0.0], [1.0, 0.0, 0.0, 5.0, 1.0]])
n_samples = 100
(n_clusters, n_features) = centers.shape
(X, true_labels) = make_blobs(n_samples=n_samples, centers=centers, cluster_std=1.0, random_state=42)
X_as_any_csr = [container(X) for container in CSR_CONTAINERS]
data_containers = [np.array] + CSR_CONTAINERS
data_containers_ids = ['dense', 'sparse_matrix', 'sparse_array'] if len(X_as_any_csr) == 2 else ['dense', 'sparse_matrix']

@pytest.mark.parametrize('array_constr', data_containers, ids=data_containers_ids)
@pytest.mark.parametrize('algo', ['lloyd', 'elkan'])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_kmeans_results(array_constr, algo, dtype):
    if False:
        print('Hello World!')
    X = array_constr([[0, 0], [0.5, 0], [0.5, 1], [1, 1]], dtype=dtype)
    sample_weight = [3, 1, 1, 3]
    init_centers = np.array([[0, 0], [1, 1]], dtype=dtype)
    expected_labels = [0, 0, 1, 1]
    expected_inertia = 0.375
    expected_centers = np.array([[0.125, 0], [0.875, 1]], dtype=dtype)
    expected_n_iter = 2
    kmeans = KMeans(n_clusters=2, n_init=1, init=init_centers, algorithm=algo)
    kmeans.fit(X, sample_weight=sample_weight)
    assert_array_equal(kmeans.labels_, expected_labels)
    assert_allclose(kmeans.inertia_, expected_inertia)
    assert_allclose(kmeans.cluster_centers_, expected_centers)
    assert kmeans.n_iter_ == expected_n_iter

@pytest.mark.parametrize('array_constr', data_containers, ids=data_containers_ids)
@pytest.mark.parametrize('algo', ['lloyd', 'elkan'])
def test_kmeans_relocated_clusters(array_constr, algo):
    if False:
        print('Hello World!')
    X = array_constr([[0, 0], [0.5, 0], [0.5, 1], [1, 1]])
    init_centers = np.array([[0.5, 0.5], [3, 3]])
    kmeans = KMeans(n_clusters=2, n_init=1, init=init_centers, algorithm=algo)
    kmeans.fit(X)
    expected_n_iter = 3
    expected_inertia = 0.25
    assert_allclose(kmeans.inertia_, expected_inertia)
    assert kmeans.n_iter_ == expected_n_iter
    try:
        expected_labels = [0, 0, 1, 1]
        expected_centers = [[0.25, 0], [0.75, 1]]
        assert_array_equal(kmeans.labels_, expected_labels)
        assert_allclose(kmeans.cluster_centers_, expected_centers)
    except AssertionError:
        expected_labels = [1, 1, 0, 0]
        expected_centers = [[0.75, 1.0], [0.25, 0.0]]
        assert_array_equal(kmeans.labels_, expected_labels)
        assert_allclose(kmeans.cluster_centers_, expected_centers)

@pytest.mark.parametrize('array_constr', data_containers, ids=data_containers_ids)
def test_relocate_empty_clusters(array_constr):
    if False:
        return 10
    X = np.array([-10.0, -9.5, -9, -8.5, -8, -1, 1, 9, 9.5, 10]).reshape(-1, 1)
    X = array_constr(X)
    sample_weight = np.ones(10)
    centers_old = np.array([-10.0, -10, -10]).reshape(-1, 1)
    centers_new = np.array([-16.5, -10, -10]).reshape(-1, 1)
    weight_in_clusters = np.array([10.0, 0, 0])
    labels = np.zeros(10, dtype=np.int32)
    if array_constr is np.array:
        _relocate_empty_clusters_dense(X, sample_weight, centers_old, centers_new, weight_in_clusters, labels)
    else:
        _relocate_empty_clusters_sparse(X.data, X.indices, X.indptr, sample_weight, centers_old, centers_new, weight_in_clusters, labels)
    assert_array_equal(weight_in_clusters, [8, 1, 1])
    assert_allclose(centers_new, [[-36], [10], [9.5]])

@pytest.mark.parametrize('distribution', ['normal', 'blobs'])
@pytest.mark.parametrize('array_constr', data_containers, ids=data_containers_ids)
@pytest.mark.parametrize('tol', [0.01, 1e-08, 1e-100, 0])
def test_kmeans_elkan_results(distribution, array_constr, tol, global_random_seed):
    if False:
        i = 10
        return i + 15
    rnd = np.random.RandomState(global_random_seed)
    if distribution == 'normal':
        X = rnd.normal(size=(5000, 10))
    else:
        (X, _) = make_blobs(random_state=rnd)
    X[X < 0] = 0
    X = array_constr(X)
    km_lloyd = KMeans(n_clusters=5, random_state=global_random_seed, n_init=1, tol=tol)
    km_elkan = KMeans(algorithm='elkan', n_clusters=5, random_state=global_random_seed, n_init=1, tol=tol)
    km_lloyd.fit(X)
    km_elkan.fit(X)
    assert_allclose(km_elkan.cluster_centers_, km_lloyd.cluster_centers_)
    assert_array_equal(km_elkan.labels_, km_lloyd.labels_)
    assert km_elkan.n_iter_ == km_lloyd.n_iter_
    assert km_elkan.inertia_ == pytest.approx(km_lloyd.inertia_, rel=1e-06)

@pytest.mark.parametrize('algorithm', ['lloyd', 'elkan'])
def test_kmeans_convergence(algorithm, global_random_seed):
    if False:
        for i in range(10):
            print('nop')
    rnd = np.random.RandomState(global_random_seed)
    X = rnd.normal(size=(5000, 10))
    max_iter = 300
    km = KMeans(algorithm=algorithm, n_clusters=5, random_state=global_random_seed, n_init=1, tol=0, max_iter=max_iter).fit(X)
    assert km.n_iter_ < max_iter

@pytest.mark.parametrize('algorithm', ['auto', 'full'])
def test_algorithm_auto_full_deprecation_warning(algorithm):
    if False:
        while True:
            i = 10
    X = np.random.rand(100, 2)
    kmeans = KMeans(algorithm=algorithm)
    with pytest.warns(FutureWarning, match=f"algorithm='{algorithm}' is deprecated, it will be removed in 1.3. Using 'lloyd' instead."):
        kmeans.fit(X)
        assert kmeans._algorithm == 'lloyd'

@pytest.mark.parametrize('Estimator', [KMeans, MiniBatchKMeans])
def test_predict_sample_weight_deprecation_warning(Estimator):
    if False:
        i = 10
        return i + 15
    X = np.random.rand(100, 2)
    sample_weight = np.random.uniform(size=100)
    kmeans = Estimator()
    kmeans.fit(X, sample_weight=sample_weight)
    warn_msg = "'sample_weight' was deprecated in version 1.3 and will be removed in 1.5."
    with pytest.warns(FutureWarning, match=warn_msg):
        kmeans.predict(X, sample_weight=sample_weight)

@pytest.mark.parametrize('X_csr', X_as_any_csr)
def test_minibatch_update_consistency(X_csr, global_random_seed):
    if False:
        print('Hello World!')
    rng = np.random.RandomState(global_random_seed)
    centers_old = centers + rng.normal(size=centers.shape)
    centers_old_csr = centers_old.copy()
    centers_new = np.zeros_like(centers_old)
    centers_new_csr = np.zeros_like(centers_old_csr)
    weight_sums = np.zeros(centers_old.shape[0], dtype=X.dtype)
    weight_sums_csr = np.zeros(centers_old.shape[0], dtype=X.dtype)
    sample_weight = np.ones(X.shape[0], dtype=X.dtype)
    X_mb = X[:10]
    X_mb_csr = X_csr[:10]
    sample_weight_mb = sample_weight[:10]
    old_inertia = _mini_batch_step(X_mb, sample_weight_mb, centers_old, centers_new, weight_sums, np.random.RandomState(global_random_seed), random_reassign=False)
    assert old_inertia > 0.0
    (labels, new_inertia) = _labels_inertia(X_mb, sample_weight_mb, centers_new)
    assert new_inertia > 0.0
    assert new_inertia < old_inertia
    old_inertia_csr = _mini_batch_step(X_mb_csr, sample_weight_mb, centers_old_csr, centers_new_csr, weight_sums_csr, np.random.RandomState(global_random_seed), random_reassign=False)
    assert old_inertia_csr > 0.0
    (labels_csr, new_inertia_csr) = _labels_inertia(X_mb_csr, sample_weight_mb, centers_new_csr)
    assert new_inertia_csr > 0.0
    assert new_inertia_csr < old_inertia_csr
    assert_array_equal(labels, labels_csr)
    assert_allclose(centers_new, centers_new_csr)
    assert_allclose(old_inertia, old_inertia_csr)
    assert_allclose(new_inertia, new_inertia_csr)

def _check_fitted_model(km):
    if False:
        for i in range(10):
            print('nop')
    centers = km.cluster_centers_
    assert centers.shape == (n_clusters, n_features)
    labels = km.labels_
    assert np.unique(labels).shape[0] == n_clusters
    assert_allclose(v_measure_score(true_labels, labels), 1.0)
    assert km.inertia_ > 0.0

@pytest.mark.parametrize('input_data', [X] + X_as_any_csr, ids=data_containers_ids)
@pytest.mark.parametrize('init', ['random', 'k-means++', centers, lambda X, k, random_state: centers], ids=['random', 'k-means++', 'ndarray', 'callable'])
@pytest.mark.parametrize('Estimator', [KMeans, MiniBatchKMeans])
def test_all_init(Estimator, input_data, init):
    if False:
        print('Hello World!')
    n_init = 10 if isinstance(init, str) else 1
    km = Estimator(init=init, n_clusters=n_clusters, random_state=42, n_init=n_init).fit(input_data)
    _check_fitted_model(km)

@pytest.mark.parametrize('init', ['random', 'k-means++', centers, lambda X, k, random_state: centers], ids=['random', 'k-means++', 'ndarray', 'callable'])
def test_minibatch_kmeans_partial_fit_init(init):
    if False:
        return 10
    n_init = 10 if isinstance(init, str) else 1
    km = MiniBatchKMeans(init=init, n_clusters=n_clusters, random_state=0, n_init=n_init)
    for i in range(100):
        km.partial_fit(X)
    _check_fitted_model(km)

@pytest.mark.parametrize('init, expected_n_init', [('k-means++', 1), ('random', 'default'), (lambda X, n_clusters, random_state: random_state.uniform(size=(n_clusters, X.shape[1])), 'default'), ('array-like', 1)])
@pytest.mark.parametrize('Estimator', [KMeans, MiniBatchKMeans])
def test_kmeans_init_auto_with_initial_centroids(Estimator, init, expected_n_init):
    if False:
        for i in range(10):
            print('nop')
    'Check that `n_init="auto"` chooses the right number of initializations.\n    Non-regression test for #26657:\n    https://github.com/scikit-learn/scikit-learn/pull/26657\n    '
    (n_sample, n_features, n_clusters) = (100, 10, 5)
    X = np.random.randn(n_sample, n_features)
    if init == 'array-like':
        init = np.random.randn(n_clusters, n_features)
    if expected_n_init == 'default':
        expected_n_init = 3 if Estimator is MiniBatchKMeans else 10
    kmeans = Estimator(n_clusters=n_clusters, init=init, n_init='auto').fit(X)
    assert kmeans._n_init == expected_n_init

@pytest.mark.parametrize('Estimator', [KMeans, MiniBatchKMeans])
def test_fortran_aligned_data(Estimator, global_random_seed):
    if False:
        i = 10
        return i + 15
    X_fortran = np.asfortranarray(X)
    centers_fortran = np.asfortranarray(centers)
    km_c = Estimator(n_clusters=n_clusters, init=centers, n_init=1, random_state=global_random_seed).fit(X)
    km_f = Estimator(n_clusters=n_clusters, init=centers_fortran, n_init=1, random_state=global_random_seed).fit(X_fortran)
    assert_allclose(km_c.cluster_centers_, km_f.cluster_centers_)
    assert_array_equal(km_c.labels_, km_f.labels_)

def test_minibatch_kmeans_verbose():
    if False:
        while True:
            i = 10
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, verbose=1)
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        km.fit(X)
    finally:
        sys.stdout = old_stdout

@pytest.mark.parametrize('algorithm', ['lloyd', 'elkan'])
@pytest.mark.parametrize('tol', [0.01, 0])
def test_kmeans_verbose(algorithm, tol, capsys):
    if False:
        for i in range(10):
            print('nop')
    X = np.random.RandomState(0).normal(size=(5000, 10))
    KMeans(algorithm=algorithm, n_clusters=n_clusters, random_state=42, init='random', n_init=1, tol=tol, verbose=1).fit(X)
    captured = capsys.readouterr()
    assert re.search('Initialization complete', captured.out)
    assert re.search('Iteration [0-9]+, inertia', captured.out)
    if tol == 0:
        assert re.search('strict convergence', captured.out)
    else:
        assert re.search('center shift .* within tolerance', captured.out)

def test_minibatch_kmeans_warning_init_size():
    if False:
        return 10
    with pytest.warns(RuntimeWarning, match='init_size.* should be larger than n_clusters'):
        MiniBatchKMeans(init_size=10, n_clusters=20).fit(X)

@pytest.mark.parametrize('Estimator', [KMeans, MiniBatchKMeans])
def test_warning_n_init_precomputed_centers(Estimator):
    if False:
        return 10
    with pytest.warns(RuntimeWarning, match='Explicit initial center position passed: performing only one init'):
        Estimator(init=centers, n_clusters=n_clusters, n_init=10).fit(X)

def test_minibatch_sensible_reassign(global_random_seed):
    if False:
        for i in range(10):
            print('nop')
    (zeroed_X, true_labels) = make_blobs(n_samples=100, centers=5, random_state=global_random_seed)
    zeroed_X[::2, :] = 0
    km = MiniBatchKMeans(n_clusters=20, batch_size=10, random_state=global_random_seed, init='random').fit(zeroed_X)
    assert km.cluster_centers_.any(axis=1).sum() > 10
    km = MiniBatchKMeans(n_clusters=20, batch_size=200, random_state=global_random_seed, init='random').fit(zeroed_X)
    assert km.cluster_centers_.any(axis=1).sum() > 10
    km = MiniBatchKMeans(n_clusters=20, random_state=global_random_seed, init='random')
    for i in range(100):
        km.partial_fit(zeroed_X)
    assert km.cluster_centers_.any(axis=1).sum() > 10

@pytest.mark.parametrize('input_data', [X] + X_as_any_csr, ids=data_containers_ids)
def test_minibatch_reassign(input_data, global_random_seed):
    if False:
        print('Hello World!')
    perfect_centers = np.empty((n_clusters, n_features))
    for i in range(n_clusters):
        perfect_centers[i] = X[true_labels == i].mean(axis=0)
    sample_weight = np.ones(n_samples)
    centers_new = np.empty_like(perfect_centers)
    score_before = -_labels_inertia(input_data, sample_weight, perfect_centers, 1)[1]
    _mini_batch_step(input_data, sample_weight, perfect_centers, centers_new, np.zeros(n_clusters), np.random.RandomState(global_random_seed), random_reassign=True, reassignment_ratio=1)
    score_after = -_labels_inertia(input_data, sample_weight, centers_new, 1)[1]
    assert score_before > score_after
    _mini_batch_step(input_data, sample_weight, perfect_centers, centers_new, np.zeros(n_clusters), np.random.RandomState(global_random_seed), random_reassign=True, reassignment_ratio=1e-15)
    assert_allclose(centers_new, perfect_centers)

def test_minibatch_with_many_reassignments():
    if False:
        for i in range(10):
            print('nop')
    MiniBatchKMeans(n_clusters=100, batch_size=10, init_size=n_samples, random_state=42, verbose=True).fit(X)

def test_minibatch_kmeans_init_size():
    if False:
        print('Hello World!')
    km = MiniBatchKMeans(n_clusters=10, batch_size=5, n_init=1).fit(X)
    assert km._init_size == 15
    km = MiniBatchKMeans(n_clusters=10, batch_size=1, n_init=1).fit(X)
    assert km._init_size == 30
    km = MiniBatchKMeans(n_clusters=10, batch_size=5, n_init=1, init_size=n_samples + 1).fit(X)
    assert km._init_size == n_samples

@pytest.mark.parametrize('tol, max_no_improvement', [(0.0001, None), (0, 10)])
def test_minibatch_declared_convergence(capsys, tol, max_no_improvement):
    if False:
        print('Hello World!')
    (X, _, centers) = make_blobs(centers=3, random_state=0, return_centers=True)
    km = MiniBatchKMeans(n_clusters=3, init=centers, batch_size=20, tol=tol, random_state=0, max_iter=10, n_init=1, verbose=1, max_no_improvement=max_no_improvement)
    km.fit(X)
    assert 1 < km.n_iter_ < 10
    captured = capsys.readouterr()
    if max_no_improvement is None:
        assert 'Converged (small centers change)' in captured.out
    if tol == 0:
        assert 'Converged (lack of improvement in inertia)' in captured.out

def test_minibatch_iter_steps():
    if False:
        while True:
            i = 10
    batch_size = 30
    n_samples = X.shape[0]
    km = MiniBatchKMeans(n_clusters=3, batch_size=batch_size, random_state=0).fit(X)
    assert km.n_iter_ == np.ceil(km.n_steps_ * batch_size / n_samples)
    assert isinstance(km.n_iter_, int)
    km = MiniBatchKMeans(n_clusters=3, batch_size=batch_size, random_state=0, tol=0, max_no_improvement=None, max_iter=10).fit(X)
    assert km.n_iter_ == 10
    assert km.n_steps_ == 10 * n_samples // batch_size
    assert isinstance(km.n_steps_, int)

def test_kmeans_copyx():
    if False:
        i = 10
        return i + 15
    my_X = X.copy()
    km = KMeans(copy_x=False, n_clusters=n_clusters, random_state=42)
    km.fit(my_X)
    _check_fitted_model(km)
    assert_allclose(my_X, X)

@pytest.mark.parametrize('Estimator', [KMeans, MiniBatchKMeans])
def test_score_max_iter(Estimator, global_random_seed):
    if False:
        return 10
    X = np.random.RandomState(global_random_seed).randn(100, 10)
    km1 = Estimator(n_init=1, random_state=global_random_seed, max_iter=1)
    s1 = km1.fit(X).score(X)
    km2 = Estimator(n_init=1, random_state=global_random_seed, max_iter=10)
    s2 = km2.fit(X).score(X)
    assert s2 > s1

@pytest.mark.parametrize('array_constr', data_containers, ids=data_containers_ids)
@pytest.mark.parametrize('Estimator, algorithm', [(KMeans, 'lloyd'), (KMeans, 'elkan'), (MiniBatchKMeans, None)])
@pytest.mark.parametrize('max_iter', [2, 100])
def test_kmeans_predict(Estimator, algorithm, array_constr, max_iter, global_dtype, global_random_seed):
    if False:
        return 10
    (X, _) = make_blobs(n_samples=200, n_features=10, centers=10, random_state=global_random_seed)
    X = array_constr(X, dtype=global_dtype)
    km = Estimator(n_clusters=10, init='random', n_init=10, max_iter=max_iter, random_state=global_random_seed)
    if algorithm is not None:
        km.set_params(algorithm=algorithm)
    km.fit(X)
    labels = km.labels_
    pred = km.predict(X)
    assert_array_equal(pred, labels)
    pred = km.fit_predict(X)
    assert_array_equal(pred, labels)
    pred = km.predict(km.cluster_centers_)
    assert_array_equal(pred, np.arange(10))

@pytest.mark.parametrize('X_csr', X_as_any_csr)
@pytest.mark.parametrize('Estimator', [KMeans, MiniBatchKMeans])
def test_dense_sparse(Estimator, X_csr, global_random_seed):
    if False:
        print('Hello World!')
    sample_weight = np.random.RandomState(global_random_seed).random_sample((n_samples,))
    km_dense = Estimator(n_clusters=n_clusters, random_state=global_random_seed, n_init=1)
    km_dense.fit(X, sample_weight=sample_weight)
    km_sparse = Estimator(n_clusters=n_clusters, random_state=global_random_seed, n_init=1)
    km_sparse.fit(X_csr, sample_weight=sample_weight)
    assert_array_equal(km_dense.labels_, km_sparse.labels_)
    assert_allclose(km_dense.cluster_centers_, km_sparse.cluster_centers_)

@pytest.mark.parametrize('X_csr', X_as_any_csr)
@pytest.mark.parametrize('init', ['random', 'k-means++', centers], ids=['random', 'k-means++', 'ndarray'])
@pytest.mark.parametrize('Estimator', [KMeans, MiniBatchKMeans])
def test_predict_dense_sparse(Estimator, init, X_csr):
    if False:
        while True:
            i = 10
    n_init = 10 if isinstance(init, str) else 1
    km = Estimator(n_clusters=n_clusters, init=init, n_init=n_init, random_state=0)
    km.fit(X_csr)
    assert_array_equal(km.predict(X), km.labels_)
    km.fit(X)
    assert_array_equal(km.predict(X_csr), km.labels_)

@pytest.mark.parametrize('array_constr', data_containers, ids=data_containers_ids)
@pytest.mark.parametrize('dtype', [np.int32, np.int64])
@pytest.mark.parametrize('init', ['k-means++', 'ndarray'])
@pytest.mark.parametrize('Estimator', [KMeans, MiniBatchKMeans])
def test_integer_input(Estimator, array_constr, dtype, init, global_random_seed):
    if False:
        i = 10
        return i + 15
    X_dense = np.array([[0, 0], [10, 10], [12, 9], [-1, 1], [2, 0], [8, 10]])
    X = array_constr(X_dense, dtype=dtype)
    n_init = 1 if init == 'ndarray' else 10
    init = X_dense[:2] if init == 'ndarray' else init
    km = Estimator(n_clusters=2, init=init, n_init=n_init, random_state=global_random_seed)
    if Estimator is MiniBatchKMeans:
        km.set_params(batch_size=2)
    km.fit(X)
    assert km.cluster_centers_.dtype == np.float64
    expected_labels = [0, 1, 1, 0, 0, 1]
    assert_allclose(v_measure_score(km.labels_, expected_labels), 1.0)
    if Estimator is MiniBatchKMeans:
        km = clone(km).partial_fit(X)
        assert km.cluster_centers_.dtype == np.float64

@pytest.mark.parametrize('Estimator', [KMeans, MiniBatchKMeans])
def test_transform(Estimator, global_random_seed):
    if False:
        return 10
    km = Estimator(n_clusters=n_clusters, random_state=global_random_seed).fit(X)
    Xt = km.transform(km.cluster_centers_)
    assert_allclose(Xt, pairwise_distances(km.cluster_centers_))
    assert_array_equal(Xt.diagonal(), np.zeros(n_clusters))
    Xt = km.transform(X)
    assert_allclose(Xt, pairwise_distances(X, km.cluster_centers_))

@pytest.mark.parametrize('Estimator', [KMeans, MiniBatchKMeans])
def test_fit_transform(Estimator, global_random_seed):
    if False:
        print('Hello World!')
    X1 = Estimator(random_state=global_random_seed, n_init=1).fit(X).transform(X)
    X2 = Estimator(random_state=global_random_seed, n_init=1).fit_transform(X)
    assert_allclose(X1, X2)

def test_n_init(global_random_seed):
    if False:
        print('Hello World!')
    previous_inertia = np.inf
    for n_init in [1, 5, 10]:
        km = KMeans(n_clusters=n_clusters, init='random', n_init=n_init, random_state=global_random_seed, max_iter=1).fit(X)
        assert km.inertia_ <= previous_inertia

def test_k_means_function(global_random_seed):
    if False:
        i = 10
        return i + 15
    (cluster_centers, labels, inertia) = k_means(X, n_clusters=n_clusters, sample_weight=None, random_state=global_random_seed)
    assert cluster_centers.shape == (n_clusters, n_features)
    assert np.unique(labels).shape[0] == n_clusters
    assert_allclose(v_measure_score(true_labels, labels), 1.0)
    assert inertia > 0.0

@pytest.mark.parametrize('input_data', [X] + X_as_any_csr, ids=data_containers_ids)
@pytest.mark.parametrize('Estimator', [KMeans, MiniBatchKMeans])
def test_float_precision(Estimator, input_data, global_random_seed):
    if False:
        return 10
    km = Estimator(n_init=1, random_state=global_random_seed)
    inertia = {}
    Xt = {}
    centers = {}
    labels = {}
    for dtype in [np.float64, np.float32]:
        X = input_data.astype(dtype, copy=False)
        km.fit(X)
        inertia[dtype] = km.inertia_
        Xt[dtype] = km.transform(X)
        centers[dtype] = km.cluster_centers_
        labels[dtype] = km.labels_
        assert km.cluster_centers_.dtype == dtype
        if Estimator is MiniBatchKMeans:
            km.partial_fit(X[0:3])
            assert km.cluster_centers_.dtype == dtype
    assert_allclose(inertia[np.float32], inertia[np.float64], rtol=0.0001)
    assert_allclose(Xt[np.float32], Xt[np.float64], atol=Xt[np.float64].max() * 0.0001)
    assert_allclose(centers[np.float32], centers[np.float64], atol=centers[np.float64].max() * 0.0001)
    assert_array_equal(labels[np.float32], labels[np.float64])

@pytest.mark.parametrize('dtype', [np.int32, np.int64, np.float32, np.float64])
@pytest.mark.parametrize('Estimator', [KMeans, MiniBatchKMeans])
def test_centers_not_mutated(Estimator, dtype):
    if False:
        i = 10
        return i + 15
    X_new_type = X.astype(dtype, copy=False)
    centers_new_type = centers.astype(dtype, copy=False)
    km = Estimator(init=centers_new_type, n_clusters=n_clusters, n_init=1)
    km.fit(X_new_type)
    assert not np.may_share_memory(km.cluster_centers_, centers_new_type)

@pytest.mark.parametrize('input_data', [X] + X_as_any_csr, ids=data_containers_ids)
def test_kmeans_init_fitted_centers(input_data):
    if False:
        while True:
            i = 10
    km1 = KMeans(n_clusters=n_clusters).fit(input_data)
    km2 = KMeans(n_clusters=n_clusters, init=km1.cluster_centers_, n_init=1).fit(input_data)
    assert_allclose(km1.cluster_centers_, km2.cluster_centers_)

def test_kmeans_warns_less_centers_than_unique_points(global_random_seed):
    if False:
        while True:
            i = 10
    X = np.asarray([[0, 0], [0, 1], [1, 0], [1, 0]])
    km = KMeans(n_clusters=4, random_state=global_random_seed)
    msg = 'Number of distinct clusters \\(3\\) found smaller than n_clusters \\(4\\). Possibly due to duplicate points in X.'
    with pytest.warns(ConvergenceWarning, match=msg):
        km.fit(X)
        assert set(km.labels_) == set(range(3))

def _sort_centers(centers):
    if False:
        return 10
    return np.sort(centers, axis=0)

def test_weighted_vs_repeated(global_random_seed):
    if False:
        i = 10
        return i + 15
    sample_weight = np.random.RandomState(global_random_seed).randint(1, 5, size=n_samples)
    X_repeat = np.repeat(X, sample_weight, axis=0)
    km = KMeans(init=centers, n_init=1, n_clusters=n_clusters, random_state=global_random_seed)
    km_weighted = clone(km).fit(X, sample_weight=sample_weight)
    repeated_labels = np.repeat(km_weighted.labels_, sample_weight)
    km_repeated = clone(km).fit(X_repeat)
    assert_array_equal(km_repeated.labels_, repeated_labels)
    assert_allclose(km_weighted.inertia_, km_repeated.inertia_)
    assert_allclose(_sort_centers(km_weighted.cluster_centers_), _sort_centers(km_repeated.cluster_centers_))

@pytest.mark.parametrize('input_data', [X] + X_as_any_csr, ids=data_containers_ids)
@pytest.mark.parametrize('Estimator', [KMeans, MiniBatchKMeans])
def test_unit_weights_vs_no_weights(Estimator, input_data, global_random_seed):
    if False:
        print('Hello World!')
    sample_weight = np.ones(n_samples)
    km = Estimator(n_clusters=n_clusters, random_state=global_random_seed, n_init=1)
    km_none = clone(km).fit(input_data, sample_weight=None)
    km_ones = clone(km).fit(input_data, sample_weight=sample_weight)
    assert_array_equal(km_none.labels_, km_ones.labels_)
    assert_allclose(km_none.cluster_centers_, km_ones.cluster_centers_)

@pytest.mark.parametrize('input_data', [X] + X_as_any_csr, ids=data_containers_ids)
@pytest.mark.parametrize('Estimator', [KMeans, MiniBatchKMeans])
def test_scaled_weights(Estimator, input_data, global_random_seed):
    if False:
        i = 10
        return i + 15
    sample_weight = np.random.RandomState(global_random_seed).uniform(size=n_samples)
    km = Estimator(n_clusters=n_clusters, random_state=global_random_seed, n_init=1)
    km_orig = clone(km).fit(input_data, sample_weight=sample_weight)
    km_scaled = clone(km).fit(input_data, sample_weight=0.5 * sample_weight)
    assert_array_equal(km_orig.labels_, km_scaled.labels_)
    assert_allclose(km_orig.cluster_centers_, km_scaled.cluster_centers_)

def test_kmeans_elkan_iter_attribute():
    if False:
        while True:
            i = 10
    km = KMeans(algorithm='elkan', max_iter=1).fit(X)
    assert km.n_iter_ == 1

@pytest.mark.parametrize('array_constr', data_containers, ids=data_containers_ids)
def test_kmeans_empty_cluster_relocated(array_constr):
    if False:
        while True:
            i = 10
    X = array_constr([[-1], [1]])
    sample_weight = [1.9, 0.1]
    init = np.array([[-1], [10]])
    km = KMeans(n_clusters=2, init=init, n_init=1)
    km.fit(X, sample_weight=sample_weight)
    assert len(set(km.labels_)) == 2
    assert_allclose(km.cluster_centers_, [[-1], [1]])

@pytest.mark.parametrize('Estimator', [KMeans, MiniBatchKMeans])
def test_result_equal_in_diff_n_threads(Estimator, global_random_seed):
    if False:
        while True:
            i = 10
    rnd = np.random.RandomState(global_random_seed)
    X = rnd.normal(size=(50, 10))
    with threadpool_limits(limits=1, user_api='openmp'):
        result_1 = Estimator(n_clusters=n_clusters, random_state=global_random_seed).fit(X).labels_
    with threadpool_limits(limits=2, user_api='openmp'):
        result_2 = Estimator(n_clusters=n_clusters, random_state=global_random_seed).fit(X).labels_
    assert_array_equal(result_1, result_2)

def test_warning_elkan_1_cluster():
    if False:
        for i in range(10):
            print('nop')
    with pytest.warns(RuntimeWarning, match="algorithm='elkan' doesn't make sense for a single cluster"):
        KMeans(n_clusters=1, algorithm='elkan').fit(X)

@pytest.mark.parametrize('array_constr', data_containers, ids=data_containers_ids)
@pytest.mark.parametrize('algo', ['lloyd', 'elkan'])
def test_k_means_1_iteration(array_constr, algo, global_random_seed):
    if False:
        return 10
    X = np.random.RandomState(global_random_seed).uniform(size=(100, 5))
    init_centers = X[:5]
    X = array_constr(X)

    def py_kmeans(X, init):
        if False:
            print('Hello World!')
        new_centers = init.copy()
        labels = pairwise_distances_argmin(X, init)
        for label in range(init.shape[0]):
            new_centers[label] = X[labels == label].mean(axis=0)
        labels = pairwise_distances_argmin(X, new_centers)
        return (labels, new_centers)
    (py_labels, py_centers) = py_kmeans(X, init_centers)
    cy_kmeans = KMeans(n_clusters=5, n_init=1, init=init_centers, algorithm=algo, max_iter=1).fit(X)
    cy_labels = cy_kmeans.labels_
    cy_centers = cy_kmeans.cluster_centers_
    assert_array_equal(py_labels, cy_labels)
    assert_allclose(py_centers, cy_centers)

@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('squared', [True, False])
def test_euclidean_distance(dtype, squared, global_random_seed):
    if False:
        print('Hello World!')
    rng = np.random.RandomState(global_random_seed)
    a_sparse = sp.random(1, 100, density=0.5, format='csr', random_state=rng, dtype=dtype)
    a_dense = a_sparse.toarray().reshape(-1)
    b = rng.randn(100).astype(dtype, copy=False)
    b_squared_norm = (b ** 2).sum()
    expected = ((a_dense - b) ** 2).sum()
    expected = expected if squared else np.sqrt(expected)
    distance_dense_dense = _euclidean_dense_dense_wrapper(a_dense, b, squared)
    distance_sparse_dense = _euclidean_sparse_dense_wrapper(a_sparse.data, a_sparse.indices, b, b_squared_norm, squared)
    rtol = 0.0001 if dtype == np.float32 else 1e-07
    assert_allclose(distance_dense_dense, distance_sparse_dense, rtol=rtol)
    assert_allclose(distance_dense_dense, expected, rtol=rtol)
    assert_allclose(distance_sparse_dense, expected, rtol=rtol)

@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_inertia(dtype, global_random_seed):
    if False:
        return 10
    rng = np.random.RandomState(global_random_seed)
    X_sparse = sp.random(100, 10, density=0.5, format='csr', random_state=rng, dtype=dtype)
    X_dense = X_sparse.toarray()
    sample_weight = rng.randn(100).astype(dtype, copy=False)
    centers = rng.randn(5, 10).astype(dtype, copy=False)
    labels = rng.randint(5, size=100, dtype=np.int32)
    distances = ((X_dense - centers[labels]) ** 2).sum(axis=1)
    expected = np.sum(distances * sample_weight)
    inertia_dense = _inertia_dense(X_dense, sample_weight, centers, labels, n_threads=1)
    inertia_sparse = _inertia_sparse(X_sparse, sample_weight, centers, labels, n_threads=1)
    rtol = 0.0001 if dtype == np.float32 else 1e-06
    assert_allclose(inertia_dense, inertia_sparse, rtol=rtol)
    assert_allclose(inertia_dense, expected, rtol=rtol)
    assert_allclose(inertia_sparse, expected, rtol=rtol)
    label = 1
    mask = labels == label
    distances = ((X_dense[mask] - centers[label]) ** 2).sum(axis=1)
    expected = np.sum(distances * sample_weight[mask])
    inertia_dense = _inertia_dense(X_dense, sample_weight, centers, labels, n_threads=1, single_label=label)
    inertia_sparse = _inertia_sparse(X_sparse, sample_weight, centers, labels, n_threads=1, single_label=label)
    assert_allclose(inertia_dense, inertia_sparse, rtol=rtol)
    assert_allclose(inertia_dense, expected, rtol=rtol)
    assert_allclose(inertia_sparse, expected, rtol=rtol)

@pytest.mark.parametrize('Klass, default_n_init', [(KMeans, 10), (MiniBatchKMeans, 3)])
def test_change_n_init_future_warning(Klass, default_n_init):
    if False:
        print('Hello World!')
    est = Klass(n_init=1)
    with warnings.catch_warnings():
        warnings.simplefilter('error', FutureWarning)
        est.fit(X)
    default_n_init = 10 if Klass.__name__ == 'KMeans' else 3
    msg = f"The default value of `n_init` will change from {default_n_init} to 'auto' in 1.4"
    est = Klass()
    with pytest.warns(FutureWarning, match=msg):
        est.fit(X)

@pytest.mark.parametrize('Klass, default_n_init', [(KMeans, 10), (MiniBatchKMeans, 3)])
def test_n_init_auto(Klass, default_n_init):
    if False:
        for i in range(10):
            print('nop')
    est = Klass(n_init='auto', init='k-means++')
    est.fit(X)
    assert est._n_init == 1
    est = Klass(n_init='auto', init='random')
    est.fit(X)
    assert est._n_init == 10 if Klass.__name__ == 'KMeans' else 3

@pytest.mark.parametrize('Estimator', [KMeans, MiniBatchKMeans])
def test_sample_weight_unchanged(Estimator):
    if False:
        i = 10
        return i + 15
    X = np.array([[1], [2], [4]])
    sample_weight = np.array([0.5, 0.2, 0.3])
    Estimator(n_clusters=2, random_state=0).fit(X, sample_weight=sample_weight)
    assert_array_equal(sample_weight, np.array([0.5, 0.2, 0.3]))

@pytest.mark.parametrize('Estimator', [KMeans, MiniBatchKMeans])
@pytest.mark.parametrize('param, match', [({'n_clusters': n_samples + 1}, 'n_samples.* should be >= n_clusters'), ({'init': X[:2]}, 'The shape of the initial centers .* does not match the number of clusters'), ({'init': lambda X_, k, random_state: X_[:2]}, 'The shape of the initial centers .* does not match the number of clusters'), ({'init': X[:8, :2]}, 'The shape of the initial centers .* does not match the number of features of the data'), ({'init': lambda X_, k, random_state: X_[:8, :2]}, 'The shape of the initial centers .* does not match the number of features of the data')])
def test_wrong_params(Estimator, param, match):
    if False:
        i = 10
        return i + 15
    km = Estimator(n_init=1)
    with pytest.raises(ValueError, match=match):
        km.set_params(**param).fit(X)

@pytest.mark.parametrize('param, match', [({'x_squared_norms': X[:2]}, 'The length of x_squared_norms .* should be equal to the length of n_samples')])
def test_kmeans_plusplus_wrong_params(param, match):
    if False:
        return 10
    with pytest.raises(ValueError, match=match):
        kmeans_plusplus(X, n_clusters, **param)

@pytest.mark.parametrize('input_data', [X] + X_as_any_csr)
@pytest.mark.parametrize('dtype', [np.float64, np.float32])
def test_kmeans_plusplus_output(input_data, dtype, global_random_seed):
    if False:
        for i in range(10):
            print('nop')
    data = input_data.astype(dtype)
    (centers, indices) = kmeans_plusplus(data, n_clusters, random_state=global_random_seed)
    assert indices.shape[0] == n_clusters
    assert (indices >= 0).all()
    assert (indices <= data.shape[0]).all()
    assert centers.shape[0] == n_clusters
    assert (centers.max(axis=0) <= data.max(axis=0)).all()
    assert (centers.min(axis=0) >= data.min(axis=0)).all()
    assert_allclose(X[indices].astype(dtype), centers)

@pytest.mark.parametrize('x_squared_norms', [row_norms(X, squared=True), None])
def test_kmeans_plusplus_norms(x_squared_norms):
    if False:
        return 10
    (centers, indices) = kmeans_plusplus(X, n_clusters, x_squared_norms=x_squared_norms)
    assert_allclose(X[indices], centers)

def test_kmeans_plusplus_dataorder(global_random_seed):
    if False:
        while True:
            i = 10
    (centers_c, _) = kmeans_plusplus(X, n_clusters, random_state=global_random_seed)
    X_fortran = np.asfortranarray(X)
    (centers_fortran, _) = kmeans_plusplus(X_fortran, n_clusters, random_state=global_random_seed)
    assert_allclose(centers_c, centers_fortran)

def test_is_same_clustering():
    if False:
        while True:
            i = 10
    labels1 = np.array([1, 0, 0, 1, 2, 0, 2, 1], dtype=np.int32)
    assert _is_same_clustering(labels1, labels1, 3)
    labels2 = np.array([0, 2, 2, 0, 1, 2, 1, 0], dtype=np.int32)
    assert _is_same_clustering(labels1, labels2, 3)
    labels3 = np.array([1, 0, 0, 2, 2, 0, 2, 1], dtype=np.int32)
    assert not _is_same_clustering(labels1, labels3, 3)

@pytest.mark.parametrize('kwargs', ({'init': np.str_('k-means++')}, {'init': [[0, 0], [1, 1]], 'n_init': 1}))
def test_kmeans_with_array_like_or_np_scalar_init(kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Check that init works with numpy scalar strings.\n\n    Non-regression test for #21964.\n    '
    X = np.asarray([[0, 0], [0.5, 0], [0.5, 1], [1, 1]], dtype=np.float64)
    clustering = KMeans(n_clusters=2, **kwargs)
    clustering.fit(X)

@pytest.mark.parametrize('Klass, method', [(KMeans, 'fit'), (MiniBatchKMeans, 'fit'), (MiniBatchKMeans, 'partial_fit')])
def test_feature_names_out(Klass, method):
    if False:
        i = 10
        return i + 15
    'Check `feature_names_out` for `KMeans` and `MiniBatchKMeans`.'
    class_name = Klass.__name__.lower()
    kmeans = Klass()
    getattr(kmeans, method)(X)
    n_clusters = kmeans.cluster_centers_.shape[0]
    names_out = kmeans.get_feature_names_out()
    assert_array_equal([f'{class_name}{i}' for i in range(n_clusters)], names_out)

@pytest.mark.parametrize('csr_container', CSR_CONTAINERS + [None])
def test_predict_does_not_change_cluster_centers(csr_container):
    if False:
        i = 10
        return i + 15
    'Check that predict does not change cluster centers.\n\n    Non-regression test for gh-24253.\n    '
    (X, _) = make_blobs(n_samples=200, n_features=10, centers=10, random_state=0)
    if csr_container is not None:
        X = csr_container(X)
    kmeans = KMeans()
    y_pred1 = kmeans.fit_predict(X)
    kmeans.cluster_centers_ = create_memmap_backed_data(kmeans.cluster_centers_)
    kmeans.labels_ = create_memmap_backed_data(kmeans.labels_)
    y_pred2 = kmeans.predict(X)
    assert_array_equal(y_pred1, y_pred2)

@pytest.mark.parametrize('init', ['k-means++', 'random'])
def test_sample_weight_init(init, global_random_seed):
    if False:
        i = 10
        return i + 15
    "Check that sample weight is used during init.\n\n    `_init_centroids` is shared across all classes inheriting from _BaseKMeans so\n    it's enough to check for KMeans.\n    "
    rng = np.random.RandomState(global_random_seed)
    (X, _) = make_blobs(n_samples=200, n_features=10, centers=10, random_state=global_random_seed)
    x_squared_norms = row_norms(X, squared=True)
    kmeans = KMeans()
    clusters_weighted = kmeans._init_centroids(X=X, x_squared_norms=x_squared_norms, init=init, sample_weight=rng.uniform(size=X.shape[0]), n_centroids=5, random_state=np.random.RandomState(global_random_seed))
    clusters = kmeans._init_centroids(X=X, x_squared_norms=x_squared_norms, init=init, sample_weight=np.ones(X.shape[0]), n_centroids=5, random_state=np.random.RandomState(global_random_seed))
    with pytest.raises(AssertionError):
        assert_allclose(clusters_weighted, clusters)

@pytest.mark.parametrize('init', ['k-means++', 'random'])
def test_sample_weight_zero(init, global_random_seed):
    if False:
        for i in range(10):
            print('nop')
    "Check that if sample weight is 0, this sample won't be chosen.\n\n    `_init_centroids` is shared across all classes inheriting from _BaseKMeans so\n    it's enough to check for KMeans.\n    "
    rng = np.random.RandomState(global_random_seed)
    (X, _) = make_blobs(n_samples=100, n_features=5, centers=5, random_state=global_random_seed)
    sample_weight = rng.uniform(size=X.shape[0])
    sample_weight[::2] = 0
    x_squared_norms = row_norms(X, squared=True)
    kmeans = KMeans()
    clusters_weighted = kmeans._init_centroids(X=X, x_squared_norms=x_squared_norms, init=init, sample_weight=sample_weight, n_centroids=10, random_state=np.random.RandomState(global_random_seed))
    d = euclidean_distances(X[::2], clusters_weighted)
    assert not np.any(np.isclose(d, 0))