"""
Tests for HDBSCAN clustering algorithm
Based on the DBSCAN test code
"""
import numpy as np
import pytest
from scipy import stats
from scipy.spatial import distance
from sklearn.cluster import HDBSCAN
from sklearn.cluster._hdbscan._tree import CONDENSED_dtype, _condense_tree, _do_labelling
from sklearn.cluster._hdbscan.hdbscan import _OUTLIER_ENCODING
from sklearn.datasets import make_blobs
from sklearn.metrics import fowlkes_mallows_score
from sklearn.metrics.pairwise import _VALID_METRICS, euclidean_distances
from sklearn.neighbors import BallTree, KDTree
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
(X, y) = make_blobs(n_samples=200, random_state=10)
(X, y) = shuffle(X, y, random_state=7)
X = StandardScaler().fit_transform(X)
ALGORITHMS = ['kd_tree', 'ball_tree', 'brute', 'auto']
OUTLIER_SET = {-1} | {out['label'] for (_, out) in _OUTLIER_ENCODING.items()}

def check_label_quality(labels, threshold=0.99):
    if False:
        i = 10
        return i + 15
    n_clusters = len(set(labels) - OUTLIER_SET)
    assert n_clusters == 3
    assert fowlkes_mallows_score(labels, y) > threshold

@pytest.mark.parametrize('outlier_type', _OUTLIER_ENCODING)
def test_outlier_data(outlier_type):
    if False:
        print('Hello World!')
    '\n    Tests if np.inf and np.nan data are each treated as special outliers.\n    '
    outlier = {'infinite': np.inf, 'missing': np.nan}[outlier_type]
    prob_check = {'infinite': lambda x, y: x == y, 'missing': lambda x, y: np.isnan(x)}[outlier_type]
    label = _OUTLIER_ENCODING[outlier_type]['label']
    prob = _OUTLIER_ENCODING[outlier_type]['prob']
    X_outlier = X.copy()
    X_outlier[0] = [outlier, 1]
    X_outlier[5] = [outlier, outlier]
    model = HDBSCAN().fit(X_outlier)
    (missing_labels_idx,) = (model.labels_ == label).nonzero()
    assert_array_equal(missing_labels_idx, [0, 5])
    (missing_probs_idx,) = prob_check(model.probabilities_, prob).nonzero()
    assert_array_equal(missing_probs_idx, [0, 5])
    clean_indices = list(range(1, 5)) + list(range(6, 200))
    clean_model = HDBSCAN().fit(X_outlier[clean_indices])
    assert_array_equal(clean_model.labels_, model.labels_[clean_indices])

def test_hdbscan_distance_matrix():
    if False:
        i = 10
        return i + 15
    '\n    Tests that HDBSCAN works with precomputed distance matrices, and throws the\n    appropriate errors when needed.\n    '
    D = euclidean_distances(X)
    D_original = D.copy()
    labels = HDBSCAN(metric='precomputed', copy=True).fit_predict(D)
    assert_allclose(D, D_original)
    check_label_quality(labels)
    msg = 'The precomputed distance matrix.*has shape'
    with pytest.raises(ValueError, match=msg):
        HDBSCAN(metric='precomputed', copy=True).fit_predict(X)
    msg = 'The precomputed distance matrix.*values'
    D[0, 1] = 10
    D[1, 0] = 1
    with pytest.raises(ValueError, match=msg):
        HDBSCAN(metric='precomputed').fit_predict(D)

@pytest.mark.parametrize('sparse_constructor', [*CSR_CONTAINERS, *CSC_CONTAINERS])
def test_hdbscan_sparse_distance_matrix(sparse_constructor):
    if False:
        return 10
    '\n    Tests that HDBSCAN works with sparse distance matrices.\n    '
    D = distance.squareform(distance.pdist(X))
    D /= np.max(D)
    threshold = stats.scoreatpercentile(D.flatten(), 50)
    D[D >= threshold] = 0.0
    D = sparse_constructor(D)
    D.eliminate_zeros()
    labels = HDBSCAN(metric='precomputed').fit_predict(D)
    check_label_quality(labels)

def test_hdbscan_feature_array():
    if False:
        i = 10
        return i + 15
    '\n    Tests that HDBSCAN works with feature array, including an arbitrary\n    goodness of fit check. Note that the check is a simple heuristic.\n    '
    labels = HDBSCAN().fit_predict(X)
    check_label_quality(labels)

@pytest.mark.parametrize('algo', ALGORITHMS)
@pytest.mark.parametrize('metric', _VALID_METRICS)
def test_hdbscan_algorithms(algo, metric):
    if False:
        while True:
            i = 10
    '\n    Tests that HDBSCAN works with the expected combinations of algorithms and\n    metrics, or raises the expected errors.\n    '
    labels = HDBSCAN(algorithm=algo).fit_predict(X)
    check_label_quality(labels)
    if algo in ('brute', 'auto'):
        return
    ALGOS_TREES = {'kd_tree': KDTree, 'ball_tree': BallTree}
    metric_params = {'mahalanobis': {'V': np.eye(X.shape[1])}, 'seuclidean': {'V': np.ones(X.shape[1])}, 'minkowski': {'p': 2}, 'wminkowski': {'p': 2, 'w': np.ones(X.shape[1])}}.get(metric, None)
    hdb = HDBSCAN(algorithm=algo, metric=metric, metric_params=metric_params)
    if metric not in ALGOS_TREES[algo].valid_metrics:
        with pytest.raises(ValueError):
            hdb.fit(X)
    elif metric == 'wminkowski':
        with pytest.warns(FutureWarning):
            hdb.fit(X)
    else:
        hdb.fit(X)

def test_dbscan_clustering():
    if False:
        return 10
    '\n    Tests that HDBSCAN can generate a sufficiently accurate dbscan clustering.\n    This test is more of a sanity check than a rigorous evaluation.\n    '
    clusterer = HDBSCAN().fit(X)
    labels = clusterer.dbscan_clustering(0.3)
    check_label_quality(labels, threshold=0.92)

@pytest.mark.parametrize('cut_distance', (0.1, 0.5, 1))
def test_dbscan_clustering_outlier_data(cut_distance):
    if False:
        i = 10
        return i + 15
    '\n    Tests if np.inf and np.nan data are each treated as special outliers.\n    '
    missing_label = _OUTLIER_ENCODING['missing']['label']
    infinite_label = _OUTLIER_ENCODING['infinite']['label']
    X_outlier = X.copy()
    X_outlier[0] = [np.inf, 1]
    X_outlier[2] = [1, np.nan]
    X_outlier[5] = [np.inf, np.nan]
    model = HDBSCAN().fit(X_outlier)
    labels = model.dbscan_clustering(cut_distance=cut_distance)
    missing_labels_idx = np.flatnonzero(labels == missing_label)
    assert_array_equal(missing_labels_idx, [2, 5])
    infinite_labels_idx = np.flatnonzero(labels == infinite_label)
    assert_array_equal(infinite_labels_idx, [0])
    clean_idx = list(set(range(200)) - set(missing_labels_idx + infinite_labels_idx))
    clean_model = HDBSCAN().fit(X_outlier[clean_idx])
    clean_labels = clean_model.dbscan_clustering(cut_distance=cut_distance)
    assert_array_equal(clean_labels, labels[clean_idx])

def test_hdbscan_best_balltree_metric():
    if False:
        for i in range(10):
            print('nop')
    '\n    Tests that HDBSCAN using `BallTree` works.\n    '
    labels = HDBSCAN(metric='seuclidean', metric_params={'V': np.ones(X.shape[1])}).fit_predict(X)
    check_label_quality(labels)

def test_hdbscan_no_clusters():
    if False:
        i = 10
        return i + 15
    '\n    Tests that HDBSCAN correctly does not generate a valid cluster when the\n    `min_cluster_size` is too large for the data.\n    '
    labels = HDBSCAN(min_cluster_size=len(X) - 1).fit_predict(X)
    assert set(labels).issubset(OUTLIER_SET)

def test_hdbscan_min_cluster_size():
    if False:
        return 10
    '\n    Test that the smallest non-noise cluster has at least `min_cluster_size`\n    many points\n    '
    for min_cluster_size in range(2, len(X), 1):
        labels = HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(X)
        true_labels = [label for label in labels if label != -1]
        if len(true_labels) != 0:
            assert np.min(np.bincount(true_labels)) >= min_cluster_size

def test_hdbscan_callable_metric():
    if False:
        while True:
            i = 10
    '\n    Tests that HDBSCAN works when passed a callable metric.\n    '
    metric = distance.euclidean
    labels = HDBSCAN(metric=metric).fit_predict(X)
    check_label_quality(labels)

@pytest.mark.parametrize('tree', ['kd_tree', 'ball_tree'])
def test_hdbscan_precomputed_non_brute(tree):
    if False:
        while True:
            i = 10
    '\n    Tests that HDBSCAN correctly raises an error when passing precomputed data\n    while requesting a tree-based algorithm.\n    '
    hdb = HDBSCAN(metric='precomputed', algorithm=tree)
    msg = 'precomputed is not a valid metric for'
    with pytest.raises(ValueError, match=msg):
        hdb.fit(X)

@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_hdbscan_sparse(csr_container):
    if False:
        for i in range(10):
            print('nop')
    '\n    Tests that HDBSCAN works correctly when passing sparse feature data.\n    Evaluates correctness by comparing against the same data passed as a dense\n    array.\n    '
    dense_labels = HDBSCAN().fit(X).labels_
    check_label_quality(dense_labels)
    _X_sparse = csr_container(X)
    X_sparse = _X_sparse.copy()
    sparse_labels = HDBSCAN().fit(X_sparse).labels_
    assert_array_equal(dense_labels, sparse_labels)
    for (outlier_val, outlier_type) in ((np.inf, 'infinite'), (np.nan, 'missing')):
        X_dense = X.copy()
        X_dense[0, 0] = outlier_val
        dense_labels = HDBSCAN().fit(X_dense).labels_
        check_label_quality(dense_labels)
        assert dense_labels[0] == _OUTLIER_ENCODING[outlier_type]['label']
        X_sparse = _X_sparse.copy()
        X_sparse[0, 0] = outlier_val
        sparse_labels = HDBSCAN().fit(X_sparse).labels_
        assert_array_equal(dense_labels, sparse_labels)
    msg = 'Sparse data matrices only support algorithm `brute`.'
    with pytest.raises(ValueError, match=msg):
        HDBSCAN(metric='euclidean', algorithm='ball_tree').fit(X_sparse)

@pytest.mark.parametrize('algorithm', ALGORITHMS)
def test_hdbscan_centers(algorithm):
    if False:
        print('Hello World!')
    '\n    Tests that HDBSCAN centers are calculated and stored properly, and are\n    accurate to the data.\n    '
    centers = [(0.0, 0.0), (3.0, 3.0)]
    (H, _) = make_blobs(n_samples=1000, random_state=0, centers=centers, cluster_std=0.5)
    hdb = HDBSCAN(store_centers='both').fit(H)
    for (center, centroid, medoid) in zip(centers, hdb.centroids_, hdb.medoids_):
        assert_allclose(center, centroid, rtol=1, atol=0.05)
        assert_allclose(center, medoid, rtol=1, atol=0.05)
    hdb = HDBSCAN(algorithm=algorithm, store_centers='both', min_cluster_size=X.shape[0]).fit(X)
    assert hdb.centroids_.shape[0] == 0
    assert hdb.medoids_.shape[0] == 0

def test_hdbscan_allow_single_cluster_with_epsilon():
    if False:
        i = 10
        return i + 15
    '\n    Tests that HDBSCAN single-cluster selection with epsilon works correctly.\n    '
    rng = np.random.RandomState(0)
    no_structure = rng.rand(150, 2)
    labels = HDBSCAN(min_cluster_size=5, cluster_selection_epsilon=0.0, cluster_selection_method='eom', allow_single_cluster=True).fit_predict(no_structure)
    (unique_labels, counts) = np.unique(labels, return_counts=True)
    assert len(unique_labels) == 2
    assert counts[unique_labels == -1] > 30
    labels = HDBSCAN(min_cluster_size=5, cluster_selection_epsilon=0.18, cluster_selection_method='eom', allow_single_cluster=True, algorithm='kd_tree').fit_predict(no_structure)
    (unique_labels, counts) = np.unique(labels, return_counts=True)
    assert len(unique_labels) == 2
    assert counts[unique_labels == -1] == 2

def test_hdbscan_better_than_dbscan():
    if False:
        for i in range(10):
            print('nop')
    '\n    Validate that HDBSCAN can properly cluster this difficult synthetic\n    dataset. Note that DBSCAN fails on this (see HDBSCAN plotting\n    example)\n    '
    centers = [[-0.85, -0.85], [-0.85, 0.85], [3, 3], [3, -3]]
    (X, y) = make_blobs(n_samples=750, centers=centers, cluster_std=[0.2, 0.35, 1.35, 1.35], random_state=0)
    labels = HDBSCAN().fit(X).labels_
    n_clusters = len(set(labels)) - int(-1 in labels)
    assert n_clusters == 4
    fowlkes_mallows_score(labels, y) > 0.99

@pytest.mark.parametrize('kwargs, X', [({'metric': 'precomputed'}, np.array([[1, np.inf], [np.inf, 1]])), ({'metric': 'precomputed'}, [[1, 2], [2, 1]]), ({}, [[1, 2], [3, 4]])])
def test_hdbscan_usable_inputs(X, kwargs):
    if False:
        while True:
            i = 10
    '\n    Tests that HDBSCAN works correctly for array-likes and precomputed inputs\n    with non-finite points.\n    '
    HDBSCAN(min_samples=1, **kwargs).fit(X)

@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_hdbscan_sparse_distances_too_few_nonzero(csr_container):
    if False:
        i = 10
        return i + 15
    '\n    Tests that HDBSCAN raises the correct error when there are too few\n    non-zero distances.\n    '
    X = csr_container(np.zeros((10, 10)))
    msg = 'There exists points with fewer than'
    with pytest.raises(ValueError, match=msg):
        HDBSCAN(metric='precomputed').fit(X)

def test_hdbscan_tree_invalid_metric():
    if False:
        i = 10
        return i + 15
    '\n    Tests that HDBSCAN correctly raises an error for invalid metric choices.\n    '
    metric_callable = lambda x: x
    msg = '.* is not a valid metric for a .*-based algorithm\\. Please select a different metric\\.'
    with pytest.raises(ValueError, match=msg):
        HDBSCAN(algorithm='kd_tree', metric=metric_callable).fit(X)
    with pytest.raises(ValueError, match=msg):
        HDBSCAN(algorithm='ball_tree', metric=metric_callable).fit(X)
    metrics_not_kd = list(set(BallTree.valid_metrics) - set(KDTree.valid_metrics))
    if len(metrics_not_kd) > 0:
        with pytest.raises(ValueError, match=msg):
            HDBSCAN(algorithm='kd_tree', metric=metrics_not_kd[0]).fit(X)

def test_hdbscan_too_many_min_samples():
    if False:
        for i in range(10):
            print('nop')
    '\n    Tests that HDBSCAN correctly raises an error when setting `min_samples`\n    larger than the number of samples.\n    '
    hdb = HDBSCAN(min_samples=len(X) + 1)
    msg = 'min_samples (.*) must be at most'
    with pytest.raises(ValueError, match=msg):
        hdb.fit(X)

def test_hdbscan_precomputed_dense_nan():
    if False:
        i = 10
        return i + 15
    '\n    Tests that HDBSCAN correctly raises an error when providing precomputed\n    distances with `np.nan` values.\n    '
    X_nan = X.copy()
    X_nan[0, 0] = np.nan
    msg = 'np.nan values found in precomputed-dense'
    hdb = HDBSCAN(metric='precomputed')
    with pytest.raises(ValueError, match=msg):
        hdb.fit(X_nan)

@pytest.mark.parametrize('allow_single_cluster', [True, False])
@pytest.mark.parametrize('epsilon', [0, 0.1])
def test_labelling_distinct(global_random_seed, allow_single_cluster, epsilon):
    if False:
        i = 10
        return i + 15
    '\n    Tests that the `_do_labelling` helper function correctly assigns labels.\n    '
    n_samples = 48
    (X, y) = make_blobs(n_samples, random_state=global_random_seed, centers=[[0, 0], [10, 0], [0, 10]])
    est = HDBSCAN().fit(X)
    condensed_tree = _condense_tree(est._single_linkage_tree_, min_cluster_size=est.min_cluster_size)
    clusters = {n_samples + 2, n_samples + 3, n_samples + 4}
    cluster_label_map = {n_samples + 2: 0, n_samples + 3: 1, n_samples + 4: 2}
    labels = _do_labelling(condensed_tree=condensed_tree, clusters=clusters, cluster_label_map=cluster_label_map, allow_single_cluster=allow_single_cluster, cluster_selection_epsilon=epsilon)
    first_with_label = {_y: np.where(y == _y)[0][0] for _y in list(set(y))}
    y_to_labels = {_y: labels[first_with_label[_y]] for _y in list(set(y))}
    aligned_target = np.vectorize(y_to_labels.get)(y)
    assert_array_equal(labels, aligned_target)

def test_labelling_thresholding():
    if False:
        i = 10
        return i + 15
    '\n    Tests that the `_do_labelling` helper function correctly thresholds the\n    incoming lambda values given various `cluster_selection_epsilon` values.\n    '
    n_samples = 5
    MAX_LAMBDA = 1.5
    condensed_tree = np.array([(5, 2, MAX_LAMBDA, 1), (5, 1, 0.1, 1), (5, 0, MAX_LAMBDA, 1), (5, 3, 0.2, 1), (5, 4, 0.3, 1)], dtype=CONDENSED_dtype)
    labels = _do_labelling(condensed_tree=condensed_tree, clusters={n_samples}, cluster_label_map={n_samples: 0, n_samples + 1: 1}, allow_single_cluster=True, cluster_selection_epsilon=1)
    num_noise = condensed_tree['value'] < 1
    assert sum(num_noise) == sum(labels == -1)
    labels = _do_labelling(condensed_tree=condensed_tree, clusters={n_samples}, cluster_label_map={n_samples: 0, n_samples + 1: 1}, allow_single_cluster=True, cluster_selection_epsilon=0)
    num_noise = condensed_tree['value'] < MAX_LAMBDA
    assert sum(num_noise) == sum(labels == -1)

def test_hdbscan_warning_on_deprecated_algorithm_name():
    if False:
        print('Hello World!')
    msg = "`algorithm='kdtree'`has been deprecated in 1.4 and will be renamed to'kd_tree'`in 1.6. To keep the past behaviour, set `algorithm='kd_tree'`."
    with pytest.warns(FutureWarning, match=msg):
        HDBSCAN(algorithm='kdtree').fit(X)
    msg = "`algorithm='balltree'`has been deprecated in 1.4 and will be renamed to'ball_tree'`in 1.6. To keep the past behaviour, set `algorithm='ball_tree'`."
    with pytest.warns(FutureWarning, match=msg):
        HDBSCAN(algorithm='balltree').fit(X)