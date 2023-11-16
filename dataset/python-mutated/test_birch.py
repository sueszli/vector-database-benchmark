"""
Tests for the birch clustering algorithm.
"""
import numpy as np
import pytest
from sklearn.cluster import AgglomerativeClustering, Birch
from sklearn.cluster.tests.common import generate_clustered_data
from sklearn.datasets import make_blobs
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import pairwise_distances_argmin, v_measure_score
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.fixes import CSR_CONTAINERS

def test_n_samples_leaves_roots(global_random_seed, global_dtype):
    if False:
        print('Hello World!')
    (X, y) = make_blobs(n_samples=10, random_state=global_random_seed)
    X = X.astype(global_dtype, copy=False)
    brc = Birch()
    brc.fit(X)
    n_samples_root = sum([sc.n_samples_ for sc in brc.root_.subclusters_])
    n_samples_leaves = sum([sc.n_samples_ for leaf in brc._get_leaves() for sc in leaf.subclusters_])
    assert n_samples_leaves == X.shape[0]
    assert n_samples_root == X.shape[0]

def test_partial_fit(global_random_seed, global_dtype):
    if False:
        while True:
            i = 10
    (X, y) = make_blobs(n_samples=100, random_state=global_random_seed)
    X = X.astype(global_dtype, copy=False)
    brc = Birch(n_clusters=3)
    brc.fit(X)
    brc_partial = Birch(n_clusters=None)
    brc_partial.partial_fit(X[:50])
    brc_partial.partial_fit(X[50:])
    assert_allclose(brc_partial.subcluster_centers_, brc.subcluster_centers_)
    brc_partial.set_params(n_clusters=3)
    brc_partial.partial_fit(None)
    assert_array_equal(brc_partial.subcluster_labels_, brc.subcluster_labels_)

def test_birch_predict(global_random_seed, global_dtype):
    if False:
        i = 10
        return i + 15
    rng = np.random.RandomState(global_random_seed)
    X = generate_clustered_data(n_clusters=3, n_features=3, n_samples_per_cluster=10)
    X = X.astype(global_dtype, copy=False)
    shuffle_indices = np.arange(30)
    rng.shuffle(shuffle_indices)
    X_shuffle = X[shuffle_indices, :]
    brc = Birch(n_clusters=4, threshold=1.0)
    brc.fit(X_shuffle)
    assert brc.subcluster_centers_.dtype == global_dtype
    assert_array_equal(brc.labels_, brc.predict(X_shuffle))
    centroids = brc.subcluster_centers_
    nearest_centroid = brc.subcluster_labels_[pairwise_distances_argmin(X_shuffle, centroids)]
    assert_allclose(v_measure_score(nearest_centroid, brc.labels_), 1.0)

def test_n_clusters(global_random_seed, global_dtype):
    if False:
        while True:
            i = 10
    (X, y) = make_blobs(n_samples=100, centers=10, random_state=global_random_seed)
    X = X.astype(global_dtype, copy=False)
    brc1 = Birch(n_clusters=10)
    brc1.fit(X)
    assert len(brc1.subcluster_centers_) > 10
    assert len(np.unique(brc1.labels_)) == 10
    gc = AgglomerativeClustering(n_clusters=10)
    brc2 = Birch(n_clusters=gc)
    brc2.fit(X)
    assert_array_equal(brc1.subcluster_labels_, brc2.subcluster_labels_)
    assert_array_equal(brc1.labels_, brc2.labels_)
    brc4 = Birch(threshold=10000.0)
    with pytest.warns(ConvergenceWarning):
        brc4.fit(X)

@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_sparse_X(global_random_seed, global_dtype, csr_container):
    if False:
        i = 10
        return i + 15
    (X, y) = make_blobs(n_samples=100, centers=10, random_state=global_random_seed)
    X = X.astype(global_dtype, copy=False)
    brc = Birch(n_clusters=10)
    brc.fit(X)
    csr = csr_container(X)
    brc_sparse = Birch(n_clusters=10)
    brc_sparse.fit(csr)
    assert brc_sparse.subcluster_centers_.dtype == global_dtype
    assert_array_equal(brc.labels_, brc_sparse.labels_)
    assert_allclose(brc.subcluster_centers_, brc_sparse.subcluster_centers_)

def test_partial_fit_second_call_error_checks():
    if False:
        print('Hello World!')
    (X, y) = make_blobs(n_samples=100)
    brc = Birch(n_clusters=3)
    brc.partial_fit(X, y)
    msg = 'X has 1 features, but Birch is expecting 2 features'
    with pytest.raises(ValueError, match=msg):
        brc.partial_fit(X[:, [0]], y)

def check_branching_factor(node, branching_factor):
    if False:
        return 10
    subclusters = node.subclusters_
    assert branching_factor >= len(subclusters)
    for cluster in subclusters:
        if cluster.child_:
            check_branching_factor(cluster.child_, branching_factor)

def test_branching_factor(global_random_seed, global_dtype):
    if False:
        while True:
            i = 10
    (X, y) = make_blobs(random_state=global_random_seed)
    X = X.astype(global_dtype, copy=False)
    branching_factor = 9
    brc = Birch(n_clusters=None, branching_factor=branching_factor, threshold=0.01)
    brc.fit(X)
    check_branching_factor(brc.root_, branching_factor)
    brc = Birch(n_clusters=3, branching_factor=branching_factor, threshold=0.01)
    brc.fit(X)
    check_branching_factor(brc.root_, branching_factor)

def check_threshold(birch_instance, threshold):
    if False:
        while True:
            i = 10
    'Use the leaf linked list for traversal'
    current_leaf = birch_instance.dummy_leaf_.next_leaf_
    while current_leaf:
        subclusters = current_leaf.subclusters_
        for sc in subclusters:
            assert threshold >= sc.radius
        current_leaf = current_leaf.next_leaf_

def test_threshold(global_random_seed, global_dtype):
    if False:
        i = 10
        return i + 15
    (X, y) = make_blobs(n_samples=80, centers=4, random_state=global_random_seed)
    X = X.astype(global_dtype, copy=False)
    brc = Birch(threshold=0.5, n_clusters=None)
    brc.fit(X)
    check_threshold(brc, 0.5)
    brc = Birch(threshold=5.0, n_clusters=None)
    brc.fit(X)
    check_threshold(brc, 5.0)

def test_birch_n_clusters_long_int():
    if False:
        i = 10
        return i + 15
    (X, _) = make_blobs(random_state=0)
    n_clusters = np.int64(5)
    Birch(n_clusters=n_clusters).fit(X)

def test_feature_names_out():
    if False:
        return 10
    'Check `get_feature_names_out` for `Birch`.'
    (X, _) = make_blobs(n_samples=80, n_features=4, random_state=0)
    brc = Birch(n_clusters=4)
    brc.fit(X)
    n_clusters = brc.subcluster_centers_.shape[0]
    names_out = brc.get_feature_names_out()
    assert_array_equal([f'birch{i}' for i in range(n_clusters)], names_out)

def test_transform_match_across_dtypes(global_random_seed):
    if False:
        while True:
            i = 10
    (X, _) = make_blobs(n_samples=80, n_features=4, random_state=global_random_seed)
    brc = Birch(n_clusters=4, threshold=1.1)
    Y_64 = brc.fit_transform(X)
    Y_32 = brc.fit_transform(X.astype(np.float32))
    assert_allclose(Y_64, Y_32, atol=1e-06)

def test_subcluster_dtype(global_dtype):
    if False:
        i = 10
        return i + 15
    X = make_blobs(n_samples=80, n_features=4, random_state=0)[0].astype(global_dtype, copy=False)
    brc = Birch(n_clusters=4)
    assert brc.fit(X).subcluster_centers_.dtype == global_dtype

def test_both_subclusters_updated():
    if False:
        while True:
            i = 10
    'Check that both subclusters are updated when a node a split, even when there are\n    duplicated data points. Non-regression test for #23269.\n    '
    X = np.array([[-2.6192791, -1.5053215], [-2.9993038, -1.6863596], [-2.3724914, -1.3438171], [-2.336792, -1.3417323], [-2.4089134, -1.3290224], [-2.3724914, -1.3438171], [-3.364009, -1.8846745], [-2.3724914, -1.3438171], [-2.617677, -1.5003285], [-2.2960556, -1.3260119], [-2.3724914, -1.3438171], [-2.5459878, -1.4533926], [-2.25979, -1.3003055], [-2.4089134, -1.3290224], [-2.3724914, -1.3438171], [-2.4089134, -1.3290224], [-2.5459878, -1.4533926], [-2.3724914, -1.3438171], [-2.9720619, -1.7058647], [-2.336792, -1.3417323], [-2.3724914, -1.3438171]], dtype=np.float32)
    Birch(branching_factor=5, threshold=1e-05, n_clusters=None).fit(X)