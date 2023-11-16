"""
Several basic tests for hierarchical clustering procedures

"""
import itertools
import shutil
from functools import partial
from tempfile import mkdtemp
import numpy as np
import pytest
from scipy.cluster import hierarchy
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import AgglomerativeClustering, FeatureAgglomeration, ward_tree
from sklearn.cluster._agglomerative import _TREE_BUILDERS, _fix_connectivity, _hc_cut, linkage_tree
from sklearn.cluster._hierarchical_fast import average_merge, max_merge, mst_linkage_core
from sklearn.datasets import make_circles, make_moons
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.metrics import DistanceMetric
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.pairwise import PAIRED_DISTANCES, cosine_distances, manhattan_distances, pairwise_distances
from sklearn.metrics.tests.test_dist_metrics import METRICS_DEFAULT_PARAMS
from sklearn.neighbors import kneighbors_graph
from sklearn.utils._fast_dict import IntFloatDict
from sklearn.utils._testing import assert_almost_equal, assert_array_almost_equal, assert_array_equal, create_memmap_backed_data, ignore_warnings
from sklearn.utils.fixes import LIL_CONTAINERS

def test_linkage_misc():
    if False:
        print('Hello World!')
    rng = np.random.RandomState(42)
    X = rng.normal(size=(5, 5))
    with pytest.raises(ValueError):
        linkage_tree(X, linkage='foo')
    with pytest.raises(ValueError):
        linkage_tree(X, connectivity=np.ones((4, 4)))
    FeatureAgglomeration().fit(X)
    dis = cosine_distances(X)
    res = linkage_tree(dis, affinity='precomputed')
    assert_array_equal(res[0], linkage_tree(X, affinity='cosine')[0])
    res = linkage_tree(X, affinity=manhattan_distances)
    assert_array_equal(res[0], linkage_tree(X, affinity='manhattan')[0])

def test_structured_linkage_tree():
    if False:
        i = 10
        return i + 15
    rng = np.random.RandomState(0)
    mask = np.ones([10, 10], dtype=bool)
    mask[4:7, 4:7] = 0
    X = rng.randn(50, 100)
    connectivity = grid_to_graph(*mask.shape)
    for tree_builder in _TREE_BUILDERS.values():
        (children, n_components, n_leaves, parent) = tree_builder(X.T, connectivity=connectivity)
        n_nodes = 2 * X.shape[1] - 1
        assert len(children) + n_leaves == n_nodes
        with pytest.raises(ValueError):
            tree_builder(X.T, connectivity=np.ones((4, 4)))
        with pytest.raises(ValueError):
            tree_builder(X.T[:0], connectivity=connectivity)

def test_unstructured_linkage_tree():
    if False:
        i = 10
        return i + 15
    rng = np.random.RandomState(0)
    X = rng.randn(50, 100)
    for this_X in (X, X[0]):
        with ignore_warnings():
            with pytest.warns(UserWarning):
                (children, n_nodes, n_leaves, parent) = ward_tree(this_X.T, n_clusters=10)
        n_nodes = 2 * X.shape[1] - 1
        assert len(children) + n_leaves == n_nodes
    for tree_builder in _TREE_BUILDERS.values():
        for this_X in (X, X[0]):
            with ignore_warnings():
                with pytest.warns(UserWarning):
                    (children, n_nodes, n_leaves, parent) = tree_builder(this_X.T, n_clusters=10)
            n_nodes = 2 * X.shape[1] - 1
            assert len(children) + n_leaves == n_nodes

def test_height_linkage_tree():
    if False:
        while True:
            i = 10
    rng = np.random.RandomState(0)
    mask = np.ones([10, 10], dtype=bool)
    X = rng.randn(50, 100)
    connectivity = grid_to_graph(*mask.shape)
    for linkage_func in _TREE_BUILDERS.values():
        (children, n_nodes, n_leaves, parent) = linkage_func(X.T, connectivity=connectivity)
        n_nodes = 2 * X.shape[1] - 1
        assert len(children) + n_leaves == n_nodes

def test_zero_cosine_linkage_tree():
    if False:
        for i in range(10):
            print('nop')
    X = np.array([[0, 1], [0, 0]])
    msg = 'Cosine affinity cannot be used when X contains zero vectors'
    with pytest.raises(ValueError, match=msg):
        linkage_tree(X, affinity='cosine')

@pytest.mark.parametrize('n_clusters, distance_threshold', [(None, 0.5), (10, None)])
@pytest.mark.parametrize('compute_distances', [True, False])
@pytest.mark.parametrize('linkage', ['ward', 'complete', 'average', 'single'])
def test_agglomerative_clustering_distances(n_clusters, compute_distances, distance_threshold, linkage):
    if False:
        for i in range(10):
            print('nop')
    rng = np.random.RandomState(0)
    mask = np.ones([10, 10], dtype=bool)
    n_samples = 100
    X = rng.randn(n_samples, 50)
    connectivity = grid_to_graph(*mask.shape)
    clustering = AgglomerativeClustering(n_clusters=n_clusters, connectivity=connectivity, linkage=linkage, distance_threshold=distance_threshold, compute_distances=compute_distances)
    clustering.fit(X)
    if compute_distances or distance_threshold is not None:
        assert hasattr(clustering, 'distances_')
        n_children = clustering.children_.shape[0]
        n_nodes = n_children + 1
        assert clustering.distances_.shape == (n_nodes - 1,)
    else:
        assert not hasattr(clustering, 'distances_')

@pytest.mark.parametrize('lil_container', LIL_CONTAINERS)
def test_agglomerative_clustering(global_random_seed, lil_container):
    if False:
        print('Hello World!')
    rng = np.random.RandomState(global_random_seed)
    mask = np.ones([10, 10], dtype=bool)
    n_samples = 100
    X = rng.randn(n_samples, 50)
    connectivity = grid_to_graph(*mask.shape)
    for linkage in ('ward', 'complete', 'average', 'single'):
        clustering = AgglomerativeClustering(n_clusters=10, connectivity=connectivity, linkage=linkage)
        clustering.fit(X)
        try:
            tempdir = mkdtemp()
            clustering = AgglomerativeClustering(n_clusters=10, connectivity=connectivity, memory=tempdir, linkage=linkage)
            clustering.fit(X)
            labels = clustering.labels_
            assert np.size(np.unique(labels)) == 10
        finally:
            shutil.rmtree(tempdir)
        clustering = AgglomerativeClustering(n_clusters=10, connectivity=connectivity, linkage=linkage)
        clustering.compute_full_tree = False
        clustering.fit(X)
        assert_almost_equal(normalized_mutual_info_score(clustering.labels_, labels), 1)
        clustering.connectivity = None
        clustering.fit(X)
        assert np.size(np.unique(clustering.labels_)) == 10
        clustering = AgglomerativeClustering(n_clusters=10, connectivity=lil_container(connectivity.toarray()[:10, :10]), linkage=linkage)
        with pytest.raises(ValueError):
            clustering.fit(X)
    clustering = AgglomerativeClustering(n_clusters=10, connectivity=connectivity.toarray(), metric='manhattan', linkage='ward')
    with pytest.raises(ValueError):
        clustering.fit(X)
    for metric in PAIRED_DISTANCES.keys():
        clustering = AgglomerativeClustering(n_clusters=10, connectivity=np.ones((n_samples, n_samples)), metric=metric, linkage='complete')
        clustering.fit(X)
        clustering2 = AgglomerativeClustering(n_clusters=10, connectivity=None, metric=metric, linkage='complete')
        clustering2.fit(X)
        assert_almost_equal(normalized_mutual_info_score(clustering2.labels_, clustering.labels_), 1)
    clustering = AgglomerativeClustering(n_clusters=10, connectivity=connectivity, linkage='complete')
    clustering.fit(X)
    X_dist = pairwise_distances(X)
    clustering2 = AgglomerativeClustering(n_clusters=10, connectivity=connectivity, metric='precomputed', linkage='complete')
    clustering2.fit(X_dist)
    assert_array_equal(clustering.labels_, clustering2.labels_)

def test_agglomerative_clustering_memory_mapped():
    if False:
        i = 10
        return i + 15
    'AgglomerativeClustering must work on mem-mapped dataset.\n\n    Non-regression test for issue #19875.\n    '
    rng = np.random.RandomState(0)
    Xmm = create_memmap_backed_data(rng.randn(50, 100))
    AgglomerativeClustering(metric='euclidean', linkage='single').fit(Xmm)

def test_ward_agglomeration(global_random_seed):
    if False:
        print('Hello World!')
    rng = np.random.RandomState(global_random_seed)
    mask = np.ones([10, 10], dtype=bool)
    X = rng.randn(50, 100)
    connectivity = grid_to_graph(*mask.shape)
    agglo = FeatureAgglomeration(n_clusters=5, connectivity=connectivity)
    agglo.fit(X)
    assert np.size(np.unique(agglo.labels_)) == 5
    X_red = agglo.transform(X)
    assert X_red.shape[1] == 5
    X_full = agglo.inverse_transform(X_red)
    assert np.unique(X_full[0]).size == 5
    assert_array_almost_equal(agglo.transform(X_full), X_red)
    with pytest.raises(ValueError):
        agglo.fit(X[:0])

def test_single_linkage_clustering():
    if False:
        i = 10
        return i + 15
    (moons, moon_labels) = make_moons(noise=0.05, random_state=42)
    clustering = AgglomerativeClustering(n_clusters=2, linkage='single')
    clustering.fit(moons)
    assert_almost_equal(normalized_mutual_info_score(clustering.labels_, moon_labels), 1)
    (circles, circle_labels) = make_circles(factor=0.5, noise=0.025, random_state=42)
    clustering = AgglomerativeClustering(n_clusters=2, linkage='single')
    clustering.fit(circles)
    assert_almost_equal(normalized_mutual_info_score(clustering.labels_, circle_labels), 1)

def assess_same_labelling(cut1, cut2):
    if False:
        print('Hello World!')
    'Util for comparison with scipy'
    co_clust = []
    for cut in [cut1, cut2]:
        n = len(cut)
        k = cut.max() + 1
        ecut = np.zeros((n, k))
        ecut[np.arange(n), cut] = 1
        co_clust.append(np.dot(ecut, ecut.T))
    assert (co_clust[0] == co_clust[1]).all()

def test_sparse_scikit_vs_scipy(global_random_seed):
    if False:
        for i in range(10):
            print('nop')
    (n, p, k) = (10, 5, 3)
    rng = np.random.RandomState(global_random_seed)
    connectivity = np.ones((n, n))
    for linkage in _TREE_BUILDERS.keys():
        for i in range(5):
            X = 0.1 * rng.normal(size=(n, p))
            X -= 4.0 * np.arange(n)[:, np.newaxis]
            X -= X.mean(axis=1)[:, np.newaxis]
            out = hierarchy.linkage(X, method=linkage)
            children_ = out[:, :2].astype(int, copy=False)
            (children, _, n_leaves, _) = _TREE_BUILDERS[linkage](X, connectivity=connectivity)
            children.sort(axis=1)
            assert_array_equal(children, children_, 'linkage tree differs from scipy impl for linkage: ' + linkage)
            cut = _hc_cut(k, children, n_leaves)
            cut_ = _hc_cut(k, children_, n_leaves)
            assess_same_labelling(cut, cut_)
    with pytest.raises(ValueError):
        _hc_cut(n_leaves + 1, children, n_leaves)

def test_vector_scikit_single_vs_scipy_single(global_random_seed):
    if False:
        for i in range(10):
            print('nop')
    (n_samples, n_features, n_clusters) = (10, 5, 3)
    rng = np.random.RandomState(global_random_seed)
    X = 0.1 * rng.normal(size=(n_samples, n_features))
    X -= 4.0 * np.arange(n_samples)[:, np.newaxis]
    X -= X.mean(axis=1)[:, np.newaxis]
    out = hierarchy.linkage(X, method='single')
    children_scipy = out[:, :2].astype(int)
    (children, _, n_leaves, _) = _TREE_BUILDERS['single'](X)
    children.sort(axis=1)
    assert_array_equal(children, children_scipy, 'linkage tree differs from scipy impl for single linkage.')
    cut = _hc_cut(n_clusters, children, n_leaves)
    cut_scipy = _hc_cut(n_clusters, children_scipy, n_leaves)
    assess_same_labelling(cut, cut_scipy)

@pytest.mark.parametrize('metric_param_grid', METRICS_DEFAULT_PARAMS)
def test_mst_linkage_core_memory_mapped(metric_param_grid):
    if False:
        while True:
            i = 10
    'The MST-LINKAGE-CORE algorithm must work on mem-mapped dataset.\n\n    Non-regression test for issue #19875.\n    '
    rng = np.random.RandomState(seed=1)
    X = rng.normal(size=(20, 4))
    Xmm = create_memmap_backed_data(X)
    (metric, param_grid) = metric_param_grid
    keys = param_grid.keys()
    for vals in itertools.product(*param_grid.values()):
        kwargs = dict(zip(keys, vals))
        distance_metric = DistanceMetric.get_metric(metric, **kwargs)
        mst = mst_linkage_core(X, distance_metric)
        mst_mm = mst_linkage_core(Xmm, distance_metric)
        np.testing.assert_equal(mst, mst_mm)

def test_identical_points():
    if False:
        while True:
            i = 10
    X = np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1], [2, 2, 2], [2, 2, 2]])
    true_labels = np.array([0, 0, 1, 1, 2, 2])
    connectivity = kneighbors_graph(X, n_neighbors=3, include_self=False)
    connectivity = 0.5 * (connectivity + connectivity.T)
    (connectivity, n_components) = _fix_connectivity(X, connectivity, 'euclidean')
    for linkage in ('single', 'average', 'average', 'ward'):
        clustering = AgglomerativeClustering(n_clusters=3, linkage=linkage, connectivity=connectivity)
        clustering.fit(X)
        assert_almost_equal(normalized_mutual_info_score(clustering.labels_, true_labels), 1)

def test_connectivity_propagation():
    if False:
        while True:
            i = 10
    X = np.array([(0.014, 0.12), (0.014, 0.099), (0.014, 0.097), (0.017, 0.153), (0.017, 0.153), (0.018, 0.153), (0.018, 0.153), (0.018, 0.153), (0.018, 0.153), (0.018, 0.153), (0.018, 0.153), (0.018, 0.153), (0.018, 0.152), (0.018, 0.149), (0.018, 0.144)])
    connectivity = kneighbors_graph(X, 10, include_self=False)
    ward = AgglomerativeClustering(n_clusters=4, connectivity=connectivity, linkage='ward')
    ward.fit(X)

def test_ward_tree_children_order(global_random_seed):
    if False:
        return 10
    (n, p) = (10, 5)
    rng = np.random.RandomState(global_random_seed)
    connectivity = np.ones((n, n))
    for i in range(5):
        X = 0.1 * rng.normal(size=(n, p))
        X -= 4.0 * np.arange(n)[:, np.newaxis]
        X -= X.mean(axis=1)[:, np.newaxis]
        out_unstructured = ward_tree(X)
        out_structured = ward_tree(X, connectivity=connectivity)
        assert_array_equal(out_unstructured[0], out_structured[0])

def test_ward_linkage_tree_return_distance(global_random_seed):
    if False:
        while True:
            i = 10
    (n, p) = (10, 5)
    rng = np.random.RandomState(global_random_seed)
    connectivity = np.ones((n, n))
    for i in range(5):
        X = 0.1 * rng.normal(size=(n, p))
        X -= 4.0 * np.arange(n)[:, np.newaxis]
        X -= X.mean(axis=1)[:, np.newaxis]
        out_unstructured = ward_tree(X, return_distance=True)
        out_structured = ward_tree(X, connectivity=connectivity, return_distance=True)
        children_unstructured = out_unstructured[0]
        children_structured = out_structured[0]
        assert_array_equal(children_unstructured, children_structured)
        dist_unstructured = out_unstructured[-1]
        dist_structured = out_structured[-1]
        assert_array_almost_equal(dist_unstructured, dist_structured)
        for linkage in ['average', 'complete', 'single']:
            structured_items = linkage_tree(X, connectivity=connectivity, linkage=linkage, return_distance=True)[-1]
            unstructured_items = linkage_tree(X, linkage=linkage, return_distance=True)[-1]
            structured_dist = structured_items[-1]
            unstructured_dist = unstructured_items[-1]
            structured_children = structured_items[0]
            unstructured_children = unstructured_items[0]
            assert_array_almost_equal(structured_dist, unstructured_dist)
            assert_array_almost_equal(structured_children, unstructured_children)
    X = np.array([[1.43054825, -7.5693489], [6.95887839, 6.82293382], [2.87137846, -9.68248579], [7.87974764, -6.05485803], [8.24018364, -6.09495602], [7.39020262, 8.54004355]])
    linkage_X_ward = np.array([[3.0, 4.0, 0.36265956, 2.0], [1.0, 5.0, 1.77045373, 2.0], [0.0, 2.0, 2.55760419, 2.0], [6.0, 8.0, 9.10208346, 4.0], [7.0, 9.0, 24.7784379, 6.0]])
    linkage_X_complete = np.array([[3.0, 4.0, 0.36265956, 2.0], [1.0, 5.0, 1.77045373, 2.0], [0.0, 2.0, 2.55760419, 2.0], [6.0, 8.0, 6.96742194, 4.0], [7.0, 9.0, 18.77445997, 6.0]])
    linkage_X_average = np.array([[3.0, 4.0, 0.36265956, 2.0], [1.0, 5.0, 1.77045373, 2.0], [0.0, 2.0, 2.55760419, 2.0], [6.0, 8.0, 6.55832839, 4.0], [7.0, 9.0, 15.44089605, 6.0]])
    (n_samples, n_features) = np.shape(X)
    connectivity_X = np.ones((n_samples, n_samples))
    out_X_unstructured = ward_tree(X, return_distance=True)
    out_X_structured = ward_tree(X, connectivity=connectivity_X, return_distance=True)
    assert_array_equal(linkage_X_ward[:, :2], out_X_unstructured[0])
    assert_array_equal(linkage_X_ward[:, :2], out_X_structured[0])
    assert_array_almost_equal(linkage_X_ward[:, 2], out_X_unstructured[4])
    assert_array_almost_equal(linkage_X_ward[:, 2], out_X_structured[4])
    linkage_options = ['complete', 'average', 'single']
    X_linkage_truth = [linkage_X_complete, linkage_X_average]
    for (linkage, X_truth) in zip(linkage_options, X_linkage_truth):
        out_X_unstructured = linkage_tree(X, return_distance=True, linkage=linkage)
        out_X_structured = linkage_tree(X, connectivity=connectivity_X, linkage=linkage, return_distance=True)
        assert_array_equal(X_truth[:, :2], out_X_unstructured[0])
        assert_array_equal(X_truth[:, :2], out_X_structured[0])
        assert_array_almost_equal(X_truth[:, 2], out_X_unstructured[4])
        assert_array_almost_equal(X_truth[:, 2], out_X_structured[4])

def test_connectivity_fixing_non_lil():
    if False:
        i = 10
        return i + 15
    x = np.array([[0, 0], [1, 1]])
    m = np.array([[True, False], [False, True]])
    c = grid_to_graph(n_x=2, n_y=2, mask=m)
    w = AgglomerativeClustering(connectivity=c, linkage='ward')
    with pytest.warns(UserWarning):
        w.fit(x)

def test_int_float_dict():
    if False:
        return 10
    rng = np.random.RandomState(0)
    keys = np.unique(rng.randint(100, size=10).astype(np.intp, copy=False))
    values = rng.rand(len(keys))
    d = IntFloatDict(keys, values)
    for (key, value) in zip(keys, values):
        assert d[key] == value
    other_keys = np.arange(50, dtype=np.intp)[::2]
    other_values = np.full(50, 0.5)[::2]
    other = IntFloatDict(other_keys, other_values)
    max_merge(d, other, mask=np.ones(100, dtype=np.intp), n_a=1, n_b=1)
    average_merge(d, other, mask=np.ones(100, dtype=np.intp), n_a=1, n_b=1)

def test_connectivity_callable():
    if False:
        print('Hello World!')
    rng = np.random.RandomState(0)
    X = rng.rand(20, 5)
    connectivity = kneighbors_graph(X, 3, include_self=False)
    aglc1 = AgglomerativeClustering(connectivity=connectivity)
    aglc2 = AgglomerativeClustering(connectivity=partial(kneighbors_graph, n_neighbors=3, include_self=False))
    aglc1.fit(X)
    aglc2.fit(X)
    assert_array_equal(aglc1.labels_, aglc2.labels_)

def test_connectivity_ignores_diagonal():
    if False:
        while True:
            i = 10
    rng = np.random.RandomState(0)
    X = rng.rand(20, 5)
    connectivity = kneighbors_graph(X, 3, include_self=False)
    connectivity_include_self = kneighbors_graph(X, 3, include_self=True)
    aglc1 = AgglomerativeClustering(connectivity=connectivity)
    aglc2 = AgglomerativeClustering(connectivity=connectivity_include_self)
    aglc1.fit(X)
    aglc2.fit(X)
    assert_array_equal(aglc1.labels_, aglc2.labels_)

def test_compute_full_tree():
    if False:
        print('Hello World!')
    rng = np.random.RandomState(0)
    X = rng.randn(10, 2)
    connectivity = kneighbors_graph(X, 5, include_self=False)
    agc = AgglomerativeClustering(n_clusters=2, connectivity=connectivity)
    agc.fit(X)
    n_samples = X.shape[0]
    n_nodes = agc.children_.shape[0]
    assert n_nodes == n_samples - 1
    n_clusters = 101
    X = rng.randn(200, 2)
    connectivity = kneighbors_graph(X, 10, include_self=False)
    agc = AgglomerativeClustering(n_clusters=n_clusters, connectivity=connectivity)
    agc.fit(X)
    n_samples = X.shape[0]
    n_nodes = agc.children_.shape[0]
    assert n_nodes == n_samples - n_clusters

def test_n_components():
    if False:
        i = 10
        return i + 15
    rng = np.random.RandomState(0)
    X = rng.rand(5, 5)
    connectivity = np.eye(5)
    for linkage_func in _TREE_BUILDERS.values():
        assert ignore_warnings(linkage_func)(X, connectivity=connectivity)[1] == 5

def test_affinity_passed_to_fix_connectivity():
    if False:
        print('Hello World!')
    size = 2
    rng = np.random.RandomState(0)
    X = rng.randn(size, size)
    mask = np.array([True, False, False, True])
    connectivity = grid_to_graph(n_x=size, n_y=size, mask=mask, return_as=np.ndarray)

    class FakeAffinity:

        def __init__(self):
            if False:
                print('Hello World!')
            self.counter = 0

        def increment(self, *args, **kwargs):
            if False:
                return 10
            self.counter += 1
            return self.counter
    fa = FakeAffinity()
    linkage_tree(X, connectivity=connectivity, affinity=fa.increment)
    assert fa.counter == 3

@pytest.mark.parametrize('linkage', ['ward', 'complete', 'average'])
def test_agglomerative_clustering_with_distance_threshold(linkage, global_random_seed):
    if False:
        while True:
            i = 10
    rng = np.random.RandomState(global_random_seed)
    mask = np.ones([10, 10], dtype=bool)
    n_samples = 100
    X = rng.randn(n_samples, 50)
    connectivity = grid_to_graph(*mask.shape)
    distance_threshold = 10
    for conn in [None, connectivity]:
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold, connectivity=conn, linkage=linkage)
        clustering.fit(X)
        clusters_produced = clustering.labels_
        num_clusters_produced = len(np.unique(clustering.labels_))
        tree_builder = _TREE_BUILDERS[linkage]
        (children, n_components, n_leaves, parent, distances) = tree_builder(X, connectivity=conn, n_clusters=None, return_distance=True)
        num_clusters_at_threshold = np.count_nonzero(distances >= distance_threshold) + 1
        assert num_clusters_at_threshold == num_clusters_produced
        clusters_at_threshold = _hc_cut(n_clusters=num_clusters_produced, children=children, n_leaves=n_leaves)
        assert np.array_equiv(clusters_produced, clusters_at_threshold)

def test_small_distance_threshold(global_random_seed):
    if False:
        i = 10
        return i + 15
    rng = np.random.RandomState(global_random_seed)
    n_samples = 10
    X = rng.randint(-300, 300, size=(n_samples, 3))
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=1.0, linkage='single').fit(X)
    all_distances = pairwise_distances(X, metric='minkowski', p=2)
    np.fill_diagonal(all_distances, np.inf)
    assert np.all(all_distances > 0.1)
    assert clustering.n_clusters_ == n_samples

def test_cluster_distances_with_distance_threshold(global_random_seed):
    if False:
        for i in range(10):
            print('nop')
    rng = np.random.RandomState(global_random_seed)
    n_samples = 100
    X = rng.randint(-10, 10, size=(n_samples, 3))
    distance_threshold = 4
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold, linkage='single').fit(X)
    labels = clustering.labels_
    D = pairwise_distances(X, metric='minkowski', p=2)
    np.fill_diagonal(D, np.inf)
    for label in np.unique(labels):
        in_cluster_mask = labels == label
        max_in_cluster_distance = D[in_cluster_mask][:, in_cluster_mask].min(axis=0).max()
        min_out_cluster_distance = D[in_cluster_mask][:, ~in_cluster_mask].min(axis=0).min()
        if in_cluster_mask.sum() > 1:
            assert max_in_cluster_distance < distance_threshold
        assert min_out_cluster_distance >= distance_threshold

@pytest.mark.parametrize('linkage', ['ward', 'complete', 'average'])
@pytest.mark.parametrize(('threshold', 'y_true'), [(0.5, [1, 0]), (1.0, [1, 0]), (1.5, [0, 0])])
def test_agglomerative_clustering_with_distance_threshold_edge_case(linkage, threshold, y_true):
    if False:
        print('Hello World!')
    X = [[0], [1]]
    clusterer = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold, linkage=linkage)
    y_pred = clusterer.fit_predict(X)
    assert adjusted_rand_score(y_true, y_pred) == 1

def test_dist_threshold_invalid_parameters():
    if False:
        for i in range(10):
            print('nop')
    X = [[0], [1]]
    with pytest.raises(ValueError, match='Exactly one of '):
        AgglomerativeClustering(n_clusters=None, distance_threshold=None).fit(X)
    with pytest.raises(ValueError, match='Exactly one of '):
        AgglomerativeClustering(n_clusters=2, distance_threshold=1).fit(X)
    X = [[0], [1]]
    with pytest.raises(ValueError, match='compute_full_tree must be True if'):
        AgglomerativeClustering(n_clusters=None, distance_threshold=1, compute_full_tree=False).fit(X)

def test_invalid_shape_precomputed_dist_matrix():
    if False:
        return 10
    rng = np.random.RandomState(0)
    X = rng.rand(5, 3)
    with pytest.raises(ValueError, match='Distance matrix should be square, got matrix of shape \\(5, 3\\)'):
        AgglomerativeClustering(metric='precomputed', linkage='complete').fit(X)

def test_precomputed_connectivity_affinity_with_2_connected_components():
    if False:
        print('Hello World!')
    'Check that connecting components works when connectivity and\n    affinity are both precomputed and the number of connected components is\n    greater than 1. Non-regression test for #16151.\n    '
    connectivity_matrix = np.array([[0, 1, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0]])
    assert connected_components(connectivity_matrix)[0] == 2
    rng = np.random.RandomState(0)
    X = rng.randn(5, 10)
    X_dist = pairwise_distances(X)
    clusterer_precomputed = AgglomerativeClustering(affinity='precomputed', connectivity=connectivity_matrix, linkage='complete')
    msg = 'Completing it to avoid stopping the tree early'
    with pytest.warns(UserWarning, match=msg):
        clusterer_precomputed.fit(X_dist)
    clusterer = AgglomerativeClustering(connectivity=connectivity_matrix, linkage='complete')
    with pytest.warns(UserWarning, match=msg):
        clusterer.fit(X)
    assert_array_equal(clusterer.labels_, clusterer_precomputed.labels_)
    assert_array_equal(clusterer.children_, clusterer_precomputed.children_)

def test_deprecate_affinity():
    if False:
        print('Hello World!')
    rng = np.random.RandomState(42)
    X = rng.randn(50, 10)
    af = AgglomerativeClustering(affinity='euclidean')
    msg = 'Attribute `affinity` was deprecated in version 1.2 and will be removed in 1.4. Use `metric` instead'
    with pytest.warns(FutureWarning, match=msg):
        af.fit(X)
    with pytest.warns(FutureWarning, match=msg):
        af.fit_predict(X)
    af = AgglomerativeClustering(metric='euclidean', affinity='euclidean')
    msg = 'Both `affinity` and `metric` attributes were set. Attribute'
    with pytest.raises(ValueError, match=msg):
        af.fit(X)
    with pytest.raises(ValueError, match=msg):
        af.fit_predict(X)