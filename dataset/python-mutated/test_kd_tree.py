import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal
from sklearn.neighbors._kd_tree import KDTree, KDTree32, KDTree64
from sklearn.neighbors.tests.test_ball_tree import get_dataset_for_binary_tree
from sklearn.utils.parallel import Parallel, delayed
DIMENSION = 3
METRICS = {'euclidean': {}, 'manhattan': {}, 'chebyshev': {}, 'minkowski': dict(p=3)}
KD_TREE_CLASSES = [KDTree64, KDTree32]

def test_KDTree_is_KDTree64_subclass():
    if False:
        print('Hello World!')
    assert issubclass(KDTree, KDTree64)

@pytest.mark.parametrize('BinarySearchTree', KD_TREE_CLASSES)
def test_array_object_type(BinarySearchTree):
    if False:
        while True:
            i = 10
    'Check that we do not accept object dtype array.'
    X = np.array([(1, 2, 3), (2, 5), (5, 5, 1, 2)], dtype=object)
    with pytest.raises(ValueError, match='setting an array element with a sequence'):
        BinarySearchTree(X)

@pytest.mark.parametrize('BinarySearchTree', KD_TREE_CLASSES)
def test_kdtree_picklable_with_joblib(BinarySearchTree):
    if False:
        i = 10
        return i + 15
    'Make sure that KDTree queries work when joblib memmaps.\n\n    Non-regression test for #21685 and #21228.'
    rng = np.random.RandomState(0)
    X = rng.random_sample((10, 3))
    tree = BinarySearchTree(X, leaf_size=2)
    Parallel(n_jobs=2, max_nbytes=1)((delayed(tree.query)(data) for data in 2 * [X]))

@pytest.mark.parametrize('metric', METRICS)
def test_kd_tree_numerical_consistency(global_random_seed, metric):
    if False:
        return 10
    (X_64, X_32, Y_64, Y_32) = get_dataset_for_binary_tree(random_seed=global_random_seed, features=50)
    metric_params = METRICS.get(metric, {})
    kd_64 = KDTree64(X_64, leaf_size=2, metric=metric, **metric_params)
    kd_32 = KDTree32(X_32, leaf_size=2, metric=metric, **metric_params)
    k = 4
    (dist_64, ind_64) = kd_64.query(Y_64, k=k)
    (dist_32, ind_32) = kd_32.query(Y_32, k=k)
    assert_allclose(dist_64, dist_32, rtol=1e-05)
    assert_equal(ind_64, ind_32)
    assert dist_64.dtype == np.float64
    assert dist_32.dtype == np.float32
    r = 2.38
    ind_64 = kd_64.query_radius(Y_64, r=r)
    ind_32 = kd_32.query_radius(Y_32, r=r)
    for (_ind64, _ind32) in zip(ind_64, ind_32):
        assert_equal(_ind64, _ind32)
    (ind_64, dist_64) = kd_64.query_radius(Y_64, r=r, return_distance=True)
    (ind_32, dist_32) = kd_32.query_radius(Y_32, r=r, return_distance=True)
    for (_ind64, _ind32, _dist_64, _dist_32) in zip(ind_64, ind_32, dist_64, dist_32):
        assert_equal(_ind64, _ind32)
        assert_allclose(_dist_64, _dist_32, rtol=1e-05)
        assert _dist_64.dtype == np.float64
        assert _dist_32.dtype == np.float32

@pytest.mark.parametrize('metric', METRICS)
def test_kernel_density_numerical_consistency(global_random_seed, metric):
    if False:
        while True:
            i = 10
    (X_64, X_32, Y_64, Y_32) = get_dataset_for_binary_tree(random_seed=global_random_seed)
    metric_params = METRICS.get(metric, {})
    kd_64 = KDTree64(X_64, leaf_size=2, metric=metric, **metric_params)
    kd_32 = KDTree32(X_32, leaf_size=2, metric=metric, **metric_params)
    kernel = 'gaussian'
    h = 0.1
    density64 = kd_64.kernel_density(Y_64, h=h, kernel=kernel, breadth_first=True)
    density32 = kd_32.kernel_density(Y_32, h=h, kernel=kernel, breadth_first=True)
    assert_allclose(density64, density32, rtol=1e-05)
    assert density64.dtype == np.float64
    assert density32.dtype == np.float32