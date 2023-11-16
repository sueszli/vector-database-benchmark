import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pytest import approx
from sklearn.ensemble._hist_gradient_boosting.binning import _BinMapper
from sklearn.ensemble._hist_gradient_boosting.common import G_H_DTYPE, X_BINNED_DTYPE, X_BITSET_INNER_DTYPE, X_DTYPE, Y_DTYPE
from sklearn.ensemble._hist_gradient_boosting.grower import TreeGrower
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
n_threads = _openmp_effective_n_threads()

def _make_training_data(n_bins=256, constant_hessian=True):
    if False:
        while True:
            i = 10
    rng = np.random.RandomState(42)
    n_samples = 10000
    X_binned = rng.randint(0, n_bins - 1, size=(n_samples, 2), dtype=X_BINNED_DTYPE)
    X_binned = np.asfortranarray(X_binned)

    def true_decision_function(input_features):
        if False:
            i = 10
            return i + 15
        'Ground truth decision function\n\n        This is a very simple yet asymmetric decision tree. Therefore the\n        grower code should have no trouble recovering the decision function\n        from 10000 training samples.\n        '
        if input_features[0] <= n_bins // 2:
            return -1
        else:
            return -1 if input_features[1] <= n_bins // 3 else 1
    target = np.array([true_decision_function(x) for x in X_binned], dtype=Y_DTYPE)
    all_gradients = target.astype(G_H_DTYPE)
    shape_hessians = 1 if constant_hessian else all_gradients.shape
    all_hessians = np.ones(shape=shape_hessians, dtype=G_H_DTYPE)
    return (X_binned, all_gradients, all_hessians)

def _check_children_consistency(parent, left, right):
    if False:
        while True:
            i = 10
    assert parent.left_child is left
    assert parent.right_child is right
    assert len(left.sample_indices) + len(right.sample_indices) == len(parent.sample_indices)
    assert set(left.sample_indices).union(set(right.sample_indices)) == set(parent.sample_indices)
    assert set(left.sample_indices).intersection(set(right.sample_indices)) == set()

@pytest.mark.parametrize('n_bins, constant_hessian, stopping_param, shrinkage', [(11, True, 'min_gain_to_split', 0.5), (11, False, 'min_gain_to_split', 1.0), (11, True, 'max_leaf_nodes', 1.0), (11, False, 'max_leaf_nodes', 0.1), (42, True, 'max_leaf_nodes', 0.01), (42, False, 'max_leaf_nodes', 1.0), (256, True, 'min_gain_to_split', 1.0), (256, True, 'max_leaf_nodes', 0.1)])
def test_grow_tree(n_bins, constant_hessian, stopping_param, shrinkage):
    if False:
        return 10
    (X_binned, all_gradients, all_hessians) = _make_training_data(n_bins=n_bins, constant_hessian=constant_hessian)
    n_samples = X_binned.shape[0]
    if stopping_param == 'max_leaf_nodes':
        stopping_param = {'max_leaf_nodes': 3}
    else:
        stopping_param = {'min_gain_to_split': 0.01}
    grower = TreeGrower(X_binned, all_gradients, all_hessians, n_bins=n_bins, shrinkage=shrinkage, min_samples_leaf=1, **stopping_param)
    assert grower.root.left_child is None
    assert grower.root.right_child is None
    root_split = grower.root.split_info
    assert root_split.feature_idx == 0
    assert root_split.bin_idx == n_bins // 2
    assert len(grower.splittable_nodes) == 1
    (left_node, right_node) = grower.split_next()
    _check_children_consistency(grower.root, left_node, right_node)
    assert len(left_node.sample_indices) > 0.4 * n_samples
    assert len(left_node.sample_indices) < 0.6 * n_samples
    if grower.min_gain_to_split > 0:
        assert left_node.split_info.gain < grower.min_gain_to_split
        assert left_node in grower.finalized_leaves
    split_info = right_node.split_info
    assert split_info.gain > 1.0
    assert split_info.feature_idx == 1
    assert split_info.bin_idx == n_bins // 3
    assert right_node.left_child is None
    assert right_node.right_child is None
    assert len(grower.splittable_nodes) == 1
    (right_left_node, right_right_node) = grower.split_next()
    _check_children_consistency(right_node, right_left_node, right_right_node)
    assert len(right_left_node.sample_indices) > 0.1 * n_samples
    assert len(right_left_node.sample_indices) < 0.2 * n_samples
    assert len(right_right_node.sample_indices) > 0.2 * n_samples
    assert len(right_right_node.sample_indices) < 0.4 * n_samples
    assert not grower.splittable_nodes
    grower._apply_shrinkage()
    assert grower.root.left_child.value == approx(shrinkage)
    assert grower.root.right_child.left_child.value == approx(shrinkage)
    assert grower.root.right_child.right_child.value == approx(-shrinkage, rel=0.001)

def test_predictor_from_grower():
    if False:
        return 10
    n_bins = 256
    (X_binned, all_gradients, all_hessians) = _make_training_data(n_bins=n_bins)
    grower = TreeGrower(X_binned, all_gradients, all_hessians, n_bins=n_bins, shrinkage=1.0, max_leaf_nodes=3, min_samples_leaf=5)
    grower.grow()
    assert grower.n_nodes == 5
    predictor = grower.make_predictor(binning_thresholds=np.zeros((X_binned.shape[1], n_bins)))
    assert predictor.nodes.shape[0] == 5
    assert predictor.nodes['is_leaf'].sum() == 3
    input_data = np.array([[0, 0], [42, 99], [128, 254], [129, 0], [129, 85], [254, 85], [129, 86], [129, 254], [242, 100]], dtype=np.uint8)
    missing_values_bin_idx = n_bins - 1
    predictions = predictor.predict_binned(input_data, missing_values_bin_idx, n_threads)
    expected_targets = [1, 1, 1, 1, 1, 1, -1, -1, -1]
    assert np.allclose(predictions, expected_targets)
    predictions = predictor.predict_binned(X_binned, missing_values_bin_idx, n_threads)
    assert np.allclose(predictions, -all_gradients)

@pytest.mark.parametrize('n_samples, min_samples_leaf, n_bins, constant_hessian, noise', [(11, 10, 7, True, 0), (13, 10, 42, False, 0), (56, 10, 255, True, 0.1), (101, 3, 7, True, 0), (200, 42, 42, False, 0), (300, 55, 255, True, 0.1), (300, 301, 255, True, 0.1)])
def test_min_samples_leaf(n_samples, min_samples_leaf, n_bins, constant_hessian, noise):
    if False:
        i = 10
        return i + 15
    rng = np.random.RandomState(seed=0)
    X = rng.normal(size=(n_samples, 3))
    y = X[:, 0] - X[:, 1]
    if noise:
        y_scale = y.std()
        y += rng.normal(scale=noise, size=n_samples) * y_scale
    mapper = _BinMapper(n_bins=n_bins)
    X = mapper.fit_transform(X)
    all_gradients = y.astype(G_H_DTYPE)
    shape_hessian = 1 if constant_hessian else all_gradients.shape
    all_hessians = np.ones(shape=shape_hessian, dtype=G_H_DTYPE)
    grower = TreeGrower(X, all_gradients, all_hessians, n_bins=n_bins, shrinkage=1.0, min_samples_leaf=min_samples_leaf, max_leaf_nodes=n_samples)
    grower.grow()
    predictor = grower.make_predictor(binning_thresholds=mapper.bin_thresholds_)
    if n_samples >= min_samples_leaf:
        for node in predictor.nodes:
            if node['is_leaf']:
                assert node['count'] >= min_samples_leaf
    else:
        assert predictor.nodes.shape[0] == 1
        assert predictor.nodes[0]['is_leaf']
        assert predictor.nodes[0]['count'] == n_samples

@pytest.mark.parametrize('n_samples, min_samples_leaf', [(99, 50), (100, 50)])
def test_min_samples_leaf_root(n_samples, min_samples_leaf):
    if False:
        while True:
            i = 10
    rng = np.random.RandomState(seed=0)
    n_bins = 256
    X = rng.normal(size=(n_samples, 3))
    y = X[:, 0] - X[:, 1]
    mapper = _BinMapper(n_bins=n_bins)
    X = mapper.fit_transform(X)
    all_gradients = y.astype(G_H_DTYPE)
    all_hessians = np.ones(shape=1, dtype=G_H_DTYPE)
    grower = TreeGrower(X, all_gradients, all_hessians, n_bins=n_bins, shrinkage=1.0, min_samples_leaf=min_samples_leaf, max_leaf_nodes=n_samples)
    grower.grow()
    if n_samples >= min_samples_leaf * 2:
        assert len(grower.finalized_leaves) >= 2
    else:
        assert len(grower.finalized_leaves) == 1

def assert_is_stump(grower):
    if False:
        while True:
            i = 10
    for leaf in (grower.root.left_child, grower.root.right_child):
        assert leaf.left_child is None
        assert leaf.right_child is None

@pytest.mark.parametrize('max_depth', [1, 2, 3])
def test_max_depth(max_depth):
    if False:
        print('Hello World!')
    rng = np.random.RandomState(seed=0)
    n_bins = 256
    n_samples = 1000
    X = rng.normal(size=(n_samples, 3))
    y = X[:, 0] - X[:, 1]
    mapper = _BinMapper(n_bins=n_bins)
    X = mapper.fit_transform(X)
    all_gradients = y.astype(G_H_DTYPE)
    all_hessians = np.ones(shape=1, dtype=G_H_DTYPE)
    grower = TreeGrower(X, all_gradients, all_hessians, max_depth=max_depth)
    grower.grow()
    depth = max((leaf.depth for leaf in grower.finalized_leaves))
    assert depth == max_depth
    if max_depth == 1:
        assert_is_stump(grower)

def test_input_validation():
    if False:
        for i in range(10):
            print('nop')
    (X_binned, all_gradients, all_hessians) = _make_training_data()
    X_binned_float = X_binned.astype(np.float32)
    with pytest.raises(NotImplementedError, match='X_binned must be of type uint8'):
        TreeGrower(X_binned_float, all_gradients, all_hessians)
    X_binned_C_array = np.ascontiguousarray(X_binned)
    with pytest.raises(ValueError, match='X_binned should be passed as Fortran contiguous array'):
        TreeGrower(X_binned_C_array, all_gradients, all_hessians)

def test_init_parameters_validation():
    if False:
        return 10
    (X_binned, all_gradients, all_hessians) = _make_training_data()
    with pytest.raises(ValueError, match='min_gain_to_split=-1 must be positive'):
        TreeGrower(X_binned, all_gradients, all_hessians, min_gain_to_split=-1)
    with pytest.raises(ValueError, match='min_hessian_to_split=-1 must be positive'):
        TreeGrower(X_binned, all_gradients, all_hessians, min_hessian_to_split=-1)

def test_missing_value_predict_only():
    if False:
        while True:
            i = 10
    rng = np.random.RandomState(0)
    n_samples = 100
    X_binned = rng.randint(0, 256, size=(n_samples, 1), dtype=np.uint8)
    X_binned = np.asfortranarray(X_binned)
    gradients = rng.normal(size=n_samples).astype(G_H_DTYPE)
    hessians = np.ones(shape=1, dtype=G_H_DTYPE)
    grower = TreeGrower(X_binned, gradients, hessians, min_samples_leaf=5, has_missing_values=False)
    grower.grow()
    predictor = grower.make_predictor(binning_thresholds=np.zeros((X_binned.shape[1], X_binned.max() + 1)))
    node = predictor.nodes[0]
    while not node['is_leaf']:
        left = predictor.nodes[node['left']]
        right = predictor.nodes[node['right']]
        node = left if left['count'] > right['count'] else right
    prediction_main_path = node['value']
    all_nans = np.full(shape=(n_samples, 1), fill_value=np.nan)
    known_cat_bitsets = np.zeros((0, 8), dtype=X_BITSET_INNER_DTYPE)
    f_idx_map = np.zeros(0, dtype=np.uint32)
    y_pred = predictor.predict(all_nans, known_cat_bitsets, f_idx_map, n_threads)
    assert np.all(y_pred == prediction_main_path)

def test_split_on_nan_with_infinite_values():
    if False:
        print('Hello World!')
    X = np.array([0, 1, np.inf, np.nan, np.nan]).reshape(-1, 1)
    gradients = np.array([0, 0, 0, 100, 100], dtype=G_H_DTYPE)
    hessians = np.ones(shape=1, dtype=G_H_DTYPE)
    bin_mapper = _BinMapper()
    X_binned = bin_mapper.fit_transform(X)
    n_bins_non_missing = 3
    has_missing_values = True
    grower = TreeGrower(X_binned, gradients, hessians, n_bins_non_missing=n_bins_non_missing, has_missing_values=has_missing_values, min_samples_leaf=1, n_threads=n_threads)
    grower.grow()
    predictor = grower.make_predictor(binning_thresholds=bin_mapper.bin_thresholds_)
    assert predictor.nodes[0]['num_threshold'] == np.inf
    assert predictor.nodes[0]['bin_threshold'] == n_bins_non_missing - 1
    (known_cat_bitsets, f_idx_map) = bin_mapper.make_known_categories_bitsets()
    predictions = predictor.predict(X, known_cat_bitsets, f_idx_map, n_threads)
    predictions_binned = predictor.predict_binned(X_binned, missing_values_bin_idx=bin_mapper.missing_values_bin_idx_, n_threads=n_threads)
    np.testing.assert_allclose(predictions, -gradients)
    np.testing.assert_allclose(predictions_binned, -gradients)

def test_grow_tree_categories():
    if False:
        for i in range(10):
            print('nop')
    X_binned = np.array([[0, 1] * 11 + [1]], dtype=X_BINNED_DTYPE).T
    X_binned = np.asfortranarray(X_binned)
    all_gradients = np.array([10, 1] * 11 + [1], dtype=G_H_DTYPE)
    all_hessians = np.ones(1, dtype=G_H_DTYPE)
    is_categorical = np.ones(1, dtype=np.uint8)
    grower = TreeGrower(X_binned, all_gradients, all_hessians, n_bins=4, shrinkage=1.0, min_samples_leaf=1, is_categorical=is_categorical, n_threads=n_threads)
    grower.grow()
    assert grower.n_nodes == 3
    categories = [np.array([4, 9], dtype=X_DTYPE)]
    predictor = grower.make_predictor(binning_thresholds=categories)
    root = predictor.nodes[0]
    assert root['count'] == 23
    assert root['depth'] == 0
    assert root['is_categorical']
    (left, right) = (predictor.nodes[root['left']], predictor.nodes[root['right']])
    assert left['count'] >= right['count']
    expected_binned_cat_bitset = [2 ** 1] + [0] * 7
    binned_cat_bitset = predictor.binned_left_cat_bitsets
    assert_array_equal(binned_cat_bitset[0], expected_binned_cat_bitset)
    expected_raw_cat_bitsets = [2 ** 9] + [0] * 7
    raw_cat_bitsets = predictor.raw_left_cat_bitsets
    assert_array_equal(raw_cat_bitsets[0], expected_raw_cat_bitsets)
    assert root['missing_go_to_left']
    prediction_binned = predictor.predict_binned(np.asarray([[6]]).astype(X_BINNED_DTYPE), missing_values_bin_idx=6, n_threads=n_threads)
    assert_allclose(prediction_binned, [-1])
    known_cat_bitsets = np.zeros((1, 8), dtype=np.uint32)
    f_idx_map = np.array([0], dtype=np.uint32)
    prediction = predictor.predict(np.array([[np.nan]]), known_cat_bitsets, f_idx_map, n_threads)
    assert_allclose(prediction, [-1])

@pytest.mark.parametrize('min_samples_leaf', (1, 20))
@pytest.mark.parametrize('n_unique_categories', (2, 10, 100))
@pytest.mark.parametrize('target', ('binary', 'random', 'equal'))
def test_ohe_equivalence(min_samples_leaf, n_unique_categories, target):
    if False:
        return 10
    rng = np.random.RandomState(0)
    n_samples = 10000
    X_binned = rng.randint(0, n_unique_categories, size=(n_samples, 1), dtype=np.uint8)
    X_ohe = OneHotEncoder(sparse_output=False).fit_transform(X_binned)
    X_ohe = np.asfortranarray(X_ohe).astype(np.uint8)
    if target == 'equal':
        gradients = X_binned.reshape(-1)
    elif target == 'binary':
        gradients = (X_binned % 2).reshape(-1)
    else:
        gradients = rng.randn(n_samples)
    gradients = gradients.astype(G_H_DTYPE)
    hessians = np.ones(shape=1, dtype=G_H_DTYPE)
    grower_params = {'min_samples_leaf': min_samples_leaf, 'max_depth': None, 'max_leaf_nodes': None}
    grower = TreeGrower(X_binned, gradients, hessians, is_categorical=[True], **grower_params)
    grower.grow()
    predictor = grower.make_predictor(binning_thresholds=np.zeros((1, n_unique_categories)))
    preds = predictor.predict_binned(X_binned, missing_values_bin_idx=255, n_threads=n_threads)
    grower_ohe = TreeGrower(X_ohe, gradients, hessians, **grower_params)
    grower_ohe.grow()
    predictor_ohe = grower_ohe.make_predictor(binning_thresholds=np.zeros((X_ohe.shape[1], n_unique_categories)))
    preds_ohe = predictor_ohe.predict_binned(X_ohe, missing_values_bin_idx=255, n_threads=n_threads)
    assert predictor.get_max_depth() <= predictor_ohe.get_max_depth()
    if target == 'binary' and n_unique_categories > 2:
        assert predictor.get_max_depth() < predictor_ohe.get_max_depth()
    np.testing.assert_allclose(preds, preds_ohe)

def test_grower_interaction_constraints():
    if False:
        return 10
    'Check that grower respects interaction constraints.'
    n_features = 6
    interaction_cst = [{0, 1}, {1, 2}, {3, 4, 5}]
    n_samples = 10
    n_bins = 6
    root_feature_splits = []

    def get_all_children(node):
        if False:
            return 10
        res = []
        if node.is_leaf:
            return res
        for n in [node.left_child, node.right_child]:
            res.append(n)
            res.extend(get_all_children(n))
        return res
    for seed in range(20):
        rng = np.random.RandomState(seed)
        X_binned = rng.randint(0, n_bins - 1, size=(n_samples, n_features), dtype=X_BINNED_DTYPE)
        X_binned = np.asfortranarray(X_binned)
        gradients = rng.normal(size=n_samples).astype(G_H_DTYPE)
        hessians = np.ones(shape=1, dtype=G_H_DTYPE)
        grower = TreeGrower(X_binned, gradients, hessians, n_bins=n_bins, min_samples_leaf=1, interaction_cst=interaction_cst, n_threads=n_threads)
        grower.grow()
        root_feature_idx = grower.root.split_info.feature_idx
        root_feature_splits.append(root_feature_idx)
        feature_idx_to_constraint_set = {0: {0, 1}, 1: {0, 1, 2}, 2: {1, 2}, 3: {3, 4, 5}, 4: {3, 4, 5}, 5: {3, 4, 5}}
        root_constraint_set = feature_idx_to_constraint_set[root_feature_idx]
        for node in (grower.root.left_child, grower.root.right_child):
            assert_array_equal(node.allowed_features, list(root_constraint_set))
        for node in get_all_children(grower.root):
            if node.is_leaf:
                continue
            parent_interaction_cst_indices = set(node.interaction_cst_indices)
            right_interactions_cst_indices = set(node.right_child.interaction_cst_indices)
            left_interactions_cst_indices = set(node.left_child.interaction_cst_indices)
            assert right_interactions_cst_indices.issubset(parent_interaction_cst_indices)
            assert left_interactions_cst_indices.issubset(parent_interaction_cst_indices)
            assert node.split_info.feature_idx in root_constraint_set
    assert len(set(root_feature_splits)) == len(set().union(*interaction_cst)) == n_features