import re
import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.ensemble._hist_gradient_boosting.common import G_H_DTYPE, X_BINNED_DTYPE, MonotonicConstraint
from sklearn.ensemble._hist_gradient_boosting.grower import TreeGrower
from sklearn.ensemble._hist_gradient_boosting.histogram import HistogramBuilder
from sklearn.ensemble._hist_gradient_boosting.splitting import Splitter, compute_node_value
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from sklearn.utils._testing import _convert_container
n_threads = _openmp_effective_n_threads()

def is_increasing(a):
    if False:
        for i in range(10):
            print('nop')
    return (np.diff(a) >= 0.0).all()

def is_decreasing(a):
    if False:
        for i in range(10):
            print('nop')
    return (np.diff(a) <= 0.0).all()

def assert_leaves_values_monotonic(predictor, monotonic_cst):
    if False:
        return 10
    nodes = predictor.nodes

    def get_leaves_values():
        if False:
            i = 10
            return i + 15
        'get leaves values from left to right'
        values = []

        def depth_first_collect_leaf_values(node_idx):
            if False:
                for i in range(10):
                    print('nop')
            node = nodes[node_idx]
            if node['is_leaf']:
                values.append(node['value'])
                return
            depth_first_collect_leaf_values(node['left'])
            depth_first_collect_leaf_values(node['right'])
        depth_first_collect_leaf_values(0)
        return values
    values = get_leaves_values()
    if monotonic_cst == MonotonicConstraint.NO_CST:
        assert not is_increasing(values) and (not is_decreasing(values))
    elif monotonic_cst == MonotonicConstraint.POS:
        assert is_increasing(values)
    else:
        assert is_decreasing(values)

def assert_children_values_monotonic(predictor, monotonic_cst):
    if False:
        for i in range(10):
            print('nop')
    nodes = predictor.nodes
    left_lower = []
    left_greater = []
    for node in nodes:
        if node['is_leaf']:
            continue
        left_idx = node['left']
        right_idx = node['right']
        if nodes[left_idx]['value'] < nodes[right_idx]['value']:
            left_lower.append(node)
        elif nodes[left_idx]['value'] > nodes[right_idx]['value']:
            left_greater.append(node)
    if monotonic_cst == MonotonicConstraint.NO_CST:
        assert left_lower and left_greater
    elif monotonic_cst == MonotonicConstraint.POS:
        assert left_lower and (not left_greater)
    else:
        assert not left_lower and left_greater

def assert_children_values_bounded(grower, monotonic_cst):
    if False:
        return 10
    if monotonic_cst == MonotonicConstraint.NO_CST:
        return

    def recursively_check_children_node_values(node, right_sibling=None):
        if False:
            return 10
        if node.is_leaf:
            return
        if right_sibling is not None:
            middle = (node.value + right_sibling.value) / 2
            if monotonic_cst == MonotonicConstraint.POS:
                assert node.left_child.value <= node.right_child.value <= middle
                if not right_sibling.is_leaf:
                    assert middle <= right_sibling.left_child.value <= right_sibling.right_child.value
            else:
                assert node.left_child.value >= node.right_child.value >= middle
                if not right_sibling.is_leaf:
                    assert middle >= right_sibling.left_child.value >= right_sibling.right_child.value
        recursively_check_children_node_values(node.left_child, right_sibling=node.right_child)
        recursively_check_children_node_values(node.right_child)
    recursively_check_children_node_values(grower.root)

@pytest.mark.parametrize('seed', range(3))
@pytest.mark.parametrize('monotonic_cst', (MonotonicConstraint.NO_CST, MonotonicConstraint.POS, MonotonicConstraint.NEG))
def test_nodes_values(monotonic_cst, seed):
    if False:
        print('Hello World!')
    rng = np.random.RandomState(seed)
    n_samples = 1000
    n_features = 1
    X_binned = rng.randint(0, 255, size=(n_samples, n_features), dtype=np.uint8)
    X_binned = np.asfortranarray(X_binned)
    gradients = rng.normal(size=n_samples).astype(G_H_DTYPE)
    hessians = np.ones(shape=1, dtype=G_H_DTYPE)
    grower = TreeGrower(X_binned, gradients, hessians, monotonic_cst=[monotonic_cst], shrinkage=0.1)
    grower.grow()
    for leave in grower.finalized_leaves:
        leave.value /= grower.shrinkage
    predictor = grower.make_predictor(binning_thresholds=np.zeros((X_binned.shape[1], X_binned.max() + 1)))
    assert_children_values_monotonic(predictor, monotonic_cst)
    assert_children_values_bounded(grower, monotonic_cst)
    assert_leaves_values_monotonic(predictor, monotonic_cst)

@pytest.mark.parametrize('use_feature_names', (True, False))
def test_predictions(global_random_seed, use_feature_names):
    if False:
        i = 10
        return i + 15
    rng = np.random.RandomState(global_random_seed)
    n_samples = 1000
    f_0 = rng.rand(n_samples)
    f_1 = rng.rand(n_samples)
    X = np.c_[f_0, f_1]
    columns_name = ['f_0', 'f_1']
    constructor_name = 'dataframe' if use_feature_names else 'array'
    X = _convert_container(X, constructor_name, columns_name=columns_name)
    noise = rng.normal(loc=0.0, scale=0.01, size=n_samples)
    y = 5 * f_0 + np.sin(10 * np.pi * f_0) - 5 * f_1 - np.cos(10 * np.pi * f_1) + noise
    if use_feature_names:
        monotonic_cst = {'f_0': +1, 'f_1': -1}
    else:
        monotonic_cst = [+1, -1]
    gbdt = HistGradientBoostingRegressor(monotonic_cst=monotonic_cst)
    gbdt.fit(X, y)
    linspace = np.linspace(0, 1, 100)
    sin = np.sin(linspace)
    constant = np.full_like(linspace, fill_value=0.5)
    X = np.c_[linspace, constant]
    X = _convert_container(X, constructor_name, columns_name=columns_name)
    pred = gbdt.predict(X)
    assert is_increasing(pred)
    X = np.c_[sin, constant]
    X = _convert_container(X, constructor_name, columns_name=columns_name)
    pred = gbdt.predict(X)
    assert np.all((np.diff(pred) >= 0) == (np.diff(sin) >= 0))
    X = np.c_[constant, linspace]
    X = _convert_container(X, constructor_name, columns_name=columns_name)
    pred = gbdt.predict(X)
    assert is_decreasing(pred)
    X = np.c_[constant, sin]
    X = _convert_container(X, constructor_name, columns_name=columns_name)
    pred = gbdt.predict(X)
    assert ((np.diff(pred) <= 0) == (np.diff(sin) >= 0)).all()

def test_input_error():
    if False:
        for i in range(10):
            print('nop')
    X = [[1, 2], [2, 3], [3, 4]]
    y = [0, 1, 2]
    gbdt = HistGradientBoostingRegressor(monotonic_cst=[1, 0, -1])
    with pytest.raises(ValueError, match=re.escape('monotonic_cst has shape (3,) but the input data')):
        gbdt.fit(X, y)
    for monotonic_cst in ([1, 3], [1, -3], [0.3, -0.7]):
        gbdt = HistGradientBoostingRegressor(monotonic_cst=monotonic_cst)
        expected_msg = re.escape('must be an array-like of -1, 0 or 1. Observed values:')
        with pytest.raises(ValueError, match=expected_msg):
            gbdt.fit(X, y)
    gbdt = HistGradientBoostingClassifier(monotonic_cst=[0, 1])
    with pytest.raises(ValueError, match='monotonic constraints are not supported for multiclass classification'):
        gbdt.fit(X, y)

def test_input_error_related_to_feature_names():
    if False:
        while True:
            i = 10
    pd = pytest.importorskip('pandas')
    X = pd.DataFrame({'a': [0, 1, 2], 'b': [0, 1, 2]})
    y = np.array([0, 1, 0])
    monotonic_cst = {'d': 1, 'a': 1, 'c': -1}
    gbdt = HistGradientBoostingRegressor(monotonic_cst=monotonic_cst)
    expected_msg = re.escape("monotonic_cst contains 2 unexpected feature names: ['c', 'd'].")
    with pytest.raises(ValueError, match=expected_msg):
        gbdt.fit(X, y)
    monotonic_cst = {k: 1 for k in 'abcdefghijklmnopqrstuvwxyz'}
    gbdt = HistGradientBoostingRegressor(monotonic_cst=monotonic_cst)
    expected_msg = re.escape("monotonic_cst contains 24 unexpected feature names: ['c', 'd', 'e', 'f', 'g', '...'].")
    with pytest.raises(ValueError, match=expected_msg):
        gbdt.fit(X, y)
    monotonic_cst = {'a': 1}
    gbdt = HistGradientBoostingRegressor(monotonic_cst=monotonic_cst)
    expected_msg = re.escape('HistGradientBoostingRegressor was not fitted on data with feature names. Pass monotonic_cst as an integer array instead.')
    with pytest.raises(ValueError, match=expected_msg):
        gbdt.fit(X.values, y)
    monotonic_cst = {'b': -1, 'a': '+'}
    gbdt = HistGradientBoostingRegressor(monotonic_cst=monotonic_cst)
    expected_msg = re.escape("monotonic_cst['a'] must be either -1, 0 or 1. Got '+'.")
    with pytest.raises(ValueError, match=expected_msg):
        gbdt.fit(X, y)

def test_bounded_value_min_gain_to_split():
    if False:
        i = 10
        return i + 15
    l2_regularization = 0
    min_hessian_to_split = 0
    min_samples_leaf = 1
    n_bins = n_samples = 5
    X_binned = np.arange(n_samples).reshape(-1, 1).astype(X_BINNED_DTYPE)
    sample_indices = np.arange(n_samples, dtype=np.uint32)
    all_hessians = np.ones(n_samples, dtype=G_H_DTYPE)
    all_gradients = np.array([1, 1, 100, 1, 1], dtype=G_H_DTYPE)
    sum_gradients = all_gradients.sum()
    sum_hessians = all_hessians.sum()
    hessians_are_constant = False
    builder = HistogramBuilder(X_binned, n_bins, all_gradients, all_hessians, hessians_are_constant, n_threads)
    n_bins_non_missing = np.array([n_bins - 1] * X_binned.shape[1], dtype=np.uint32)
    has_missing_values = np.array([False] * X_binned.shape[1], dtype=np.uint8)
    monotonic_cst = np.array([MonotonicConstraint.NO_CST] * X_binned.shape[1], dtype=np.int8)
    is_categorical = np.zeros_like(monotonic_cst, dtype=np.uint8)
    missing_values_bin_idx = n_bins - 1
    (children_lower_bound, children_upper_bound) = (-np.inf, np.inf)
    min_gain_to_split = 2000
    splitter = Splitter(X_binned, n_bins_non_missing, missing_values_bin_idx, has_missing_values, is_categorical, monotonic_cst, l2_regularization, min_hessian_to_split, min_samples_leaf, min_gain_to_split, hessians_are_constant)
    histograms = builder.compute_histograms_brute(sample_indices)
    (current_lower_bound, current_upper_bound) = (-np.inf, np.inf)
    value = compute_node_value(sum_gradients, sum_hessians, current_lower_bound, current_upper_bound, l2_regularization)
    assert value == pytest.approx(-104 / 5)
    split_info = splitter.find_node_split(n_samples, histograms, sum_gradients, sum_hessians, value, lower_bound=children_lower_bound, upper_bound=children_upper_bound)
    assert split_info.gain == -1
    (current_lower_bound, current_upper_bound) = (-10, np.inf)
    value = compute_node_value(sum_gradients, sum_hessians, current_lower_bound, current_upper_bound, l2_regularization)
    assert value == -10
    split_info = splitter.find_node_split(n_samples, histograms, sum_gradients, sum_hessians, value, lower_bound=children_lower_bound, upper_bound=children_upper_bound)
    assert split_info.gain > min_gain_to_split