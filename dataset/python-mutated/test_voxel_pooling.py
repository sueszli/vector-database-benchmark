import open3d as o3d
import numpy as np
import pytest
import mltest
from check_gradients import check_gradients
pytestmark = mltest.default_marks
position_dtypes = pytest.mark.parametrize('pos_dtype', [np.float32, np.float64])
feature_dtypes = pytest.mark.parametrize('feat_dtype', [np.float32, np.float64, np.int32, np.int64])
position_functions = pytest.mark.parametrize('position_fn', ['average', 'center', 'nearest_neighbor'])
feature_functions = pytest.mark.parametrize('feature_fn', ['average', 'max', 'nearest_neighbor'])

@mltest.parametrize.ml_cpu_only
@position_dtypes
@feature_dtypes
@position_functions
@feature_functions
def test_voxel_pooling(ml, pos_dtype, feat_dtype, position_fn, feature_fn):
    if False:
        i = 10
        return i + 15
    points = np.array([[0.5, 0.5, 0.5], [0.7, 0.2, 0.3], [0.7, 0.5, 0.9], [1.4, 1.5, 1.4], [1.7, 1.2, 1.3]], dtype=pos_dtype)
    features = np.array([[1, 1], [2, 1], [3, 1], [4, 1], [5, 1]], dtype=feat_dtype)
    voxel_size = 1
    ans = mltest.run_op(ml, ml.device, True, ml.ops.voxel_pooling, points, features, voxel_size, position_fn, feature_fn)
    if position_fn == 'average':
        expected_positions = np.stack([np.mean(points[:3], axis=0), np.mean(points[3:], axis=0)])
    elif position_fn == 'center':
        expected_positions = np.array([[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]], dtype=pos_dtype)
    elif position_fn == 'nearest_neighbor':
        expected_positions = np.array([points[0], points[3]], dtype=pos_dtype)
    assert len(ans.pooled_positions) == 2
    if np.linalg.norm(ans.pooled_positions[0] - expected_positions[0]) < np.linalg.norm(ans.pooled_positions[0] - expected_positions[1]):
        index = [0, 1]
    else:
        index = [1, 0]
    np.testing.assert_allclose(ans.pooled_positions, expected_positions[index])
    if feature_fn == 'average':
        if np.issubdtype(feat_dtype, np.integer):
            expected_features = np.stack([np.sum(features[:3], axis=0) // 3, np.sum(features[3:], axis=0) // 2])
        else:
            expected_features = np.stack([np.mean(features[:3], axis=0), np.mean(features[3:], axis=0)])
    elif feature_fn == 'max':
        expected_features = np.stack([np.max(features[:3], axis=0), np.max(features[3:], axis=0)])
    elif feature_fn == 'nearest_neighbor':
        expected_features = np.array([features[0], features[3]])
    np.testing.assert_allclose(ans.pooled_features, expected_features[index])

@mltest.parametrize.ml_cpu_only
@position_dtypes
@feature_dtypes
@position_functions
@feature_functions
def test_voxel_pooling_empty_point_set(ml, pos_dtype, feat_dtype, position_fn, feature_fn):
    if False:
        i = 10
        return i + 15
    points = np.zeros(shape=[0, 3], dtype=pos_dtype)
    features = np.zeros(shape=[0, 5], dtype=feat_dtype)
    voxel_size = 1
    ans = mltest.run_op(ml, ml.device, True, ml.ops.voxel_pooling, points, features, voxel_size, position_fn, feature_fn)
    np.testing.assert_array_equal(points, ans.pooled_positions)
    np.testing.assert_array_equal(features, ans.pooled_features)
gradient_feature_dtypes = pytest.mark.parametrize('feat_dtype', [np.float32, np.float64])

@mltest.parametrize.ml_cpu_only
@position_dtypes
@gradient_feature_dtypes
@position_functions
@feature_functions
@pytest.mark.parametrize('empty_point_set', [False])
def test_voxel_pooling_grad(ml, pos_dtype, feat_dtype, position_fn, feature_fn, empty_point_set):
    if False:
        for i in range(10):
            print('nop')
    rng = np.random.RandomState(123)
    N = 0 if empty_point_set else 50
    channels = 4
    positions = rng.uniform(0, 1, (N, 3)).astype(pos_dtype)
    features = np.linspace(0, N * channels, num=N * channels, endpoint=False)
    np.random.shuffle(features)
    features = np.reshape(features, (N, channels)).astype(feat_dtype)
    voxel_size = 0.25

    def fn(features):
        if False:
            return 10
        ans = mltest.run_op(ml, ml.device, True, ml.ops.voxel_pooling, positions, features, voxel_size, position_fn, feature_fn)
        return ans.pooled_features

    def fn_grad(features_bp, features):
        if False:
            return 10
        return mltest.run_op_grad(ml, ml.device, True, ml.ops.voxel_pooling, features, 'pooled_features', features_bp, positions, features, voxel_size, position_fn, feature_fn)
    gradient_OK = check_gradients(features, fn, fn_grad, epsilon=1)
    assert gradient_OK