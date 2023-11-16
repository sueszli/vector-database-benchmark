"""Tests the implementation of the continuous convolution ops"""
import open3d as o3d
import numpy as np
np.set_printoptions(linewidth=600)
np.set_printoptions(threshold=np.inf)
import pytest
import mltest
from check_gradients import check_gradients
pytestmark = mltest.default_marks

@pytest.mark.parametrize('filter_size, out_channels, in_channels, with_inp_importance, with_normalization', [([3, 5, 1], 2, 7, True, False), ([3, 3, 3], 1, 1, False, False), ([5, 5, 5], 5, 3, False, True)])
@mltest.parametrize.ml_tf_only
@pytest.mark.parametrize('feat_out_type', [('float32', 'float32')])
@pytest.mark.parametrize('real_type', ['float32'])
def test_compare_to_conv3d(ml, feat_out_type, real_type, filter_size, out_channels, in_channels, with_inp_importance, with_normalization):
    if False:
        for i in range(10):
            print('nop')
    'Compares to the 3D convolution in tensorflow'
    (feat_type, out_type) = feat_out_type
    try:
        import tensorflow as tf
    except ImportError:
        return
    if ml.device_is_gpu and feat_type == 'bfloat16':
        return
    mltensor = mltest.MLTensor(ml.module)
    np_real_type = getattr(np, real_type)
    np_feat_type = getattr(tf, feat_type).as_numpy_dtype
    np_out_type = getattr(tf, out_type).as_numpy_dtype
    mltensor.set_seed(0)
    np.random.seed(0)
    conv_attrs = {'align_corners': False, 'coordinate_mapping': 'identity', 'normalize': with_normalization, 'interpolation': 'nearest_neighbor', 'max_temp_mem_MB': 0, 'output_type': getattr(tf, out_type)}
    filters = mltensor.random_uniform(size=(*filter_size, in_channels, out_channels), dtype=feat_type)
    max_grid_extent = 10
    inp_positions = np.unique(np.random.randint(0, max_grid_extent, (256, 3)).astype(np_real_type), axis=0)
    inp_positions_int = inp_positions.astype(np.int32)
    if with_inp_importance:
        inp_importance = mltensor.random_uniform(inp_positions.shape[0:1], dtype=feat_type, minval=-1, maxval=1)
    else:
        inp_importance = mltensor.empty((0,), dtype=feat_type)
    out_positions = np.unique(np.random.randint(np.max(filter_size) // 2, max_grid_extent - np.max(filter_size) // 2, (5, 3)).astype(np_real_type), axis=0)
    out_positions_int = out_positions.astype(np.int32)
    voxel_size = np.array([1, 1, 1], dtype=np_real_type)
    extent = voxel_size[np.newaxis, :] * np.array(filter_size[::-1])
    extent = extent.astype(np_real_type)
    offset = np.array([0.0, 0.0, 0.0], dtype=np_real_type)
    inp_features = mltensor.random_uniform(size=inp_positions.shape[0:1] + (in_channels,), dtype=feat_type)
    fixed_radius_search = ml.layers.FixedRadiusSearch(metric='Linf')
    (neighbors_index, neighbors_row_splits, _) = mltest.run_op(ml, ml.device, False, fixed_radius_search, inp_positions / extent, out_positions / extent, voxel_size[0] / 2 + 0.01)
    neighbors_importance = mltensor.empty((0,), dtype=feat_type)
    y = mltest.run_op(ml, ml.device, True, ml.ops.continuous_conv, filters, out_positions, extent, offset, inp_positions, inp_features, inp_importance, neighbors_index, neighbors_importance, neighbors_row_splits, **conv_attrs)
    inp_volume = np.zeros((1, max_grid_extent, max_grid_extent, max_grid_extent, in_channels))
    inp_volume = mltensor.zeros((1, max_grid_extent, max_grid_extent, max_grid_extent, in_channels), dtype=feat_type).numpy()
    if with_inp_importance:
        inp_features *= inp_importance[:, None]
    inp_volume[0, inp_positions_int[:, 2], inp_positions_int[:, 1], inp_positions_int[:, 0], :] = inp_features.numpy()
    y_conv3d = tf.nn.conv3d(inp_volume, filters, strides=[1] * 5, padding='SAME').numpy()
    y_conv3d = np.ascontiguousarray(y_conv3d[0, out_positions_int[:, 2], out_positions_int[:, 1], out_positions_int[:, 0], :]).astype(np_out_type)
    if with_normalization:
        for (i, v) in enumerate(y_conv3d):
            num_neighbors = neighbors_row_splits[i + 1] - neighbors_row_splits[i]
            v /= np_feat_type(int(num_neighbors))
    tol = {'float32': {'rtol': 1e-05, 'atol': 1e-08}, 'bfloat16': {'rtol': 0.01, 'atol': 0.01}}
    np.testing.assert_allclose(y, y_conv3d, **tol[feat_type])

@mltest.parametrize.ml
@pytest.mark.parametrize('filter_size, out_channels, in_channels, with_inp_importance, with_neighbors_importance, with_individual_extent, with_normalization, align_corners, coordinate_mapping, interpolation', [([3, 5, 1], 2, 7, True, False, False, False, True, 'identity', 'nearest_neighbor'), ([3, 3, 3], 1, 1, False, False, True, False, False, 'ball_to_cube_radial', 'linear'), ([5, 5, 5], 5, 3, False, True, False, True, False, 'ball_to_cube_volume_preserving', 'linear_border'), ([5, 1, 3], 3, 4, False, True, False, False, False, 'identity', 'linear')])
@pytest.mark.parametrize('dtype', [np.float32])
def test_cconv_gradient(ml, dtype, filter_size, out_channels, in_channels, with_inp_importance, with_neighbors_importance, with_individual_extent, with_normalization, align_corners, coordinate_mapping, interpolation):
    if False:
        i = 10
        return i + 15
    if dtype == np.float64:
        tolerance = {'atol': 1e-05, 'rtol': 0.01, 'epsilon': 1e-06}
    elif dtype == np.float32:
        tolerance = {'atol': 0.01, 'rtol': 0.1, 'epsilon': 0.001}
    np.random.seed(0)
    conv_attrs = {'align_corners': align_corners, 'coordinate_mapping': coordinate_mapping, 'normalize': with_normalization, 'interpolation': interpolation}
    filters = np.random.random(size=(*filter_size, in_channels, out_channels)).astype(dtype)
    inp_positions = np.random.rand(128, 3).astype(dtype)
    if with_inp_importance:
        inp_importance = np.random.rand(inp_positions.shape[0]).astype(dtype)
    else:
        inp_importance = np.empty((0,), dtype=dtype)
    out_positions = np.random.rand(5, 3).astype(dtype)
    if with_individual_extent:
        extent = 0.4 + 0.01 * (np.random.rand(out_positions.shape[0], 1) - 0.5)
        extent = extent.astype(dtype)
    else:
        extent = np.array([[0.4]], dtype=dtype)
    offset = np.array([0.0, 0.0, 0.0], dtype=dtype)
    inp_features = np.random.uniform(size=inp_positions.shape[0:1] + (in_channels,)).astype(dtype)
    fixed_radius_search = ml.layers.FixedRadiusSearch(metric='Linf')
    (neighbors_index, neighbors_row_splits, _) = mltest.run_op(ml, ml.device, False, fixed_radius_search, inp_positions, out_positions, extent[0, 0] / 2)
    if with_neighbors_importance:
        neighbors_importance = np.random.rand(neighbors_index.shape[0]).astype(dtype) - 0.5
        neighbors_importance_sum = mltest.run_op(ml, ml.device, False, ml.ops.reduce_subarrays_sum, neighbors_importance, neighbors_row_splits)
    else:
        neighbors_importance = np.empty((0,), dtype=dtype)
        neighbors_importance_sum = np.empty((0,), dtype=dtype)
    (inverted_neighbors_index, inverted_neighbors_row_splits, inverted_neighbors_importance) = mltest.run_op(ml, ml.device, False, ml.ops.invert_neighbors_list, inp_positions.shape[0], neighbors_index, neighbors_row_splits, neighbors_importance)

    def conv_infeats(inp_features):
        if False:
            return 10
        return mltest.run_op(ml, ml.device, True, ml.ops.continuous_conv, filters=filters, out_positions=out_positions, extents=extent, offset=offset, inp_positions=inp_positions, inp_features=inp_features, inp_importance=inp_importance, neighbors_index=neighbors_index, neighbors_importance=neighbors_importance, neighbors_row_splits=neighbors_row_splits, **conv_attrs)

    def conv_filter(filters):
        if False:
            return 10
        return mltest.run_op(ml, ml.device, True, ml.ops.continuous_conv, filters=filters, out_positions=out_positions, extents=extent, offset=offset, inp_positions=inp_positions, inp_features=inp_features, inp_importance=inp_importance, neighbors_index=neighbors_index, neighbors_importance=neighbors_importance, neighbors_row_splits=neighbors_row_splits, **conv_attrs)

    def conv_filter_backprop(out_features_gradient, filters):
        if False:
            return 10
        return mltest.run_op_grad(ml, ml.device, True, ml.ops.continuous_conv, filters, '', out_features_gradient, filters=filters, out_positions=out_positions, extents=extent, offset=offset, inp_positions=inp_positions, inp_features=inp_features, inp_importance=inp_importance, neighbors_index=neighbors_index, neighbors_importance=neighbors_importance, neighbors_row_splits=neighbors_row_splits, **conv_attrs)

    def conv_infeat_backprop(out_features_gradient, inp_features):
        if False:
            return 10
        return mltest.run_op_grad(ml, ml.device, True, ml.ops.continuous_conv, inp_features, '', out_features_gradient, filters=filters, out_positions=out_positions, extents=extent, offset=offset, inp_positions=inp_positions, inp_features=inp_features, inp_importance=inp_importance, neighbors_index=neighbors_index, neighbors_importance=neighbors_importance, neighbors_row_splits=neighbors_row_splits, **conv_attrs)

    def conv_transpose_filter(filters):
        if False:
            return 10
        return mltest.run_op(ml, ml.device, True, ml.ops.continuous_conv_transpose, filters=filters, out_positions=inp_positions, out_importance=inp_importance, extents=extent, offset=offset, inp_positions=out_positions, inp_features=y_arr, inp_neighbors_index=neighbors_index, inp_neighbors_importance_sum=neighbors_importance_sum, inp_neighbors_row_splits=neighbors_row_splits, neighbors_index=inverted_neighbors_index, neighbors_importance=inverted_neighbors_importance, neighbors_row_splits=inverted_neighbors_row_splits, **conv_attrs)

    def conv_transpose_infeats(inp_features):
        if False:
            print('Hello World!')
        return mltest.run_op(ml, ml.device, True, ml.ops.continuous_conv_transpose, filters=filters.transpose([0, 1, 2, 4, 3]), out_positions=inp_positions, out_importance=inp_importance, extents=extent, offset=offset, inp_positions=out_positions, inp_features=inp_features, inp_neighbors_index=neighbors_index, inp_neighbors_importance_sum=neighbors_importance_sum, inp_neighbors_row_splits=neighbors_row_splits, neighbors_index=inverted_neighbors_index, neighbors_importance=inverted_neighbors_importance, neighbors_row_splits=inverted_neighbors_row_splits, **conv_attrs)

    def conv_transpose_filter_backprop(out_features_gradient, filters):
        if False:
            while True:
                i = 10
        return mltest.run_op_grad(ml, ml.device, True, ml.ops.continuous_conv_transpose, filters, '', out_features_gradient, filters=filters, out_positions=inp_positions, out_importance=inp_importance, extents=extent, offset=offset, inp_positions=out_positions, inp_features=y_arr, inp_neighbors_index=neighbors_index, inp_neighbors_importance_sum=neighbors_importance_sum, inp_neighbors_row_splits=neighbors_row_splits, neighbors_index=inverted_neighbors_index, neighbors_importance=inverted_neighbors_importance, neighbors_row_splits=inverted_neighbors_row_splits, **conv_attrs)

    def conv_transpose_infeat_backprop(out_features_gradient, inp_features):
        if False:
            for i in range(10):
                print('nop')
        return mltest.run_op_grad(ml, ml.device, True, ml.ops.continuous_conv_transpose, inp_features, '', out_features_gradient, filters=filters.transpose([0, 1, 2, 4, 3]), out_positions=inp_positions, out_importance=inp_importance, extents=extent, offset=offset, inp_positions=out_positions, inp_features=inp_features, inp_neighbors_index=neighbors_index, inp_neighbors_importance_sum=neighbors_importance_sum, inp_neighbors_row_splits=neighbors_row_splits, neighbors_index=inverted_neighbors_index, neighbors_importance=inverted_neighbors_importance, neighbors_row_splits=inverted_neighbors_row_splits, **conv_attrs)
    y_arr = conv_infeats(inp_features)
    dbg = {}
    filter_gradient_OK = check_gradients(filters, conv_filter, conv_filter_backprop, debug_outputs=dbg, **tolerance)
    assert filter_gradient_OK
    feature_gradient_OK = check_gradients(inp_features, conv_infeats, conv_infeat_backprop, debug_outputs=dbg, **tolerance)
    assert feature_gradient_OK
    transpose_filter_gradient_OK = check_gradients(filters.transpose([0, 1, 2, 4, 3]), conv_transpose_filter, conv_transpose_filter_backprop, debug_outputs=dbg, **tolerance)
    assert transpose_filter_gradient_OK
    transpose_feature_gradient_OK = check_gradients(y_arr, conv_transpose_infeats, conv_transpose_infeat_backprop, debug_outputs=dbg, **tolerance)
    assert transpose_feature_gradient_OK