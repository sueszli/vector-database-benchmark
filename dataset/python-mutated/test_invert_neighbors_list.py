import open3d as o3d
import numpy as np
import pytest
import mltest
pytestmark = mltest.default_marks
value_dtypes = pytest.mark.parametrize('dtype', [np.uint8, np.int8, np.int16, np.int32, np.int64, np.float32, np.float64])
attributes = pytest.mark.parametrize('attributes', ['scalar', 'none', 'multidim'])

@value_dtypes
@attributes
@mltest.parametrize.ml
def test_invert_neighbors_list(dtype, attributes, ml):
    if False:
        for i in range(10):
            print('nop')
    num_points = 3
    edges = np.array([[0, 0], [0, 1], [0, 2], [1, 2], [2, 1], [2, 2]], dtype=np.int32)
    neighbors_index = edges[:, 1]
    neighbors_row_splits = np.array([0, 3, 4, edges.shape[0]], dtype=np.int64)
    if attributes == 'scalar':
        neighbors_attributes = np.array([10, 20, 30, 40, 50, 60], dtype=dtype)
    elif attributes == 'none':
        neighbors_attributes = np.array([], dtype=dtype)
    elif attributes == 'multidim':
        neighbors_attributes = np.array([[10, 1], [20, 2], [30, 3], [40, 4], [50, 5], [60, 6]], dtype=dtype)
    ans = mltest.run_op(ml, ml.device, True, ml.ops.invert_neighbors_list, num_points=num_points, inp_neighbors_index=neighbors_index, inp_neighbors_row_splits=neighbors_row_splits, inp_neighbors_attributes=neighbors_attributes)
    expected_neighbors_row_splits = [0, 1, 3, edges.shape[0]]
    np.testing.assert_equal(ans.neighbors_row_splits, expected_neighbors_row_splits)
    expected_neighbors_index = [set([0]), set([0, 2]), set([0, 1, 2])]
    for (i, expected_neighbors_i) in enumerate(expected_neighbors_index):
        start = ans.neighbors_row_splits[i]
        end = ans.neighbors_row_splits[i + 1]
        neighbors_i = set(ans.neighbors_index[start:end])
        assert neighbors_i == expected_neighbors_i
    if neighbors_attributes.shape == (0,):
        assert ans.neighbors_attributes.shape == (0,)
    else:
        edge_attr_map = {tuple(k): v for (k, v) in zip(edges, neighbors_attributes)}
        for (i, _) in enumerate(expected_neighbors_index):
            start = ans.neighbors_row_splits[i]
            end = ans.neighbors_row_splits[i + 1]
            neighbors_i = ans.neighbors_index[start:end]
            attributes_i = ans.neighbors_attributes[start:end]
            for (j, attr) in zip(neighbors_i, attributes_i):
                key = (j, i)
                np.testing.assert_equal(attr, edge_attr_map[key])

@mltest.parametrize.ml
def test_invert_neighbors_list_shape_checking(ml):
    if False:
        for i in range(10):
            print('nop')
    num_points = 3
    inp_neighbors_index = np.array([0, 1, 2, 2, 1, 2], dtype=np.int32)
    inp_neighbors_row_splits = np.array([0, 3, 4, 6], dtype=np.int64)
    inp_neighbors_attributes = np.array([10, 20, 30, 40, 50, 60], dtype=np.float32)
    with pytest.raises(Exception) as einfo:
        _ = mltest.run_op(ml, ml.cpu_device, False, ml.ops.invert_neighbors_list, num_points=num_points, inp_neighbors_index=inp_neighbors_index[1:], inp_neighbors_row_splits=inp_neighbors_row_splits, inp_neighbors_attributes=inp_neighbors_attributes)
    assert 'invalid shape' in str(einfo.value)
    with pytest.raises(Exception) as einfo:
        _ = mltest.run_op(ml, ml.cpu_device, False, ml.ops.invert_neighbors_list, num_points=num_points, inp_neighbors_index=inp_neighbors_index[:, np.newaxis], inp_neighbors_row_splits=inp_neighbors_row_splits, inp_neighbors_attributes=inp_neighbors_attributes)
    assert 'invalid shape' in str(einfo.value)
    with pytest.raises(Exception) as einfo:
        _ = mltest.run_op(ml, ml.cpu_device, False, ml.ops.invert_neighbors_list, num_points=num_points, inp_neighbors_index=inp_neighbors_index, inp_neighbors_row_splits=inp_neighbors_row_splits[:, np.newaxis], inp_neighbors_attributes=inp_neighbors_attributes)
    assert 'invalid shape' in str(einfo.value)
    with pytest.raises(Exception) as einfo:
        _ = mltest.run_op(ml, ml.cpu_device, False, ml.ops.invert_neighbors_list, num_points=num_points, inp_neighbors_index=inp_neighbors_index, inp_neighbors_row_splits=inp_neighbors_row_splits, inp_neighbors_attributes=inp_neighbors_attributes[1:])
    assert 'invalid shape' in str(einfo.value)