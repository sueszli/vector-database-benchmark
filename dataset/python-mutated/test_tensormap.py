import open3d as o3d
import open3d.core as o3c
import numpy as np
import pytest
import pickle
import tempfile
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../..')
from open3d_test import list_devices

class WrongType:
    pass

@pytest.mark.parametrize('device', list_devices())
def test_tensormap(device):
    if False:
        for i in range(10):
            print('nop')
    dtype = o3c.float32
    tm = o3d.t.geometry.TensorMap('positions')
    assert tm.primary_key == 'positions'
    points = o3c.Tensor.ones((0, 3), dtype, device)
    colors = o3c.Tensor.ones((0, 3), dtype, device)
    tm = o3d.t.geometry.TensorMap('positions')
    assert 'positions' not in tm
    tm.positions = points
    assert 'positions' in tm
    assert 'colors' not in tm
    tm.colors = colors
    assert 'colors' in tm
    tm = o3d.t.geometry.TensorMap('positions', {'positions': points, 'colors': colors})
    assert 'positions' in tm
    assert 'colors' in tm
    with pytest.raises(RuntimeError) as excinfo:
        del tm.positions
        assert 'cannot be deleted' in str(excinfo.value)
    tm = o3d.t.geometry.TensorMap('positions')
    tm.positions = o3c.Tensor.ones((2, 3), dtype, device)
    tm.colors = o3c.Tensor.ones((2, 3), dtype, device)
    tm.positions = np.ones((3, 3), np.float32)
    tm.colors = np.ones((3, 3), np.float32)
    assert len(tm.positions) == 3
    assert len(tm.colors) == 3
    with pytest.raises(TypeError) as e:
        tm.positions = WrongType()
    with pytest.raises(TypeError) as e:
        tm.normals = WrongType()
    with pytest.raises(KeyError) as e:
        tm.primary_key = o3c.Tensor.ones((2, 3), dtype, device)
    tm = o3d.t.geometry.TensorMap('positions')
    assert isinstance(tm, o3d.t.geometry.TensorMap)
    tm.positions = o3c.Tensor.ones((2, 3), dtype, device)
    tm.colors = o3c.Tensor.ones((2, 3), dtype, device)
    colors = tm.colors
    assert len(colors) == 2
    with pytest.raises(KeyError) as e:
        normals = tm.normals
    primary_key = tm.primary_key
    assert primary_key == 'positions'

@pytest.mark.parametrize('device', list_devices())
def test_tensormap_modify(device):
    if False:
        return 10
    tm = o3d.t.geometry.TensorMap('positions')
    tm.a = o3c.Tensor([100], device=device)
    a_alias = tm.a
    a_alias[:] = o3c.Tensor([200], device=device)
    np.testing.assert_equal(a_alias.cpu().numpy(), [200])
    np.testing.assert_equal(tm.a.cpu().numpy(), [200])
    tm = o3d.t.geometry.TensorMap('positions')
    tm.a = o3c.Tensor([100], device=device)
    a_alias = tm.a
    tm.a[:] = o3c.Tensor([200], device=device)
    np.testing.assert_equal(a_alias.cpu().numpy(), [200])
    np.testing.assert_equal(tm.a.cpu().numpy(), [200])
    tm = o3d.t.geometry.TensorMap('positions')
    tm.a = o3c.Tensor([100], device=device)
    a_alias = tm.a
    a_alias = o3c.Tensor([200], device=device)
    np.testing.assert_equal(a_alias.cpu().numpy(), [200])
    np.testing.assert_equal(tm.a.cpu().numpy(), [100])
    tm = o3d.t.geometry.TensorMap('positions')
    tm.a = o3c.Tensor([100], device=device)
    a_alias = tm.a
    tm.a = o3c.Tensor([200], device=device)
    np.testing.assert_equal(a_alias.cpu().numpy(), [100])
    np.testing.assert_equal(tm.a.cpu().numpy(), [200])
    tm = o3d.t.geometry.TensorMap('positions')
    tm.a = o3c.Tensor([100], device=device)
    a_alias = tm.a
    assert id(a_alias) != id(tm.a)
    tm = o3d.t.geometry.TensorMap('positions')
    tm.a = o3c.Tensor([100], device=device)
    a_alias = tm.a
    assert len(tm) == 1
    del tm.a
    assert len(tm) == 0
    np.testing.assert_equal(a_alias.cpu().numpy(), [100])
    a_alias[:] = 200
    np.testing.assert_equal(a_alias.cpu().numpy(), [200])
    tm = o3d.t.geometry.TensorMap('positions')
    tm.a = o3c.Tensor([100], device=device)
    tm.b = o3c.Tensor([200], device=device)
    a_alias = tm.a
    b_alias = tm.b
    (tm.a, tm.b) = (tm.b, tm.a)
    np.testing.assert_equal(a_alias.cpu().numpy(), [100])
    np.testing.assert_equal(b_alias.cpu().numpy(), [200])
    np.testing.assert_equal(tm.a.cpu().numpy(), [200])
    np.testing.assert_equal(tm.b.cpu().numpy(), [100])

@pytest.mark.parametrize('device', list_devices())
def test_tensor_dict_modify(device):
    if False:
        i = 10
        return i + 15
    '\n    Same as test_tensormap_modify(), but we put Tensors in a python dict.\n    The only difference is that the id of the alias will be the same.\n    '
    tm = dict()
    tm['a'] = o3c.Tensor([100], device=device)
    a_alias = tm['a']
    a_alias[:] = o3c.Tensor([200], device=device)
    np.testing.assert_equal(a_alias.cpu().numpy(), [200])
    np.testing.assert_equal(tm['a'].cpu().numpy(), [200])
    tm = dict()
    tm['a'] = o3c.Tensor([100], device=device)
    a_alias = tm['a']
    tm['a'][:] = o3c.Tensor([200], device=device)
    np.testing.assert_equal(a_alias.cpu().numpy(), [200])
    np.testing.assert_equal(tm['a'].cpu().numpy(), [200])
    tm = dict()
    tm['a'] = o3c.Tensor([100], device=device)
    a_alias = tm['a']
    a_alias = o3c.Tensor([200], device=device)
    np.testing.assert_equal(a_alias.cpu().numpy(), [200])
    np.testing.assert_equal(tm['a'].cpu().numpy(), [100])
    tm = dict()
    tm['a'] = o3c.Tensor([100], device=device)
    a_alias = tm['a']
    tm['a'] = o3c.Tensor([200], device=device)
    np.testing.assert_equal(a_alias.cpu().numpy(), [100])
    np.testing.assert_equal(tm['a'].cpu().numpy(), [200])
    tm = dict()
    tm['a'] = o3c.Tensor([100], device=device)
    a_alias = tm['a']
    assert id(a_alias) == id(tm['a'])
    tm = dict()
    tm['a'] = o3c.Tensor([100], device=device)
    a_alias = tm['a']
    assert len(tm) == 1
    del tm['a']
    assert len(tm) == 0
    np.testing.assert_equal(a_alias.cpu().numpy(), [100])
    a_alias[:] = 200
    np.testing.assert_equal(a_alias.cpu().numpy(), [200])
    tm = dict()
    tm['a'] = o3c.Tensor([100], device=device)
    tm['b'] = o3c.Tensor([200], device=device)
    a_alias = tm['a']
    b_alias = tm['b']
    (tm['a'], tm['b']) = (tm['b'], tm['a'])
    np.testing.assert_equal(a_alias.cpu().numpy(), [100])
    np.testing.assert_equal(b_alias.cpu().numpy(), [200])
    np.testing.assert_equal(tm['a'].cpu().numpy(), [200])
    np.testing.assert_equal(tm['b'].cpu().numpy(), [100])

def test_numpy_dict_modify():
    if False:
        for i in range(10):
            print('nop')
    '\n    Same as test_tensor_dict_modify(), but we put numpy arrays in a python dict.\n    The id of the alias will be the same.\n    '
    tm = dict()
    tm['a'] = np.array([100])
    a_alias = tm['a']
    a_alias[:] = np.array([200])
    np.testing.assert_equal(a_alias, [200])
    np.testing.assert_equal(tm['a'], [200])
    tm = dict()
    tm['a'] = np.array([100])
    a_alias = tm['a']
    tm['a'][:] = np.array([200])
    np.testing.assert_equal(a_alias, [200])
    np.testing.assert_equal(tm['a'], [200])
    tm = dict()
    tm['a'] = np.array([100])
    a_alias = tm['a']
    tm['a'] = np.array([200])
    np.testing.assert_equal(a_alias, [100])
    np.testing.assert_equal(tm['a'], [200])
    tm = dict()
    tm['a'] = np.array([100])
    a_alias = tm['a']
    a_alias = np.array([200])
    np.testing.assert_equal(a_alias, [200])
    np.testing.assert_equal(tm['a'], [100])
    tm = dict()
    tm['a'] = np.array([100])
    a_alias = tm['a']
    assert id(a_alias) == id(tm['a'])
    tm = dict()
    tm['a'] = np.array([100])
    a_alias = tm['a']
    assert len(tm) == 1
    del tm['a']
    assert len(tm) == 0
    np.testing.assert_equal(a_alias, [100])
    a_alias[:] = 200
    np.testing.assert_equal(a_alias, [200])
    tm = dict()
    tm['a'] = np.array([100])
    tm['b'] = np.array([200])
    a_alias = tm['a']
    b_alias = tm['b']
    (tm['a'], tm['b']) = (tm['b'], tm['a'])
    np.testing.assert_equal(a_alias, [100])
    np.testing.assert_equal(b_alias, [200])
    np.testing.assert_equal(tm['a'], [200])
    np.testing.assert_equal(tm['b'], [100])

@pytest.mark.parametrize('device', list_devices())
def test_pickle(device):
    if False:
        print('Hello World!')
    tm = o3d.t.geometry.TensorMap('positions')
    with tempfile.TemporaryDirectory() as temp_dir:
        file_name = f'{temp_dir}/tm.pkl'
        tm.positions = o3c.Tensor.ones((10, 3), o3c.float32, device=device)
        pickle.dump(tm, open(file_name, 'wb'))
        tm_load = pickle.load(open(file_name, 'rb'))
        assert tm_load.positions.device == device and tm_load.positions.dtype == o3c.float32
        np.testing.assert_equal(tm.positions.cpu().numpy(), tm_load.positions.cpu().numpy())