import open3d as o3d
import open3d.core as o3c
import numpy as np
import pytest
import tempfile
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/..')
from open3d_test import list_devices

def list_dtypes():
    if False:
        print('Hello World!')
    return [o3c.float32, o3c.float64, o3c.int8, o3c.int16, o3c.int32, o3c.int64, o3c.uint8, o3c.uint16, o3c.uint32, o3c.uint64, o3c.bool]

def list_non_bool_dtypes():
    if False:
        print('Hello World!')
    return [o3c.float32, o3c.float64, o3c.int8, o3c.int16, o3c.int32, o3c.int64, o3c.uint8, o3c.uint16, o3c.uint32, o3c.uint64]

@pytest.mark.parametrize('dtype', list_non_bool_dtypes())
@pytest.mark.parametrize('device', list_devices())
def test_concatenate(dtype, device):
    if False:
        print('Hello World!')
    a = o3c.Tensor(0, dtype=dtype, device=device)
    b = o3c.Tensor(0, dtype=dtype, device=device)
    c = o3c.Tensor(0, dtype=dtype, device=device)
    with pytest.raises(RuntimeError, match='Zero-dimensional tensor can only be concatenated along axis = null, but got 0.'):
        o3c.concatenate((a, b, c))
    a = o3c.Tensor([0, 1, 2], dtype=dtype, device=device)
    b = o3c.Tensor([3, 4], dtype=dtype, device=device)
    c = o3c.Tensor([5, 6, 7], dtype=dtype, device=device)
    output_t = o3c.concatenate((a, b, c))
    output_np = np.concatenate((a.cpu().numpy(), b.cpu().numpy(), c.cpu().numpy()))
    np.testing.assert_equal(output_np, output_t.cpu().numpy())
    output_t = o3c.concatenate((a, b, c), axis=-1)
    output_np = np.concatenate((a.cpu().numpy(), b.cpu().numpy(), c.cpu().numpy()), axis=-1)
    np.testing.assert_equal(output_np, output_t.cpu().numpy())
    with pytest.raises(RuntimeError, match='Index out-of-range: dim == 1, but it must satisfy -1 <= dim <= 0'):
        o3c.concatenate((a, b, c), axis=1)
    with pytest.raises(RuntimeError, match='Index out-of-range: dim == -2, but it must satisfy -1 <= dim <= 0'):
        o3c.concatenate((a, b, c), axis=-2)
    a = o3c.Tensor([[0, 1], [2, 3]], dtype=dtype, device=device)
    b = o3c.Tensor([[4, 5]], dtype=dtype, device=device)
    c = o3c.Tensor([[6, 7]], dtype=dtype, device=device)
    output_t = o3c.concatenate((a, b, c))
    output_np = np.concatenate((a.cpu().numpy(), b.cpu().numpy(), c.cpu().numpy()))
    np.testing.assert_equal(output_np, output_t.cpu().numpy())
    output_t = o3c.concatenate((a, b, c), axis=-2)
    output_np = np.concatenate((a.cpu().numpy(), b.cpu().numpy(), c.cpu().numpy()), axis=-2)
    np.testing.assert_equal(output_np, output_t.cpu().numpy())
    with pytest.raises(RuntimeError, match='All the input tensor dimensions, other than dimension size along concatenation axis must be same, but along dimension 0, the tensor at index 0 has size 2 and the tensor at index 1 has size 1.'):
        o3c.concatenate((a, b, c), axis=1)
    with pytest.raises(RuntimeError, match='All the input tensor dimensions, other than dimension size along concatenation axis must be same, but along dimension 0, the tensor at index 0 has size 2 and the tensor at index 1 has size 1.'):
        o3c.concatenate((a, b, c), axis=-1)
    a = o3c.Tensor([[0], [1], [2]], dtype=dtype, device=device)
    b = o3c.Tensor([[3], [4], [5]], dtype=dtype, device=device)
    c = o3c.Tensor([[6], [7], [8]], dtype=dtype, device=device)
    output_t = o3c.concatenate((a, b, c), axis=0)
    output_np = np.concatenate((a.cpu().numpy(), b.cpu().numpy(), c.cpu().numpy()), axis=0)
    np.testing.assert_equal(output_np, output_t.cpu().numpy())
    output_t = o3c.concatenate((a, b, c), axis=1)
    output_np = np.concatenate((a.cpu().numpy(), b.cpu().numpy(), c.cpu().numpy()), axis=1)
    np.testing.assert_equal(output_np, output_t.cpu().numpy())
    output_t = o3c.concatenate((a, b, c), axis=-1)
    output_np = np.concatenate((a.cpu().numpy(), b.cpu().numpy(), c.cpu().numpy()), axis=-1)
    np.testing.assert_equal(output_np, output_t.cpu().numpy())
    output_t = o3c.concatenate((a, b, c), axis=-2)
    output_np = np.concatenate((a.cpu().numpy(), b.cpu().numpy(), c.cpu().numpy()), axis=-2)
    np.testing.assert_equal(output_np, output_t.cpu().numpy())
    with pytest.raises(RuntimeError, match='Index out-of-range: dim == 2, but it must satisfy -2 <= dim <= 1'):
        o3c.concatenate((a, b, c), axis=2)
    with pytest.raises(RuntimeError, match='Index out-of-range: dim == -3, but it must satisfy -2 <= dim <= 1'):
        o3c.concatenate((a, b, c), axis=-3)
    a = o3c.Tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]], dtype=o3c.Dtype.Float32, device=device)
    output_t = o3c.concatenate(a, axis=1)
    output_np = np.concatenate(a.cpu().numpy(), axis=1)
    np.testing.assert_equal(output_np, output_t.cpu().numpy())
    a = o3c.Tensor([[0, 1], [2, 3]], dtype=o3c.Dtype.Float32, device=device)
    b = o3c.Tensor([[4, 5]], dtype=o3c.Dtype.Float64, device=device)
    with pytest.raises(RuntimeError, match='Tensor has dtype Float64, but is expected to have Float32'):
        o3c.concatenate((a, b))

@pytest.mark.parametrize('dtype', list_non_bool_dtypes())
@pytest.mark.parametrize('device', list_devices())
def test_append(dtype, device):
    if False:
        while True:
            i = 10
    self = o3c.Tensor(0, dtype=dtype, device=device)
    values = o3c.Tensor(1, dtype=dtype, device=device)
    output_t = o3c.append(self=self, values=values)
    output_np = np.append(arr=self.cpu().numpy(), values=values.cpu().numpy())
    np.testing.assert_equal(output_np, output_t.cpu().numpy())
    with pytest.raises(RuntimeError, match='Zero-dimensional tensor can only be concatenated along axis = null, but got 0.'):
        o3c.append(self=self, values=values, axis=0)
    self = o3c.Tensor([0, 1], dtype=dtype, device=device)
    values = o3c.Tensor([2, 3, 4], dtype=dtype, device=device)
    output_t = o3c.append(self=self, values=values)
    output_np = np.append(arr=self.cpu().numpy(), values=values.cpu().numpy())
    np.testing.assert_equal(output_np, output_t.cpu().numpy())
    output_t = o3c.append(self=self, values=values, axis=0)
    output_np = np.append(arr=self.cpu().numpy(), values=values.cpu().numpy(), axis=0)
    np.testing.assert_equal(output_np, output_t.cpu().numpy())
    output_t = o3c.append(self=self, values=values, axis=-1)
    output_np = np.append(arr=self.cpu().numpy(), values=values.cpu().numpy(), axis=-1)
    np.testing.assert_equal(output_np, output_t.cpu().numpy())
    with pytest.raises(RuntimeError, match='Index out-of-range: dim == 1, but it must satisfy -1 <= dim <= 0'):
        o3c.append(self=self, values=values, axis=1)
    with pytest.raises(RuntimeError, match='Index out-of-range: dim == -2, but it must satisfy -1 <= dim <= 0'):
        o3c.append(self=self, values=values, axis=-2)
    self = o3c.Tensor([[0, 1], [2, 3]], dtype=dtype, device=device)
    values = o3c.Tensor([[4, 5], [6, 7]], dtype=dtype, device=device)
    output_t = o3c.append(self=self, values=values)
    output_np = np.append(arr=self.cpu().numpy(), values=values.cpu().numpy())
    np.testing.assert_equal(output_np, output_t.cpu().numpy())
    output_t = o3c.append(self=self, values=values, axis=0)
    output_np = np.append(arr=self.cpu().numpy(), values=values.cpu().numpy(), axis=0)
    np.testing.assert_equal(output_np, output_t.cpu().numpy())
    output_t = o3c.append(self=self, values=values, axis=1)
    output_np = np.append(arr=self.cpu().numpy(), values=values.cpu().numpy(), axis=1)
    np.testing.assert_equal(output_np, output_t.cpu().numpy())
    output_t = o3c.append(self=self, values=values, axis=-1)
    output_np = np.append(arr=self.cpu().numpy(), values=values.cpu().numpy(), axis=-1)
    np.testing.assert_equal(output_np, output_t.cpu().numpy())
    output_t = o3c.append(self=self, values=values, axis=-2)
    output_np = np.append(arr=self.cpu().numpy(), values=values.cpu().numpy(), axis=-2)
    np.testing.assert_equal(output_np, output_t.cpu().numpy())
    with pytest.raises(RuntimeError, match='Index out-of-range: dim == 2, but it must satisfy -2 <= dim <= 1'):
        o3c.append(self=self, values=values, axis=2)
    with pytest.raises(RuntimeError, match='Index out-of-range: dim == -3, but it must satisfy -2 <= dim <= 1'):
        o3c.append(self=self, values=values, axis=-3)
    self = o3c.Tensor([[0, 1], [2, 3]], dtype=dtype, device=device)
    values = o3c.Tensor([[4, 5]], dtype=dtype, device=device)
    output_t = o3c.append(self=self, values=values)
    output_np = np.append(arr=self.cpu().numpy(), values=values.cpu().numpy())
    np.testing.assert_equal(output_np, output_t.cpu().numpy())
    output_t = o3c.append(self=self, values=values, axis=0)
    output_np = np.append(arr=self.cpu().numpy(), values=values.cpu().numpy(), axis=0)
    np.testing.assert_equal(output_np, output_t.cpu().numpy())
    output_t = o3c.append(self=self, values=values, axis=-2)
    output_np = np.append(arr=self.cpu().numpy(), values=values.cpu().numpy(), axis=-2)
    np.testing.assert_equal(output_np, output_t.cpu().numpy())
    with pytest.raises(RuntimeError, match='All the input tensor dimensions, other than dimension size along concatenation axis must be same, but along dimension 0, the tensor at index 0 has size 2 and the tensor at index 1 has size 1.'):
        o3c.append(self=self, values=values, axis=1)
    with pytest.raises(RuntimeError, match='All the input tensor dimensions, other than dimension size along concatenation axis must be same, but along dimension 0, the tensor at index 0 has size 2 and the tensor at index 1 has size 1.'):
        o3c.append(self=self, values=values, axis=-1)
    self = o3c.Tensor([[0, 1], [2, 3]], dtype=o3c.Dtype.Float32, device=device)
    values = o3c.Tensor([[4, 5]], dtype=o3c.Dtype.Float64, device=device)
    with pytest.raises(RuntimeError, match='Tensor has dtype Float64, but is expected to have Float32'):
        o3c.append(self=self, values=values)