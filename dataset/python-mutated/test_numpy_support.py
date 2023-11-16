import numpy as np
import pytest
import torch
import ray
from ray.air.util.tensor_extensions.utils import create_ragged_ndarray
from ray.data.tests.conftest import *
from ray.tests.conftest import *

class UserObj:

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        return isinstance(other, UserObj)

def do_map_batches(data):
    if False:
        print('Hello World!')
    ds = ray.data.range(1)
    ds = ds.map_batches(lambda x: {'output': data})
    return ds.take_batch()['output']

def assert_structure_equals(a, b):
    if False:
        for i in range(10):
            print('nop')
    assert type(a) == type(b), (type(a), type(b))
    assert type(a[0]) == type(b[0]), (type(a[0]), type(b[0]))
    assert a.dtype == b.dtype
    assert a.shape == b.shape
    for i in range(len(a)):
        assert np.array_equiv(a[i], b[i]), (i, a, b)

def test_list_of_scalars(ray_start_regular_shared):
    if False:
        return 10
    data = [1, 2, 3]
    output = do_map_batches(data)
    assert_structure_equals(output, np.array([1, 2, 3], dtype=np.int64))

def test_list_of_numpy_scalars(ray_start_regular_shared):
    if False:
        print('Hello World!')
    data = [np.int64(1), np.int64(2), np.int64(3)]
    output = do_map_batches(data)
    assert_structure_equals(output, np.array([1, 2, 3], dtype=np.int64))

def test_list_of_objects(ray_start_regular_shared):
    if False:
        return 10
    data = [1, 2, 3, UserObj()]
    output = do_map_batches(data)
    assert_structure_equals(output, np.array([1, 2, 3, UserObj()]))

def test_array_like(ray_start_regular_shared):
    if False:
        print('Hello World!')
    data = torch.Tensor([1, 2, 3])
    output = do_map_batches(data)
    assert_structure_equals(output, np.array([1.0, 2.0, 3.0], dtype=np.float32))

def test_list_of_arrays(ray_start_regular_shared):
    if False:
        while True:
            i = 10
    data = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    output = do_map_batches(data)
    assert_structure_equals(output, np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64))

def test_list_of_array_like(ray_start_regular_shared):
    if False:
        for i in range(10):
            print('nop')
    data = [torch.Tensor([1, 2, 3]), torch.Tensor([4, 5, 6])]
    output = do_map_batches(data)
    assert_structure_equals(output, np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))

def test_ragged_array_like(ray_start_regular_shared):
    if False:
        i = 10
        return i + 15
    data = [torch.Tensor([1, 2, 3]), torch.Tensor([1, 2])]
    output = do_map_batches(data)
    assert_structure_equals(output, np.array([np.array([1, 2, 3]), np.array([1, 2])], dtype=object))
    data = [torch.zeros((3, 5, 10)), torch.zeros((3, 8, 8))]
    output = do_map_batches(data)
    assert_structure_equals(output, create_ragged_ndarray([np.zeros((3, 5, 10)), np.zeros((3, 8, 8))]))

def test_scalar_nested_arrays(ray_start_regular_shared):
    if False:
        while True:
            i = 10
    data = [[[1]], [[2]]]
    output = do_map_batches(data)
    assert_structure_equals(output, np.array([[[1]], [[2]]]))

def test_scalar_lists_not_converted(ray_start_regular_shared):
    if False:
        for i in range(10):
            print('nop')
    data = [[1, 2], [1, 2]]
    output = do_map_batches(data)
    assert_structure_equals(output, create_ragged_ndarray([[1, 2], [1, 2]]))
    data = [[1, 2, 3], [1, 2]]
    output = do_map_batches(data)
    assert_structure_equals(output, create_ragged_ndarray([[1, 2, 3], [1, 2]]))

def test_scalar_numpy(ray_start_regular_shared):
    if False:
        i = 10
        return i + 15
    data = np.int64(1)
    ds = ray.data.range(2, parallelism=1)
    ds = ds.map(lambda x: {'output': data})
    output = ds.take_batch()['output']
    assert_structure_equals(output, np.array([1, 1], dtype=np.int64))

def test_scalar_arrays(ray_start_regular_shared):
    if False:
        for i in range(10):
            print('nop')
    data = np.array([1, 2, 3])
    ds = ray.data.range(2, parallelism=1)
    ds = ds.map(lambda x: {'output': data})
    output = ds.take_batch()['output']
    assert_structure_equals(output, np.array([[1, 2, 3], [1, 2, 3]], dtype=np.int64))

def test_bytes(ray_start_regular_shared):
    if False:
        for i in range(10):
            print('nop')
    'Tests that bytes are converted to object dtype instead of zero-terminated.'
    data = b'\x1a\n\x00\n\x1a'
    ds = ray.data.range(1, parallelism=1)
    ds = ds.map(lambda x: {'output': data})
    output = ds.take_batch()['output']
    assert_structure_equals(output, np.array([b'\x1a\n\x00\n\x1a'], dtype=object))

def test_scalar_array_like(ray_start_regular_shared):
    if False:
        for i in range(10):
            print('nop')
    data = torch.Tensor([1, 2, 3])
    ds = ray.data.range(2, parallelism=1)
    ds = ds.map(lambda x: {'output': data})
    output = ds.take_batch()['output']
    assert_structure_equals(output, np.array([[1, 2, 3], [1, 2, 3]], dtype=np.float32))

def test_scalar_ragged_arrays(ray_start_regular_shared):
    if False:
        i = 10
        return i + 15
    data = [np.array([1, 2, 3]), np.array([1, 2])]
    ds = ray.data.range(2, parallelism=1)
    ds = ds.map(lambda x: {'output': data[x['id']]})
    output = ds.take_batch()['output']
    assert_structure_equals(output, np.array([np.array([1, 2, 3]), np.array([1, 2])], dtype=object))

def test_scalar_ragged_array_like(ray_start_regular_shared):
    if False:
        for i in range(10):
            print('nop')
    data = [torch.Tensor([1, 2, 3]), torch.Tensor([1, 2])]
    ds = ray.data.range(2, parallelism=1)
    ds = ds.map(lambda x: {'output': data[x['id']]})
    output = ds.take_batch()['output']
    assert_structure_equals(output, np.array([np.array([1, 2, 3]), np.array([1, 2])], dtype=object))
    data = [torch.zeros((3, 5, 10)), torch.zeros((3, 8, 8))]
    ds = ray.data.range(2, parallelism=1)
    ds = ds.map(lambda x: {'output': data[x['id']]})
    output = ds.take_batch()['output']
    assert_structure_equals(output, create_ragged_ndarray([np.zeros((3, 5, 10)), np.zeros((3, 8, 8))]))

def test_complex_ragged_arrays(ray_start_regular_shared):
    if False:
        while True:
            i = 10
    data = [[{'a': 1}, {'a': 2}, {'a': 3}], [{'b': 1}]]
    output = do_map_batches(data)
    assert_structure_equals(output, create_ragged_ndarray(data))
    data = ['hi', 1, None, [[[[]]]], {'a': [[{'b': 2, 'c': UserObj()}]]}, UserObj()]
    output = do_map_batches(data)
    assert_structure_equals(output, create_ragged_ndarray(data))
if __name__ == '__main__':
    import sys
    sys.exit(pytest.main(['-v', __file__]))