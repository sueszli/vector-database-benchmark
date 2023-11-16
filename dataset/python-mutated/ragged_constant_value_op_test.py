"""Tests for ragged_factory_ops.constant_value."""
from absl.testing import parameterized
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor_value
from tensorflow.python.platform import googletest

@test_util.run_all_in_graph_and_eager_modes
class RaggedConstantValueOpTest(test_util.TensorFlowTestCase, parameterized.TestCase):

    @parameterized.parameters(dict(pylist='x', expected_shape=()), dict(pylist=[1, 2, 3], expected_shape=(3,)), dict(pylist=[[1, 2, 3], [4], [5, 6]], expected_shape=(3, None)), dict(pylist=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], expected_shape=(3, None)), dict(pylist=[[[1, 2], [3, 4]], [], [[5, 6], [7, 8], [9, 0]]], expected_shape=(3, None, None)), dict(pylist=[[[1, 2], [3, 4]], [], [[5, 6], [7, 8], [9, 0]]], ragged_rank=1, expected_shape=(3, None, 2)), dict(pylist=[[[1, 2], [3, 4]], [], [[5, 6], [7, 8], [9, 0]]], inner_shape=(2,), expected_shape=(3, None, 2)), dict(pylist=[[[1, 2], [3, 4]], [], [[5, 6], [7, 8], [9, 0]]], ragged_rank=1, inner_shape=(2,), expected_shape=(3, None, 2)), dict(pylist=[[[1, 2], np.array([3, np.array(4)])], np.array([]), [[5, 6], [7, 8], [9, 0]]], expected_shape=(3, None, None)), dict(pylist=[[[1, 2], np.array([3, np.array(4)])], np.array([]), [[5, 6], [7, 8], [9, 0]]], ragged_rank=1, expected_shape=(3, None, 2)), dict(pylist=[[np.array([3, np.array(4)]), [1, 2]], np.array([]), [[5, 6], [7, 8], [9, 0]]], ragged_rank=1, expected_shape=(3, None, 2)), dict(pylist=[[[1, 2], np.array([3, np.array(4)])], np.array([]), [[5, 6], [7, 8], [9, 0]]], inner_shape=(2,), expected_shape=(3, None, 2)), dict(pylist=[[[1, 2], np.array([3, np.array(4)])], np.array([]), [[5, 6], [7, 8], [9, 0]]], ragged_rank=1, inner_shape=(2,), expected_shape=(3, None, 2)), dict(pylist=[[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[2, 4], [6, 8]], [[1, 5], [7, 9]]]], expected_shape=(2, None, None, None)), dict(pylist=[[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[2, 4], [6, 8]], [[1, 5], [7, 9]]]], ragged_rank=1, expected_shape=(2, None, 2, 2)), dict(pylist=[[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[2, 4], [6, 8]], [[1, 5], [7, 9]]]], inner_shape=(2,), expected_shape=(2, None, None, 2)), dict(pylist=[[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[2, 4], [6, 8]], [[1, 5], [7, 9]]]], inner_shape=(2, 2), expected_shape=(2, None, 2, 2)), dict(pylist=np.array([[[np.array([1, 2]), [3, 4]], [[5, 6], [7, 8]]], np.array([[[2, 4], [6, 8]], [[1, 5], [7, 9]]])]), expected_shape=(2, None, None, None)), dict(pylist=[], expected_shape=(0,)), dict(pylist=[[], [], np.array([])], expected_shape=(3, None)), dict(pylist=[[[], []], [], [[], [[]]]], expected_shape=(3, None, None, None)), dict(pylist=np.array([np.array([[], []]), np.array([]), [[], [[]]]], dtype=object), expected_shape=(3, None, None, None)), dict(pylist=[], ragged_rank=1, expected_shape=(0, None)), dict(pylist=[], ragged_rank=2, expected_shape=(0, None, None)), dict(pylist=[], inner_shape=(0, 100, 20), expected_shape=(0, 100, 20)), dict(pylist=[], ragged_rank=1, inner_shape=(100, 20), expected_shape=(0, None, 100, 20)), dict(pylist=[], ragged_rank=2, inner_shape=(100, 20), expected_shape=(0, None, None, 100, 20)), dict(pylist=[[], [], []], ragged_rank=2, expected_shape=(3, None, None)), dict(pylist=[], inner_shape=(0,), expected_shape=(0,)), dict(pylist=[[]], inner_shape=(1, 0), expected_shape=(1, 0)), dict(pylist=np.array([]), ragged_rank=1, inner_shape=(100, 20), expected_shape=(0, None, 100, 20)), dict(pylist=[], expected_dtype=np.float64), dict(pylist=[[[], [[[]], []]]], expected_dtype=np.float64), dict(pylist=[[1, 2], [3], [4, 5, 6]], expected_dtype=np.int64), dict(pylist=[[1.0, 2.0], [], [4.0, 5.0, 6.0]], expected_dtype=np.float64), dict(pylist=[[1, 2], [3.0], [4, 5, 6]], expected_dtype=np.float64), dict(pylist=[[b'a', b'b'], [b'c']], expected_dtype=np.dtype('S1')), dict(pylist=[[True]], expected_dtype=np.bool_), dict(pylist=[np.array([1, 2]), np.array([3.0]), [4, 5, 6]], expected_dtype=np.float64), dict(pylist=[], dtype=np.float32), dict(pylist=[], dtype=np.dtype('S1')), dict(pylist=[[1, 2], [3], [4, 5, 6]], dtype=np.int64), dict(pylist=[[1, 2], [3], [4, 5, 6]], dtype=np.int32), dict(pylist=[[1, 2], [3], [4, 5, 6]], dtype=np.float32), dict(pylist=[[1.0, 2.0], [3.0], [4.0, 5.0, 6.0]], dtype=np.float16), dict(pylist=[[1.0, 2.0], [3.0], [4.0, 5.0, 6.0]], dtype=np.float32), dict(pylist=[[b'a', b'b'], [b'c'], [b'd', b'e', b'f']], dtype=np.dtype('S1')), dict(pylist=[], dtype=dtypes.float32, expected_dtype=np.float32), dict(pylist=[], dtype=dtypes.int32, expected_dtype=np.int32))
    def testRaggedValues(self, pylist, dtype=None, ragged_rank=None, inner_shape=None, expected_shape=None, expected_dtype=None):
        if False:
            while True:
                i = 10
        'Tests that `ragged_value(pylist).to_list() == pylist`.'
        rt = ragged_factory_ops.constant_value(pylist, dtype=dtype, ragged_rank=ragged_rank, inner_shape=inner_shape)
        pylist = _normalize_pylist(pylist)
        if expected_dtype is not None:
            self.assertEqual(rt.dtype, expected_dtype)
        elif dtype is not None:
            self.assertEqual(rt.dtype, dtype)
        if ragged_rank is not None:
            if isinstance(rt, ragged_tensor_value.RaggedTensorValue):
                self.assertEqual(rt.ragged_rank, ragged_rank)
            else:
                self.assertEqual(0, ragged_rank)
        if inner_shape is not None:
            if isinstance(rt, ragged_tensor_value.RaggedTensorValue):
                self.assertEqual(rt.flat_values.shape[1:], inner_shape)
            else:
                self.assertEqual(rt.shape, inner_shape)
        if expected_shape is not None:
            self.assertEqual(tuple(rt.shape), expected_shape)
        if rt.shape:
            if isinstance(rt, ragged_tensor_value.RaggedTensorValue):
                self.assertEqual(rt.to_list(), pylist)
            else:
                self.assertEqual(rt.tolist(), pylist)
            if expected_shape is not None:
                self.assertEqual(rt.shape, expected_shape)
        else:
            self.assertEqual(rt, pylist)
            if expected_shape is not None:
                self.assertEqual((), expected_shape)

    @parameterized.parameters(dict(pylist=12, ragged_rank=1, exception=ValueError, message='Invalid pylist=12: incompatible with ragged_rank=1'), dict(pylist=np.array(12), ragged_rank=1, exception=ValueError, message='Invalid pylist=array\\(12\\): incompatible with ragged_rank=1'), dict(pylist=12, inner_shape=(1,), exception=ValueError, message='Invalid pylist=12: incompatible with dim\\(inner_shape\\)=1'), dict(pylist=[[[1], [2]]], ragged_rank=-1, exception=ValueError, message='Invalid ragged_rank=-1: must be nonnegative'), dict(pylist=[[1, [2]]], exception=ValueError, message='all scalar values must have the same nesting depth'), dict(pylist=[[[1]], [[[2]]]], exception=ValueError, message='all scalar values must have the same nesting depth'), dict(pylist=[[1], [[]]], exception=ValueError, message='Invalid pylist=.*: empty list nesting is greater than scalar value nesting'), dict(pylist=[1, 2, 3], ragged_rank=1, exception=ValueError, message='pylist has scalar values depth 1, but ragged_rank=1 requires scalar value depth greater than 1'), dict(pylist=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], ragged_rank=2, exception=ValueError, message='pylist has scalar values depth 2, but ragged_rank=2 requires scalar value depth greater than 2'), dict(pylist=[1, 2, 3], inner_shape=(1, 1), exception=ValueError, message='cannot reshape array'), dict(pylist=[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], inner_shape=(2, 2), ragged_rank=1, exception=ValueError, message='Invalid pylist=.*: incompatible with ragged_rank=1 and dim\\(inner_shape\\)=2'), dict(pylist=[[[1, 2], [3, 4]], [[5, 6], [7, 8, 9]]], ragged_rank=1, exception=ValueError, message='inner values have inconsistent shape'), dict(pylist=[[[], [[]]]], ragged_rank=1, exception=ValueError, message='inner values have inconsistent shape'))
    def testRaggedValuesError(self, pylist, dtype=None, ragged_rank=None, inner_shape=None, exception=None, message=None):
        if False:
            while True:
                i = 10
        'Tests that `constant_value()` raises an expected exception.'
        self.assertRaisesRegex(exception, message, ragged_factory_ops.constant_value, pylist, dtype=dtype, ragged_rank=ragged_rank, inner_shape=inner_shape)

def _normalize_pylist(item):
    if False:
        i = 10
        return i + 15
    'Convert all (possibly nested) np.arrays contained in item to list.'
    if not isinstance(item, (list, np.ndarray)):
        return item
    level = (x.tolist() if isinstance(x, np.ndarray) else x for x in item)
    return [_normalize_pylist(el) if isinstance(item, (list, np.ndarray)) else el for el in level]
if __name__ == '__main__':
    googletest.main()