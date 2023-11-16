"""Tests for ragged_factory_ops.constant."""
from absl.testing import parameterized
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import ragged
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import googletest

@test_util.run_all_in_graph_and_eager_modes
class RaggedConstOpTest(test_util.TensorFlowTestCase, parameterized.TestCase):

    @parameterized.parameters(dict(pylist=b'x', expected_shape=()), dict(pylist=[1, 2, 3], expected_shape=(3,)), dict(pylist=[[1, 2, 3], [4], [5, 6]], expected_shape=(3, None)), dict(pylist=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], expected_shape=(3, None)), dict(pylist=[[[1, 2], [3, 4]], [], [[5, 6], [7, 8], [9, 0]]], expected_shape=(3, None, None)), dict(pylist=[[[1, 2], [3, 4]], [], [[5, 6], [7, 8], [9, 0]]], ragged_rank=1, expected_shape=(3, None, 2)), dict(pylist=[[[1, 2], [3, 4]], [], [[5, 6], [7, 8], [9, 0]]], inner_shape=(2,), expected_shape=(3, None, 2)), dict(pylist=[[[1, 2], [3, 4]], [], [[5, 6], [7, 8], [9, 0]]], ragged_rank=1, inner_shape=(2,), expected_shape=(3, None, 2)), dict(pylist=[[[1, 2], np.array([3, np.array(4)])], np.array([]), [[5, 6], [7, 8], [9, 0]]], expected_shape=(3, None, None)), dict(pylist=[[[1, 2], np.array([3, np.array(4)])], np.array([]), [[5, 6], [7, 8], [9, 0]]], ragged_rank=1, expected_shape=(3, None, 2)), dict(pylist=[[[1, 2], np.array([3, np.array(4)])], np.array([]), [[5, 6], [7, 8], [9, 0]]], inner_shape=(2,), expected_shape=(3, None, 2)), dict(pylist=[[[1, 2], np.array([3, np.array(4)])], np.array([]), [[5, 6], [7, 8], [9, 0]]], ragged_rank=1, inner_shape=(2,), expected_shape=(3, None, 2)), dict(pylist=[[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[2, 4], [6, 8]], [[1, 5], [7, 9]]]], expected_shape=(2, None, None, None)), dict(pylist=[[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[2, 4], [6, 8]], [[1, 5], [7, 9]]]], ragged_rank=1, expected_shape=(2, None, 2, 2)), dict(pylist=[[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[2, 4], [6, 8]], [[1, 5], [7, 9]]]], inner_shape=(2,), expected_shape=(2, None, None, 2)), dict(pylist=[[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[2, 4], [6, 8]], [[1, 5], [7, 9]]]], inner_shape=(2, 2), expected_shape=(2, None, 2, 2)), dict(pylist=np.array([[[np.array([1, 2]), [3, 4]], [[5, 6], [7, 8]]], np.array([[[2, 4], [6, 8]], [[1, 5], [7, 9]]])]), expected_shape=(2, None, None, None)), dict(pylist=[], expected_shape=(0,)), dict(pylist=[[], [], np.array([])], expected_shape=(3, None)), dict(pylist=[[[], []], [], [[], [[]]]], expected_shape=(3, None, None, None)), dict(pylist=np.array([np.array([[], []]), np.array([]), [[], [[]]]], dtype=object), expected_shape=(3, None, None, None)), dict(pylist=[], ragged_rank=1, expected_shape=(0, None)), dict(pylist=[], ragged_rank=2, expected_shape=(0, None, None)), dict(pylist=[], inner_shape=(0, 100, 20), expected_shape=(0, 100, 20)), dict(pylist=[], ragged_rank=1, inner_shape=(100, 20), expected_shape=(0, None, 100, 20)), dict(pylist=[], ragged_rank=2, inner_shape=(100, 20), expected_shape=(0, None, None, 100, 20)), dict(pylist=[[], [], []], ragged_rank=2, expected_shape=(3, None, None)), dict(pylist=[], inner_shape=(0,), expected_shape=(0,)), dict(pylist=[[]], inner_shape=(1, 0), expected_shape=(1, 0)), dict(pylist=np.array([]), ragged_rank=1, inner_shape=(100, 20), expected_shape=(0, None, 100, 20)), dict(pylist=[], expected_dtype=dtypes.float32), dict(pylist=[[[], [[[]], []]]], expected_dtype=dtypes.float32), dict(pylist=[[1, 2], [3], [4, 5, 6]], expected_dtype=dtypes.int32), dict(pylist=[[1.0, 2.0], [], [4.0, 5.0, 6.0]], expected_dtype=dtypes.float32), dict(pylist=[[1, 2], [3.0], [4, 5, 6]], expected_dtype=dtypes.float32), dict(pylist=[[b'a', b'b'], [b'c']], expected_dtype=dtypes.string), dict(pylist=[[True]], expected_dtype=dtypes.bool), dict(pylist=[np.array([1, 2]), np.array([3.0]), [4, 5, 6]], expected_dtype=dtypes.float32), dict(pylist=[], dtype=dtypes.float32), dict(pylist=[], dtype=dtypes.string), dict(pylist=[[1, 2], [3], [4, 5, 6]], dtype=dtypes.int64), dict(pylist=[[1, 2], [3], [4, 5, 6]], dtype=dtypes.int32), dict(pylist=[[1, 2], [3], [4, 5, 6]], dtype=dtypes.float32), dict(pylist=[[1.0, 2.0], [3.0], [4.0, 5.0, 6.0]], dtype=dtypes.float16), dict(pylist=[[1.0, 2.0], [3.0], [4.0, 5.0, 6.0]], dtype=dtypes.float32), dict(pylist=[[b'a', b'b'], [b'c'], [b'd', b'e', b'f']], dtype=dtypes.string))
    def testRaggedConst(self, pylist, dtype=None, ragged_rank=None, inner_shape=None, expected_shape=None, expected_dtype=None):
        if False:
            return 10
        'Tests that `ragged_const(pylist).eval().tolist() == pylist`.\n\n    Args:\n      pylist: The `pylist` argument for `ragged_const()`.\n      dtype: The `dtype` argument for `ragged_const()`.  If not None, then also\n        test that the resulting ragged tensor has this `dtype`.\n      ragged_rank: The `ragged_rank` argument for `ragged_const()`.  If not\n        None, then also test that the resulting ragged tensor has this\n        `ragged_rank`.\n      inner_shape: The `inner_shape` argument for `ragged_const()`.  If not\n        None, then also test that the resulting ragged tensor has this\n        `inner_shape`.\n      expected_shape: The expected shape for the resulting ragged tensor.\n      expected_dtype: The expected dtype for the resulting ragged tensor (used\n        to test default/inferred types when dtype=None).\n    '
        rt = ragged_factory_ops.constant(pylist, dtype=dtype, ragged_rank=ragged_rank, inner_shape=inner_shape)
        pylist = _normalize_pylist(pylist)
        if dtype is not None:
            self.assertEqual(rt.dtype, dtype)
        if expected_dtype is not None:
            self.assertEqual(rt.dtype, expected_dtype)
        if ragged_rank is not None:
            if isinstance(rt, ragged_tensor.RaggedTensor):
                self.assertEqual(rt.ragged_rank, ragged_rank)
            else:
                self.assertEqual(0, ragged_rank)
        if inner_shape is not None:
            if isinstance(rt, ragged_tensor.RaggedTensor):
                self.assertEqual(rt.flat_values.shape.as_list()[1:], list(inner_shape))
            else:
                self.assertEqual(rt.shape.as_list(), list(inner_shape))
        if expected_shape is not None:
            self.assertEqual(tuple(rt.shape.as_list()), expected_shape)
            if expected_shape and expected_shape[0] == 0 and (None not in expected_shape):
                pylist = np.zeros(expected_shape, rt.dtype.as_numpy_dtype)
        self.assertAllEqual(rt, pylist)

    @parameterized.parameters(dict(pylist=12, ragged_rank=1, exception=ValueError, message='Invalid pylist=12: incompatible with ragged_rank=1'), dict(pylist=12, inner_shape=(1,), exception=ValueError, message='Invalid pylist=12: incompatible with dim\\(inner_shape\\)=1'), dict(pylist=[[[1], [2]]], ragged_rank=-1, exception=ValueError, message='Invalid ragged_rank=-1: must be nonnegative'), dict(pylist=[[1, [2]]], exception=ValueError, message='all scalar values must have the same nesting depth'), dict(pylist=[[[1]], [[[2]]]], exception=ValueError, message='all scalar values must have the same nesting depth'), dict(pylist=[[1], [[]]], exception=ValueError, message='Invalid pylist=.*: empty list nesting is greater than scalar value nesting'), dict(pylist=[1, 2, 3], ragged_rank=1, exception=ValueError, message='pylist has scalar values depth 1, but ragged_rank=1 requires scalar value depth greater than 1'), dict(pylist=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], ragged_rank=2, exception=ValueError, message='pylist has scalar values depth 2, but ragged_rank=2 requires scalar value depth greater than 2'), dict(pylist=[1, 2, 3], inner_shape=(1, 1), exception=TypeError), dict(pylist=[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], inner_shape=(2, 2), ragged_rank=1, exception=ValueError, message='Invalid pylist=.*: incompatible with ragged_rank=1 and dim\\(inner_shape\\)=2'), dict(pylist=[[[1, 2], [3, 4]], [[5, 6], [7, 8, 9]]], ragged_rank=1, exception=ValueError, message='inner values have inconsistent shape'), dict(pylist=[[[], [[]]]], ragged_rank=1, exception=ValueError, message='inner values have inconsistent shape'))
    def testRaggedConstError(self, pylist, dtype=None, ragged_rank=None, inner_shape=None, exception=None, message=None):
        if False:
            return 10
        'Tests that `ragged_const()` raises an expected exception.'
        self.assertRaisesRegex(exception, message, ragged_factory_ops.constant, pylist, dtype=dtype, ragged_rank=ragged_rank, inner_shape=inner_shape)

    @parameterized.parameters([dict(pylist=9, scalar_depth=0, max_depth=0), dict(pylist=[9], scalar_depth=1, max_depth=1), dict(pylist=[1, 2, 3], scalar_depth=1, max_depth=1), dict(pylist=[[1], [2]], scalar_depth=2, max_depth=2), dict(pylist=[[[1], [2]], [[3]]], scalar_depth=3, max_depth=3), dict(pylist=[], scalar_depth=None, max_depth=1), dict(pylist=[[]], scalar_depth=None, max_depth=2), dict(pylist=[[], [], []], scalar_depth=None, max_depth=2), dict(pylist=[[[], []], [[], [[[]]]], []], scalar_depth=None, max_depth=5), dict(pylist=[1, [2]], exception=ValueError, message='all scalar values must have the same nesting depth'), dict(pylist=[[1], 2], exception=ValueError, message='all scalar values must have the same nesting depth'), dict(pylist=[[[[1]], []], [[2]]], exception=ValueError, message='all scalar values must have the same nesting depth')])
    def testScalarAndMaxDepthHelper(self, pylist, scalar_depth=None, max_depth=None, exception=None, message=None):
        if False:
            while True:
                i = 10
        'Tests for the _find_scalar_and_max_depth helper function.'
        if exception is not None:
            self.assertRaisesRegex(exception, message, ragged_factory_ops._find_scalar_and_max_depth, pylist)
        else:
            self.assertEqual(ragged_factory_ops._find_scalar_and_max_depth(pylist), (scalar_depth, max_depth))

    @parameterized.parameters([dict(pylist=[[1], [2, 3]], ragged_rank=1, inner_shape=()), dict(pylist=[[[1], [2]], [[3], [4], [5]]], ragged_rank=1, inner_shape=(1,)), dict(pylist=[[[1], [2]], [[3], [4], [5]]], ragged_rank=2, inner_shape=()), dict(pylist=[[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [2, 4, 6]]]], ragged_rank=1, inner_shape=(2, 3)), dict(pylist=[[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [2, 4, 6]]]], ragged_rank=2, inner_shape=(3,)), dict(pylist=[[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [2, 4, 6]]]], ragged_rank=3, inner_shape=()), dict(pylist=[[[1], [2, 3]]], ragged_rank=1, exception=ValueError, message='inner values have inconsistent shape'), dict(pylist=[[[1], [[2]]]], ragged_rank=1, exception=ValueError, message='inner values have inconsistent shape'), dict(pylist=[[[[1]], [2]]], ragged_rank=1, exception=ValueError, message='inner values have inconsistent shape')])
    def testDefaultInnerShapeForPylistHelper(self, pylist, ragged_rank, inner_shape=None, exception=None, message=None):
        if False:
            while True:
                i = 10
        'Tests for the _default_inner_shape_for_pylist helper function.'
        if exception is not None:
            self.assertRaisesRegex(exception, message, ragged.ragged_factory_ops._default_inner_shape_for_pylist, pylist, ragged_rank)
        else:
            self.assertEqual(ragged.ragged_factory_ops._default_inner_shape_for_pylist(pylist, ragged_rank), inner_shape)

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