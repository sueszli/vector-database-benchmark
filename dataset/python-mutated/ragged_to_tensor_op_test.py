"""Tests for ragged.to_tensor."""
import random
from absl.testing import parameterized
import numpy as np
from tensorflow.python.client import session
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensor
from tensorflow.python.platform import benchmark
from tensorflow.python.platform import googletest
from tensorflow.python.util import nest

def make_placeholder(t):
    if False:
        for i in range(10):
            print('nop')
    return array_ops.placeholder_with_default(t, None)

def rebuild_ragged_tensor_with_value_rowids(rt, feed_dict=None, sess=None):
    if False:
        print('Hello World!')
    'Returns a copy of `rt`, built using `from_value_rowids`.\n\n  This ensures that RaggedTensor._cached_value_rowids is populated, which\n  triggers a different code-path for converting ragged tensors to tensors.\n\n  If `feed_dict` and `sess` are specified, then build the new `RaggedTensor`\n  using placeholder tensors, and populate a feed dictionary that can be used\n  to feed the placeholders.\n\n  Args:\n    rt: The RaggedTensor to copy.\n    feed_dict: If specified, then build the new `RaggedTensor` using\n      placeholders, and populate this dict with entries to feed those\n      placeholders.\n    sess: A session used to evaluate tensors; required if feed_dict is\n      specified.\n\n  Returns:\n    A copy of `rt`, built using `from_value_rowids`.\n  '
    if isinstance(rt, ragged_tensor.RaggedTensor):
        values = rebuild_ragged_tensor_with_value_rowids(rt.values, feed_dict, sess)
        rowids = rt.value_rowids()
        nrows = rt.nrows()
        if feed_dict is not None:
            rowids_ph = make_placeholder(rowids)
            nrows_ph = make_placeholder(nrows)
            feed_dict[rowids_ph] = sess.run(rowids)
            feed_dict[nrows_ph] = sess.run(nrows)
            (rowids, nrows) = (rowids_ph, nrows_ph)
        return ragged_tensor.RaggedTensor.from_value_rowids(values, rowids, nrows)
    else:
        if feed_dict is not None:
            rt_ph = make_placeholder(rt)
            feed_dict[rt_ph] = sess.run(rt)
            rt = rt_ph
        return rt

@test_util.run_all_in_graph_and_eager_modes
class RaggedTensorToTensorOpTest(test_util.TensorFlowTestCase, parameterized.TestCase):

    def testDocStringExamples(self):
        if False:
            return 10
        'Example from ragged_to_tensor.__doc__.'
        rt = ragged_factory_ops.constant([[9, 8, 7], [], [6, 5], [4]])
        dt = rt.to_tensor()
        self.assertAllEqual(dt, [[9, 8, 7], [0, 0, 0], [6, 5, 0], [4, 0, 0]])

    @parameterized.named_parameters({'testcase_name': 'shape_2xN', 'rt_input': [[0, 1, 2], [], [3]], 'expected': [[0, 1, 2], [0, 0, 0], [3, 0, 0]]}, {'testcase_name': 'shape_2xN_default_0D', 'rt_input': [[0, 1, 2], [], [3]], 'default': 5, 'expected': [[0, 1, 2], [5, 5, 5], [3, 5, 5]]}, {'testcase_name': 'empty_first_row', 'rt_input': [[], [], [3, 4], []], 'expected': [[0, 0], [0, 0], [3, 4], [0, 0]]}, {'testcase_name': 'empty_last_row', 'rt_input': [[0, 1, 2], [], [3], []], 'expected': [[0, 1, 2], [0, 0, 0], [3, 0, 0], [0, 0, 0]]}, {'testcase_name': 'shape_4xN', 'rt_input': [[1, 2, 3], [], [4], [5, 6]], 'expected': [[1, 2, 3], [0, 0, 0], [4, 0, 0], [5, 6, 0]]}, {'testcase_name': 'shape_4xN_default_0D', 'rt_input': [[1, 2, 3], [], [4], [5, 6]], 'default': 9, 'expected': [[1, 2, 3], [9, 9, 9], [4, 9, 9], [5, 6, 9]]}, {'testcase_name': 'shape_2xN_already_dense', 'rt_input': [[6, 7, 8], [9, 10, 11]], 'expected': [[6, 7, 8], [9, 10, 11]]}, {'testcase_name': 'shape_2xN_string_already_dense', 'rt_input': [[b'a', b'b', b'c'], [b'd', b'e', b'antidisestablishmentarianism']], 'ragged_rank': 1, 'expected': [[b'a', b'b', b'c'], [b'd', b'e', b'antidisestablishmentarianism']]}, {'testcase_name': 'shape_4xNxM', 'rt_input': [[[1, 2], [], [3, 4]], [], [[5]], [[6, 7], [8]]], 'expected': [[[1, 2], [0, 0], [3, 4]], [[0, 0], [0, 0], [0, 0]], [[5, 0], [0, 0], [0, 0]], [[6, 7], [8, 0], [0, 0]]]}, {'testcase_name': 'shape_4xNxM_default_0D', 'rt_input': [[[1, 2], [], [3, 4]], [], [[5]], [[6, 7], [8]]], 'default': 9, 'expected': [[[1, 2], [9, 9], [3, 4]], [[9, 9], [9, 9], [9, 9]], [[5, 9], [9, 9], [9, 9]], [[6, 7], [8, 9], [9, 9]]]}, {'testcase_name': 'shape_1xNx1_default_0D', 'rt_input': [[[1], [2], [3]]], 'ragged_rank': 1, 'default': 0, 'expected': [[[1], [2], [3]]]}, {'testcase_name': 'shape_2xNx2_already_dense', 'rt_input': [[[6, 7], [8, 9], [10, 11]], [[12, 13], [14, 15], [16, 17]]], 'ragged_rank': 1, 'expected': [[[6, 7], [8, 9], [10, 11]], [[12, 13], [14, 15], [16, 17]]]}, {'testcase_name': 'shape_2xNx2_already_dense_default_1D', 'rt_input': [[[6, 7], [8, 9], [10, 11]], [[12, 13], [14, 15], [16, 17]]], 'ragged_rank': 1, 'default': [31, 32], 'expected': [[[6, 7], [8, 9], [10, 11]], [[12, 13], [14, 15], [16, 17]]]}, {'testcase_name': 'shape_2xNx2_string_already_dense', 'rt_input': [[[b'a', b'b'], [b'c', b'd'], [b'e', b'f']], [[b'g', b'jalapeno'], [b'kangaroo', b'llama'], [b'manzana', b'nectar']]], 'ragged_rank': 1, 'expected': [[[b'a', b'b'], [b'c', b'd'], [b'e', b'f']], [[b'g', b'jalapeno'], [b'kangaroo', b'llama'], [b'manzana', b'nectar']]]}, {'testcase_name': 'shape_4xNx1_default_1D', 'rt_input': [[[1], [2], [3]], [], [[4]], [[5], [6]]], 'ragged_rank': 1, 'default': [9], 'expected': [[[1], [2], [3]], [[9], [9], [9]], [[4], [9], [9]], [[5], [6], [9]]]}, {'testcase_name': 'shape_2xNx2_default_0D', 'rt_input': [[[6, 7], [8, 9], [10, 11]], [[12, 13], [14, 15]]], 'ragged_rank': 1, 'default': 2, 'expected': [[[6, 7], [8, 9], [10, 11]], [[12, 13], [14, 15], [2, 2]]]}, {'testcase_name': 'shape_2xNx2_default_1D', 'rt_input': [[[6, 7], [8, 9], [10, 11]], [[12, 13], [14, 15]]], 'ragged_rank': 1, 'default': [2, 3], 'expected': [[[6, 7], [8, 9], [10, 11]], [[12, 13], [14, 15], [2, 3]]]}, {'testcase_name': 'shape_1xNxMxK_default_0D', 'rt_input': [[[[1], [2]], [], [[3]]]], 'default': 9, 'expected': [[[[1], [2]], [[9], [9]], [[3], [9]]]]}, {'testcase_name': 'shape_2xNx2x2_default_2x1', 'rt_input': [[[[1, 2], [3, 4]]], []], 'ragged_rank': 1, 'default': [[5], [6]], 'expected': [[[[1, 2], [3, 4]]], [[[5, 5], [6, 6]]]]}, {'testcase_name': 'shape_2xNx2x2_default_1x2', 'rt_input': [[[[1, 2], [3, 4]]], []], 'ragged_rank': 1, 'default': [[5, 6]], 'expected': [[[[1, 2], [3, 4]]], [[[5, 6], [5, 6]]]]}, {'testcase_name': 'shape_4xN_with_crop', 'rt_input': [[0, 1, 2, 3], [], [4], []], 'shape': [2, 3], 'expected': [[0, 1, 2], [0, 0, 0]]}, {'testcase_name': 'shape_2xN_with_pad', 'rt_input': [[1, 2], [3]], 'shape': [3, 3], 'expected': [[1, 2, 0], [3, 0, 0], [0, 0, 0]]}, {'testcase_name': 'shape_4xN_with_crop_and_pad', 'rt_input': [[0, 1, 2, 3], [], [4], []], 'shape': [2, 8], 'expected': [[0, 1, 2, 3, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]]}, {'testcase_name': 'shape_4xN_with_tuple_shape', 'rt_input': [[0, 1, 2, 3], [], [4], []], 'shape': (2, 3), 'expected': [[0, 1, 2], [0, 0, 0]]}, {'testcase_name': 'shape_4xN_with_tensorshape_shape', 'rt_input': [[0, 1, 2, 3], [], [4], []], 'shape': tensor_shape.TensorShape([2, 3]), 'expected': [[0, 1, 2], [0, 0, 0]]}, {'testcase_name': 'shape_4xN_with_partial_shape', 'rt_input': [[0, 1, 2, 3], [], [4], []], 'shape': tensor_shape.TensorShape([2, None]), 'expected': [[0, 1, 2, 3], [0, 0, 0, 0]]}, {'testcase_name': 'shape_0xN', 'rt_input': [], 'ragged_rank': 1, 'expected': [], 'expected_shape': [0, 0]}, {'testcase_name': 'shape_0xNxM', 'rt_input': [], 'ragged_rank': 2, 'expected': [], 'expected_shape': [0, 0, 0]}, {'testcase_name': 'shape_2xN_empty', 'rt_input': [[], []], 'expected': [[], []], 'expected_shape': [2, 0]})
    def testRaggedTensorToTensor(self, rt_input, expected, ragged_rank=None, inner_shape=None, default=None, shape=None, expected_shape=None):
        if False:
            while True:
                i = 10
        rt1 = ragged_factory_ops.constant(rt_input, ragged_rank=ragged_rank, inner_shape=inner_shape)
        rt2 = rebuild_ragged_tensor_with_value_rowids(rt1)
        for rt in [rt1, rt2]:
            for use_placeholder in [False, True]:
                if use_placeholder:
                    if default is not None:
                        default = make_placeholder(default)
                    rt = nest.map_structure(make_placeholder, rt, expand_composites=True)
                dt = rt.to_tensor(default_value=default, shape=shape)
                self.assertIsInstance(dt, tensor_lib.Tensor)
                self.assertEqual(rt.dtype, dt.dtype)
                if shape is not None:
                    self.assertTrue(dt.shape.is_compatible_with(shape))
                else:
                    self.assertTrue(dt.shape.is_compatible_with(rt.shape))
                if expected_shape is not None:
                    expected = np.ndarray(expected_shape, buffer=np.array(expected))
                self.assertAllEqual(dt, expected)

    @parameterized.parameters([{'rt_input': [[1, 2, 3]], 'default': 'a', 'error_type': TypeError, 'error': 'Expected int32|Cannot convert'}, {'rt_input': [[1, 2, 3]], 'default': [0], 'error': 'default_value\\.shape=.* and rt_input\\.flat_values\\.shape=.* are incompatible: default_value\\.rank = 1  must be less than rt_input\\.flat_values\\.rank = 1'}, {'rt_input': [[[1, 2], [3, 4]], [[5, 6]]], 'ragged_rank': 1, 'default': [7, 8, 9], 'error': 'default_value\\.shape.* and rt_input\\.flat_values\\.shape.* are incompatible: default_value\\.shape\\[-1\\] = 3 but rt_input\\.flat_values\\.shape\\[-1\\] = 2'}, {'rt_input': [[1, 2, 3]], 'shape': [3, 3, 3], 'error': 'rt_input\\.shape and shape=\\[.,.,.\\] are incompatible: rt_input\\.rank = 2 but shape\\.rank = 3'}, {'rt_input': [[[1, 2, 3]]], 'ragged_rank': 1, 'shape': [1, 1, 4], 'error': 'rt_input\\.shape and shape=\\[1,1,4\\] are incompatible: rt_input\\.shape\\[2\\] = 3 but shape\\[2\\] = 4'}])
    def testError(self, rt_input, error, error_type=(ValueError, errors.InvalidArgumentError), default=None, ragged_rank=None, shape=None):
        if False:
            print('Hello World!')
        rt = ragged_factory_ops.constant(rt_input, ragged_rank=ragged_rank)
        with self.assertRaisesRegex(error_type, error):
            self.evaluate(rt.to_tensor(default_value=default, shape=shape))
        rt_placeholder = nest.map_structure(make_placeholder, rt, expand_composites=True)
        with self.assertRaisesRegex(error_type, error):
            self.evaluate(rt_placeholder.to_tensor(default_value=default, shape=shape))

    def test_shape_limit_shape_is_tensor(self):
        if False:
            i = 10
            return i + 15
        input_data = ragged_factory_ops.constant([[0, 1, 2, 3], [], [4], []])
        actual = input_data.to_tensor(shape=constant_op.constant([2, 3], dtype=dtypes.int64))
        self.assertAllEqual(actual, [[0, 1, 2], [0, 0, 0]])
        self.assertEqual(actual.shape.as_list(), [2, 3])

    def test_shape_limit_shape_is_tensor_unknown_rank(self):
        if False:
            print('Hello World!')
        input_data = ragged_factory_ops.constant([[0, 1, 2, 3], [], [4], []])
        actual = input_data.to_tensor(shape=constant_op.constant(-1, dtype=dtypes.int64))
        self.assertAllEqual(actual, [[0, 1, 2, 3], [0, 0, 0, 0], [4, 0, 0, 0], [0, 0, 0, 0]])
        self.assertTrue(actual.shape.is_compatible_with([4, 4]))

    def test_shape_limit_shape_is_tensor_unknown_dim(self):
        if False:
            return 10
        input_data = ragged_factory_ops.constant([[0, 1, 2, 3], [], [4], []])
        actual = input_data.to_tensor(shape=constant_op.constant([2, -1], dtype=dtypes.int64))
        self.assertAllEqual(actual, [[0, 1, 2, 3], [0, 0, 0, 0]])
        self.assertTrue(actual.shape.is_compatible_with([2, None]))

    def test_shape_limit_shape_is_tensor_int32(self):
        if False:
            print('Hello World!')
        input_data = ragged_factory_ops.constant([[0, 1, 2, 3], [], [4], []])
        actual = input_data.to_tensor(shape=constant_op.constant([2, 3], dtype=dtypes.int32))
        self.assertAllEqual(actual, [[0, 1, 2], [0, 0, 0]])
        self.assertEqual(actual.shape.as_list(), [2, 3])

    def test_shape_expand_first_dim(self):
        if False:
            while True:
                i = 10
        input_data = ragged_factory_ops.constant([[0, 1, 2], [], [3]])
        actual = input_data.to_tensor(shape=[4, 4])
        self.assertAllEqual(actual, [[0, 1, 2, 0], [0, 0, 0, 0], [3, 0, 0, 0], [0, 0, 0, 0]])
        self.assertEqual(actual.shape.as_list(), [4, 4])

    def test_value_transposed(self):
        if False:
            for i in range(10):
                print('nop')
        my_value = array_ops.transpose(constant_op.constant([[0, 1, 2, 3], [4, 5, 6, 7]]))
        input_data = RaggedTensor.from_value_rowids(values=my_value, value_rowids=constant_op.constant([0, 1, 2, 3], dtype=dtypes.int64), nrows=constant_op.constant(4, dtype=dtypes.int64), validate=True)
        self.assertAllEqual(input_data, [[[0, 4]], [[1, 5]], [[2, 6]], [[3, 7]]])

    def test_broadcast_default(self):
        if False:
            for i in range(10):
                print('nop')
        input_data = ragged_factory_ops.constant([[[[1, 2], [3, 4]]], []], ragged_rank=1)
        default_value = make_placeholder([[5], [6]])
        actual = input_data.to_tensor(default_value=default_value)
        expected = [[[[1, 2], [3, 4]]], [[[5, 5], [6, 6]]]]
        self.assertAllEqual(actual, expected)

    def test_broadcast_default_no_placeholder(self):
        if False:
            i = 10
            return i + 15
        input_data = ragged_factory_ops.constant([[[[1, 2], [3, 4]]], []], ragged_rank=1)
        default_value = constant_op.constant([[5], [6]], shape=None)
        actual = input_data.to_tensor(default_value=default_value)
        expected = [[[[1, 2], [3, 4]]], [[[5, 5], [6, 6]]]]
        self.assertAllEqual(actual, expected)

    def test_shape_expand_second_dim(self):
        if False:
            for i in range(10):
                print('nop')
        input_data = ragged_factory_ops.constant([[0, 1, 2], [], [3], []])
        actual = input_data.to_tensor(shape=[3, 4])
        self.assertAllEqual(actual, [[0, 1, 2, 0], [0, 0, 0, 0], [3, 0, 0, 0]])

    @parameterized.parameters(([2, 3, 4], None, [2, 3, 4]), ([2, 3, 4], [None, None, None], [2, 3, 4]), ([2, 3, 4], [None, 3, None], [2, 3, 4]), ([2, 3, 4], [None, 3, 4], [2, 3, 4]), ([2, 3, 4], [2, 3, 4], [2, 3, 4]))
    def test_preserve_shape_roundtrip(self, input_shape, to_tensor_shape, expected_shape):
        if False:
            return 10
        tensor = array_ops.zeros(input_shape)
        ragged_from_tensor = RaggedTensor.from_tensor(tensor, ragged_rank=2)
        recovered_tensor = ragged_from_tensor.to_tensor(shape=to_tensor_shape)
        self.assertAllEqual(tensor.shape.as_list(), expected_shape)
        self.assertAllEqual(ragged_from_tensor.shape.as_list(), expected_shape)
        self.assertAllEqual(recovered_tensor.shape.as_list(), expected_shape)

    def test_empty_tensor_with_shape(self):
        if False:
            i = 10
            return i + 15
        input_data = RaggedTensor.from_value_rowids(values=constant_op.constant([], dtype=dtypes.int64), value_rowids=constant_op.constant([], dtype=dtypes.int64), nrows=constant_op.constant(2, dtype=dtypes.int64), validate=True)
        actual = input_data.to_tensor(default_value=3, shape=[2, 3])
        self.assertAllEqual(actual, [[3, 3, 3], [3, 3, 3]])

    @parameterized.named_parameters([dict(testcase_name='2d_default_shape', shape=None, rt_value=[[1, 2, 3], [4], [5, 6]], rt_grad=[[9, 8, 7], [6], [3, 2]], default_value=0, default_grad=sum([5, 4, 1]), output_value=[[1, 2, 3], [4, 0, 0], [5, 6, 0]], output_grad=[[9, 8, 7], [6, 5, 4], [3, 2, 1]]), dict(testcase_name='2d_pad', shape=[4, 4], rt_value=[[1, 2, 3], [4], [5, 6]], rt_grad=[[9, 8, 7], [5], [1, 0]], default_value=0, default_grad=sum([6, 4, 3, 2, 1, 2, 3, 4, 5, 6]), output_value=[[1, 2, 3, 0], [4, 0, 0, 0], [5, 6, 0, 0], [0, 0, 0, 0]], output_grad=[[9, 8, 7, 6], [5, 4, 3, 2], [1, 0, 1, 2], [3, 4, 5, 6]]), dict(testcase_name='2d_pad_and_crop', shape=[5, 3], rt_value=[[1, 2, 3], [4], [5, 6, 7, 8, 9], [8]], rt_grad=[[9, 8, 7], [6], [3, 2, 1, 0, 0], [2]], default_value=0, default_grad=sum([5, 4, 3, 4, 5, 6, 7]), output_value=[[1, 2, 3], [4, 0, 0], [5, 6, 7], [8, 0, 0], [0, 0, 0]], output_grad=[[9, 8, 7], [6, 5, 4], [3, 2, 1], [2, 3, 4], [5, 6, 7]]), dict(testcase_name='3d_rrank_2', shape=[2, 2, 2], rt_value=[[[9, 8, 7], [6]], [[5, 4]]], rt_grad=[[[1, 2, 0], [3]], [[5, 6]]], default_value=3, default_grad=sum([4, 7, 8]), output_value=[[[9, 8], [6, 3]], [[5, 4], [3, 3]]], output_grad=[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), dict(testcase_name='3d_rrank_1_with_0d_default', ragged_rank=1, shape=[2, 2, 2], rt_value=[[[9, 8], [7, 6]], [[5, 4]]], rt_grad=[[[1, 2], [3, 4]], [[5, 6]]], default_value=3, default_grad=sum([7, 8]), output_value=[[[9, 8], [7, 6]], [[5, 4], [3, 3]]], output_grad=[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), dict(testcase_name='3d_rrank_1_with_1d_default', ragged_rank=1, shape=[2, 2, 2], rt_value=[[[9, 8], [7, 6]], [[5, 4]]], rt_grad=[[[1, 2], [3, 4]], [[5, 6]]], default_value=[3, 2], default_grad=[7, 8], output_value=[[[9, 8], [7, 6]], [[5, 4], [3, 2]]], output_grad=[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), dict(testcase_name='3d_rrank_1_with_1d_broadcast_default', ragged_rank=1, shape=[2, 2, 2], rt_value=[[[9, 8], [7, 6]], [[5, 4]]], rt_grad=[[[1, 2], [3, 4]], [[5, 6]]], default_value=[3], default_grad=[7 + 8], output_value=[[[9, 8], [7, 6]], [[5, 4], [3, 3]]], output_grad=[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), dict(testcase_name='4d_rrank_1_with_2d_default', ragged_rank=1, shape=[3, 3, 2, 1], rt_value=[[[[9], [8]], [[7], [6]]], [[[5], [4]]]], rt_grad=[[[[1], [2]], [[3], [4]]], [[[7], [8]]]], default_value=[[3], [2]], default_grad=[[5 + 9 + 2 + 4 + 6 + 8], [6 + 1 + 3 + 5 + 7 + 9]], output_value=[[[[9], [8]], [[7], [6]], [[3], [2]]], [[[5], [4]], [[3], [2]], [[3], [2]]], [[[3], [2]], [[3], [2]], [[3], [2]]]], output_grad=[[[[1], [2]], [[3], [4]], [[5], [6]]], [[[7], [8]], [[9], [1]], [[2], [3]]], [[[4], [5]], [[6], [7]], [[8], [9]]]]), dict(testcase_name='4d_rrank_1_with_with_0d_default', ragged_rank=1, shape=[3, 3, 2, 1], rt_value=[[[[9], [8]], [[7], [6]]], [[[5], [4]]]], rt_grad=[[[[1], [2]], [[3], [4]]], [[[7], [8]]]], default_value=3, default_grad=5 + 9 + 2 + 4 + 6 + 8 + 6 + 1 + 3 + 5 + 7 + 9, output_value=[[[[9], [8]], [[7], [6]], [[3], [3]]], [[[5], [4]], [[3], [3]], [[3], [3]]], [[[3], [3]], [[3], [3]], [[3], [3]]]], output_grad=[[[[1], [2]], [[3], [4]], [[5], [6]]], [[[7], [8]], [[9], [1]], [[2], [3]]], [[[4], [5]], [[6], [7]], [[8], [9]]]]), dict(testcase_name='zero_size', shape=[0, 0], rt_value=[[9, 8], [7, 6, 5], [4]], rt_grad=[[0, 0], [0, 0, 0], [0]], default_value=3, default_grad=0, output_value=[], output_grad=[])])
    def test_gradient(self, shape, rt_value, rt_grad, default_value, default_grad, output_value, output_grad, ragged_rank=None):
        if False:
            for i in range(10):
                print('nop')
        'Tests that ragged_to_dense generates the right gradient.\n\n    Args:\n      shape: The `shape` arg for `ragged_to_dense`.\n      rt_value: The `rt_input` arg for `ragged_to_dense`.\n      rt_grad: The expected gradient for `rt_value`.  Corresponds 1:1 with\n        `rt_value`.\n      default_value: The `default_value` arg for `ragged_to_dense`.\n      default_grad: The expected gradient for `default_value`.  Corresponds 1:1\n        with `default_value`.\n      output_value: The expected output of `ragged_to_dense`.\n      output_grad: The gradient for the output (used to generate the gradients\n        `rt_grad` and `default_grad`).  Corresponds 1:1 with `output_value`.\n      ragged_rank: Ragged rank for `rt_value`.\n    '
        rt_value = ragged_factory_ops.constant(rt_value, dtype=dtypes.float32, ragged_rank=ragged_rank)
        rt_grad = ragged_factory_ops.constant(rt_grad, dtype=dtypes.float32, ragged_rank=ragged_rank)
        default_value = constant_op.constant(default_value, dtype=dtypes.float32)
        default_grad = constant_op.constant(default_grad, dtype=dtypes.float32)
        output_value = constant_op.constant(output_value, dtype=dtypes.float32, shape=shape)
        output_grad = constant_op.constant(output_grad, dtype=dtypes.float32, shape=shape)
        shape = tensor_shape.as_shape(shape)
        for partition_type in ['row_splits', 'value_rowids']:
            rt_val = self.rt_with_partition_type(rt_value, partition_type)
            if context.executing_eagerly():
                self._test_gradient_helper(rt_val, default_value, shape, output_grad, output_value, rt_grad, default_grad)
            else:
                for shape_info in ['known', 'unknown_dims', 'unknown_rank']:
                    rt_val = self.wrap_in_placeholder(rt_val, shape_info)
                    default_val = self.wrap_in_placeholder(default_value, shape_info)
                    shape_val = self.wrap_in_placeholder(shape, shape_info)
                    self._test_gradient_helper(rt_val, default_val, shape_val, output_grad, output_value, rt_grad, default_grad)

    def _test_gradient_helper(self, rt_val, default_val, shape_val, output_grad, expected_output_val, expected_rt_grad, expected_default_grad):
        if False:
            return 10
        if context.executing_eagerly():
            with backprop.GradientTape() as tape:
                tape.watch([rt_val, default_val])
                out = rt_val.to_tensor(default_val, shape=shape_val)
                (actual_rt_grad, actual_default_grad) = tape.gradient(out, (rt_val, default_val), output_gradients=output_grad)
        else:
            out = rt_val.to_tensor(default_val, shape=shape_val)
            (actual_rt_grad, actual_default_grad) = gradients_impl.gradients(ys=out, xs=(rt_val, default_val), grad_ys=output_grad)
        self.assertAllClose(out, expected_output_val)
        self.assertIsInstance(actual_rt_grad, RaggedTensor)
        self.assertAllClose(actual_rt_grad, expected_rt_grad)
        self.assertAllClose(actual_default_grad, expected_default_grad)

    def rt_with_partition_type(self, rt, partition_type):
        if False:
            return 10
        if isinstance(rt, tensor_lib.Tensor):
            return rt
        if partition_type == 'row_splits':
            return rt
        if partition_type == 'value_rowids':
            return ragged_tensor.RaggedTensor.from_value_rowids(self.rt_with_partition_type(rt.values, partition_type), rt.value_rowids(), rt.nrows())
        raise AssertionError('Unexpected partition_type %r' % partition_type)

    def wrap_in_placeholder(self, arg, shape_info):
        if False:
            while True:
                i = 10
        "Wraps `arg` in a placeholder to limit static shape info.\n\n    Args:\n      arg: The value to wrap.  A Tensor, RaggedTensor, or TensorShape.\n      shape_info: One of ['known', 'unknown_dims', 'unknown_rank'].\n\n    Returns:\n      * If shape_info is 'known': returns `arg`.\n      * If shape_info is 'unknown_dims': returns a placeholder wrapping `arg`\n        where the dimension sizes are unknown.  If `arg` is a TensorShape,\n        then convert it to a vector first.  If `arg` is a RaggedTensor, then\n        wrap the flat_values.\n      * If shape_info is 'unknown_rank': returns a placeholder wrapping `arg`\n        where the rank is unknown.  If `arg` is a TensorShape, then convert it\n        to a vector first.  If `arg` is a RaggedTensor, then wrap the\n        flat_values.\n    "
        if shape_info == 'known':
            return arg
        if isinstance(arg, ragged_tensor.RaggedTensor):
            return arg.with_flat_values(self.wrap_in_placeholder(arg.flat_values, shape_info))
        if isinstance(arg, tensor_shape.TensorShape):
            if arg.ndims is None:
                return arg
            arg = constant_op.constant(arg.as_list())
        if shape_info == 'unknown_rank':
            return array_ops.placeholder_with_default(arg, None)
        if shape_info == 'unknown_dims':
            return array_ops.placeholder_with_default(arg, [None] * arg.shape.rank)
        raise AssertionError('Unexpected shape_info %r' % shape_info)

    def test_shape_is_list_including_tensor_element(self):
        if False:
            print('Hello World!')
        rt = ragged_factory_ops.constant([[1, 2, 3], [4], [5, 6]])
        result = rt.to_tensor(shape=[2, constant_op.constant(2)])
        self.assertAllEqual(result, [[1, 2], [4, 0]])

class RaggedToDenseBenchmark(googletest.Benchmark):
    CONFIGS = [{'shape': [10, 10]}, {'shape': [10, 1000]}, {'shape': [1000, 10]}, {'shape': [1000, 10], 'fill': [1, 0.95]}, {'shape': [1000, 10], 'fill': [1, 0.05]}, {'shape': [1000, 10], 'dtype': dtypes.string}, {'shape': [1000, 10], 'dtype': dtypes.int64}, {'shape': [100, 100]}, {'shape': [50, 50, 32]}, {'shape': [100, 100, 100], 'min_iters': 100}, {'shape': [1000, 1000], 'min_iters': 100}, {'shape': [10, 10, 10, 10, 10]}, {'shape': [10, 10, 10, 10, 10], 'ragged_rank': 1}, {'shape': [10, 10, 10, 10, 10], 'ragged_rank': 2}, {'shape': [50, 50, 32], 'ragged_rank': 1, 'default_shape': [32]}, {'shape': [200, 50, 32], 'ragged_rank': 1, 'default_shape': [32]}]

    def run_benchmark(self, shape=(100, 100), ragged_rank=None, dtype=dtypes.float32, fill=None, default_shape=(), output_shape=None, min_iters=1000):
        if False:
            while True:
                i = 10
        'Run a benchmark with the specified configuration parameters.\n\n    Args:\n      shape: Bounding box for the input ragged tensor.\n      ragged_rank: Ragged rank for the input ragged tensor.  Defaults to\n        `len(shape)-1`.\n      dtype: Data type for the input ragged tensor.\n      fill: How full each dimension should be (0-1).  Corresponds 1:1 with\n        `shape`.  Defaults to 0.8 for each dimension.\n      default_shape: Shape for the default (padding) value.\n      output_shape: Output shape -- ragged tensor will be padded or cropped to\n        this shape.\n      min_iters: Minimum iterations for benchmark.\n    '
        if ragged_rank is None:
            ragged_rank = len(shape) - 1
        if fill is None:
            fill = [0.8 for _ in shape]
        rt_input = self._generateRaggedTensor(shape, ragged_rank, dtype, fill)
        default_value = constant_op.constant(self._generateRaggedTensor(default_shape, 0, dtype), dtype=dtype)
        mbs = np.prod(shape) / 2 ** 20
        with session.Session(config=benchmark.benchmark_config()) as sess:
            extras = {'shape': shape, 'ragged_rank': ragged_rank, 'dtype': dtype, 'fill': fill, 'default_shape': default_shape}
            rt = ragged_factory_ops.constant(rt_input, dtype, ragged_rank=ragged_rank)
            splits_rt_placeholder = ragged_factory_ops.placeholder(dtype, ragged_rank, shape[ragged_rank + 1:])
            splits_feed_dict = {splits_rt_placeholder: sess.run(rt)}
            rowids_feed_dict = {}
            rowids_rt_placeholder = rebuild_ragged_tensor_with_value_rowids(rt, rowids_feed_dict, sess)
            run_op_benchmark_kwargs = dict(sess=sess, store_memory_usage=True, min_iters=min_iters, burn_iters=max(5, min_iters // 10), mbs=mbs, extras=extras)
            ragged_to_tensor_with_splits = splits_rt_placeholder.to_tensor(default_value=default_value)
            self.run_op_benchmark(op_or_tensor=ragged_to_tensor_with_splits.op, name='ragged_to_tensor_with_splits', feed_dict=splits_feed_dict, **run_op_benchmark_kwargs)
            ragged_to_tensor_with_rowids = rowids_rt_placeholder.to_tensor(default_value=default_value)
            self.run_op_benchmark(op_or_tensor=ragged_to_tensor_with_rowids.op, name='ragged_to_tensor_with_rowids', feed_dict=rowids_feed_dict, **run_op_benchmark_kwargs)

    def _generateRaggedTensor(self, shape, ragged_rank, dtype, fill=None, axis=0):
        if False:
            while True:
                i = 10
        if axis == len(shape):
            value = random.random()
            if dtype == dtypes.string:
                value = str(value)
            if dtype.is_integer:
                value = int(value * 1000)
            return value
        if axis == 0 or axis > ragged_rank:
            slice_size = shape[axis]
        else:
            slice_size = (np.random.geometric(fill[axis], shape[axis]) == 1).sum()
        return [self._generateRaggedTensor(shape, ragged_rank, dtype, fill, axis + 1) for _ in range(slice_size)]

    def benchmark_ragged_to_dense(self):
        if False:
            return 10
        random.seed(5)
        for config in self.CONFIGS:
            self.run_benchmark(**config)
if __name__ == '__main__':
    googletest.main()