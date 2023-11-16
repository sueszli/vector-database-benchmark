"""Tests for ragged.squeeze."""
from absl.testing import parameterized
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.ragged import ragged_conversion_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_squeeze_op
from tensorflow.python.platform import googletest

@test_util.run_all_in_graph_and_eager_modes
class RaggedSqueezeTest(test_util.TensorFlowTestCase, parameterized.TestCase):

    @parameterized.parameters([{'input_list': []}, {'input_list': [[]], 'squeeze_ranks': [0]}, {'input_list': [[[[], []], [[], []]]], 'squeeze_ranks': [0]}])
    def test_passing_empty(self, input_list, squeeze_ranks=None):
        if False:
            return 10
        rt = ragged_squeeze_op.squeeze(ragged_factory_ops.constant(input_list), squeeze_ranks)
        dt = array_ops.squeeze(constant_op.constant(input_list), squeeze_ranks)
        self.assertAllEqual(ragged_conversion_ops.to_tensor(rt), dt)

    @parameterized.parameters([{'input_list': [[1]], 'squeeze_ranks': [0]}, {'input_list': [[1]], 'squeeze_ranks': [0, 1]}, {'input_list': [[1, 2]], 'squeeze_ranks': [0]}, {'input_list': [[1], [2]], 'squeeze_ranks': [1]}, {'input_list': [[[[12], [11]]]], 'squeeze_ranks': [0]}, {'input_list': [[[[12], [11]]]], 'squeeze_ranks': [1]}, {'input_list': [[[[12], [11]]]], 'squeeze_ranks': [3]}, {'input_list': [[[[12], [11]]]], 'squeeze_ranks': [0, 3]}, {'input_list': [[[[12], [11]]]], 'squeeze_ranks': [0, 1]}, {'input_list': [[[[12], [11]]]], 'squeeze_ranks': [1, 3]}, {'input_list': [[[[12], [11]]]], 'squeeze_ranks': [0, 1, 3]}, {'input_list': [[[1], [2]], [[3], [4]]], 'squeeze_ranks': [2]}, {'input_list': [[1], [2]], 'squeeze_ranks': [-1]}])
    def test_passing_simple(self, input_list, squeeze_ranks=None):
        if False:
            return 10
        rt = ragged_squeeze_op.squeeze(ragged_factory_ops.constant(input_list), squeeze_ranks)
        dt = array_ops.squeeze(constant_op.constant(input_list), squeeze_ranks)
        self.assertAllEqual(ragged_conversion_ops.to_tensor(rt), dt)

    @parameterized.parameters([{'input_list': [[1]], 'squeeze_ranks': [0]}, {'input_list': [[1, 2]], 'squeeze_ranks': [0]}, {'input_list': [[1], [2]], 'squeeze_ranks': [1]}, {'input_list': [[[[12], [11]]]], 'squeeze_ranks': [0]}, {'input_list': [[[[12], [11]]]], 'squeeze_ranks': [1]}, {'input_list': [[[[12], [11]]]], 'squeeze_ranks': [3]}, {'input_list': [[[[12], [11]]]], 'squeeze_ranks': [0, 3]}, {'input_list': [[[[12], [11]]]], 'squeeze_ranks': [0, 1]}, {'input_list': [[[[12], [11]]]], 'squeeze_ranks': [1, 3]}, {'input_list': [[[[12], [11]]]], 'squeeze_ranks': [0, 1, 3]}, {'input_list': [[[1], [2]], [[3], [4]]], 'squeeze_ranks': [2]}])
    def test_passing_simple_from_dense(self, input_list, squeeze_ranks=None):
        if False:
            while True:
                i = 10
        dt = constant_op.constant(input_list)
        rt = ragged_conversion_ops.from_tensor(dt)
        rt_s = ragged_squeeze_op.squeeze(rt, squeeze_ranks)
        dt_s = array_ops.squeeze(dt, squeeze_ranks)
        self.assertAllEqual(ragged_conversion_ops.to_tensor(rt_s), dt_s)

    @parameterized.parameters([{'input_list': [[[[[[1]], [[1, 2]]]], [[[[]], [[]]]]]], 'output_list': [[[1], [1, 2]], [[], []]], 'squeeze_ranks': [0, 2, 4]}, {'input_list': [[[[[[1]], [[1, 2]]]], [[[[]], [[]]]]]], 'output_list': [[[[[1]], [[1, 2]]]], [[[[]], [[]]]]], 'squeeze_ranks': [0]}])
    def test_passing_ragged(self, input_list, output_list, squeeze_ranks=None):
        if False:
            for i in range(10):
                print('nop')
        rt = ragged_factory_ops.constant(input_list)
        rt_s = ragged_squeeze_op.squeeze(rt, squeeze_ranks)
        ref = ragged_factory_ops.constant(output_list)
        self.assertAllEqual(rt_s, ref)

    def test_passing_text(self):
        if False:
            print('Hello World!')
        rt = ragged_factory_ops.constant([[[[[[[['H']], [['e']], [['l']], [['l']], [['o']]], [[['W']], [['o']], [['r']], [['l']], [['d']], [['!']]]]], [[[[['T']], [['h']], [['i']], [['s']]], [[['i']], [['s']]], [[['M']], [['e']], [['h']], [['r']], [['d']], [['a']], [['d']]], [[['.']]]]]]]])
        output_list = [[['H', 'e', 'l', 'l', 'o'], ['W', 'o', 'r', 'l', 'd', '!']], [['T', 'h', 'i', 's'], ['i', 's'], ['M', 'e', 'h', 'r', 'd', 'a', 'd'], ['.']]]
        ref = ragged_factory_ops.constant(output_list)
        rt_s = ragged_squeeze_op.squeeze(rt, [0, 1, 3, 6, 7])
        self.assertAllEqual(rt_s, ref)

    @parameterized.parameters([{'input_list': [[]], 'squeeze_ranks': [1]}, {'input_list': [[1, 2]], 'squeeze_ranks': [1]}, {'input_list': [[1], [2]], 'squeeze_ranks': [0]}, {'input_list': [[[[12], [11]]]], 'squeeze_ranks': [0, 2]}, {'input_list': [[[[12], [11]]]], 'squeeze_ranks': [2]}, {'input_list': [[[1], [2]], [[3], [4]]], 'squeeze_ranks': [0]}, {'input_list': [[[1], [2]], [[3], [4]]], 'squeeze_ranks': [1]}, {'input_list': [[], []], 'squeeze_ranks': [1]}, {'input_list': [[[], []], [[], []]], 'squeeze_ranks': [1]}])
    def test_failing_InvalidArgumentError(self, input_list, squeeze_ranks):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(errors.InvalidArgumentError):
            self.evaluate(ragged_squeeze_op.squeeze(ragged_factory_ops.constant(input_list), squeeze_ranks))

    @parameterized.parameters([{'input_list': [[]]}, {'input_list': [[1]]}, {'input_list': [[1, 2]]}, {'input_list': [[[1], [2]], [[3], [4]]]}, {'input_list': [[1]]}, {'input_list': [[[1], [2]], [[3], [4]]]}, {'input_list': [[[[12], [11]]]]}])
    def test_failing_no_squeeze_dim_specified(self, input_list):
        if False:
            while True:
                i = 10
        with self.assertRaises(ValueError):
            ragged_squeeze_op.squeeze(ragged_factory_ops.constant(input_list))

    @parameterized.parameters([{'input_list': [[[[12], [11]]]], 'squeeze_ranks': [0, 1, 3]}])
    def test_failing_axis_is_not_a_list(self, input_list, squeeze_ranks):
        if False:
            print('Hello World!')
        with self.assertRaises(TypeError):
            tensor_ranks = constant_op.constant(squeeze_ranks)
            ragged_squeeze_op.squeeze(ragged_factory_ops.constant(input_list), tensor_ranks)
if __name__ == '__main__':
    googletest.main()