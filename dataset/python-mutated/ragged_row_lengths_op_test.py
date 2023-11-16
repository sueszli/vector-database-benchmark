"""Tests for ragged.row_lengths."""
from absl.testing import parameterized
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import googletest

@test_util.run_all_in_graph_and_eager_modes
class RaggedRowLengthsOp(test_util.TensorFlowTestCase, parameterized.TestCase):

    @parameterized.parameters([dict(rt_input=[[[3, 1, 4], [1]], [], [[5, 9], [2]], [[6]], []], expected=[2, 0, 2, 1, 0]), dict(rt_input=[[[3, 1, 4], [1]], [], [[5, 9], [2]], [[6]], []], axis=2, expected=[[3, 1], [], [2, 1], [1], []]), dict(rt_input=[['a'], ['b', 'c', 'd'], ['e'], [], ['f']], expected=[1, 3, 1, 0, 1]), dict(rt_input=[['a'], ['b', 'c', 'd'], ['e'], [], ['f']], axis=0, expected=5), dict(rt_input=[['a', 'b', 'c', 'd', 'e', 'f', 'g']], expected=[7]), dict(rt_input=[[], ['a', 'b', 'c', 'd', 'e', 'f', 'g'], []], expected=[0, 7, 0]), dict(rt_input=[], ragged_rank=1, expected=[]), dict(rt_input=[], ragged_rank=1, axis=0, expected=0), dict(rt_input=[[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10]]], ragged_rank=1, axis=0, expected=2), dict(rt_input=[[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10]]], ragged_rank=1, axis=1, expected=[3, 2]), dict(rt_input=[[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10]]], ragged_rank=1, axis=2, expected=[[2, 2, 2], [2, 2]], expected_ragged_rank=1), dict(rt_input=[[[1, 2], [3, 4, 5], []], [[6, 7, 8, 9], [10]]], axis=0, expected=2), dict(rt_input=[[[1, 2], [3, 4, 5], []], [[6, 7, 8, 9], [10]]], axis=-3, expected=2), dict(rt_input=[[[1, 2], [3, 4, 5], []], [[6, 7, 8, 9], [10]]], axis=1, expected=[3, 2]), dict(rt_input=[[[1, 2], [3, 4, 5], []], [[6, 7, 8, 9], [10]]], axis=-2, expected=[3, 2]), dict(rt_input=[[[1, 2], [3, 4, 5], []], [[6, 7, 8, 9], [10]]], axis=2, expected=[[2, 3, 0], [4, 1]], expected_ragged_rank=1), dict(rt_input=[[[1, 2], [3, 4, 5], []], [[6, 7, 8, 9], [10]]], axis=-1, expected=[[2, 3, 0], [4, 1]], expected_ragged_rank=1)])
    def testRowLengths(self, rt_input, expected, axis=1, ragged_rank=None, expected_ragged_rank=None):
        if False:
            return 10
        rt = ragged_factory_ops.constant(rt_input, ragged_rank=ragged_rank)
        lengths = rt.row_lengths(axis)
        self.assertAllEqual(lengths, expected)
        if expected_ragged_rank is not None:
            if isinstance(lengths, ragged_tensor.RaggedTensor):
                self.assertEqual(lengths.ragged_rank, expected_ragged_rank)
            else:
                self.assertEqual(0, expected_ragged_rank)

    @parameterized.parameters([dict(rt_input=[[10, 20], [30]], axis=2, exception=(ValueError, errors.InvalidArgumentError)), dict(rt_input=[[2, 3, 0], [4, 1, 2]], axis=-3, exception=(ValueError, errors.InvalidArgumentError))])
    def testErrors(self, rt_input, exception, message=None, axis=1):
        if False:
            i = 10
            return i + 15
        rt = ragged_factory_ops.constant(rt_input)
        with self.assertRaisesRegex(exception, message):
            rt.row_lengths(axis)
if __name__ == '__main__':
    googletest.main()