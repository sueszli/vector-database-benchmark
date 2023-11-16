"""Tests for ragged_array_ops.reverse."""
from absl.testing import parameterized
from tensorflow.python.framework import test_util
from tensorflow.python.ops.ragged import ragged_array_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import googletest

@test_util.run_all_in_graph_and_eager_modes
class RaggedReverseOpTest(test_util.TensorFlowTestCase, parameterized.TestCase):

    @parameterized.parameters([dict(descr='Docstring example 1', data=[[[1, 2], [3, 4]], [[5, 6]], [[7, 8], [9, 10], [11, 12]]], axis=[0, 2], expected=[[[8, 7], [10, 9], [12, 11]], [[6, 5]], [[2, 1], [4, 3]]]), dict(descr='data.shape=[5, (D2)]; axis=[0]', data=[[1, 2], [3, 4, 5, 6], [7, 8, 9], [], [1, 2, 3]], axis=[0], expected=[[1, 2, 3], [], [7, 8, 9], [3, 4, 5, 6], [1, 2]]), dict(descr='data.shape=[5, (D2)]; axis=[1]', data=[[1, 2], [3, 4, 5, 6], [7, 8, 9], [], [1, 2, 3]], axis=[1], expected=[[2, 1], [6, 5, 4, 3], [9, 8, 7], [], [3, 2, 1]]), dict(descr='data.shape=[5, (D2), (D3)]; axis=[0, -1]', data=[[[1], [2, 3]], [[4, 5], [6, 7]], [[8]]], axis=[0, -1], expected=[[[8]], [[5, 4], [7, 6]], [[1], [3, 2]]]), dict(descr='data.shape=[2, (D2), 2]; axis=[2]', data=[[[1, 2], [3, 4]], [[5, 6]]], axis=[2], expected=[[[2, 1], [4, 3]], [[6, 5]]], ragged_rank=1), dict(descr='data.shape=[2, (D2), (D3)]; axis=[-1]', data=[[[1, 2], [3, 4]], [[5, 6]]], axis=[-1], expected=[[[2, 1], [4, 3]], [[6, 5]]]), dict(descr='data.shape=[2, (D2), (D3)]; axis=[]', data=[[[1, 2], [3, 4]], [[5, 6]]], axis=[], expected=[[[1, 2], [3, 4]], [[5, 6]]])])
    def testReverse(self, descr, data, axis, expected, ragged_rank=None):
        if False:
            return 10
        data = ragged_factory_ops.constant(data, ragged_rank=ragged_rank)
        result = ragged_array_ops.reverse(data, axis)
        expected = ragged_factory_ops.constant(expected, ragged_rank=ragged_rank)
        self.assertAllClose(result, expected)

    def testErrors(self):
        if False:
            return 10
        self.assertRaisesRegex(TypeError, '`axis` must be a list of int or a constant tensor *', ragged_array_ops.reverse, ragged_factory_ops.constant([[1], [2, 3]], ragged_rank=1), [0, None])
if __name__ == '__main__':
    googletest.main()