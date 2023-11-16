"""Tests for ragged_array_ops.expand_dims."""
from absl.testing import parameterized
from tensorflow.python.framework import test_util
from tensorflow.python.ops.ragged import ragged_array_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import googletest

@test_util.run_all_in_graph_and_eager_modes
class RaggedExpandDimsOpTest(test_util.TensorFlowTestCase, parameterized.TestCase):
    EXAMPLE4D = [[[[1, 1], [2, 2]], [[3, 3]]], [], [[], [[4, 4], [5, 5], [6, 6]]]]
    EXAMPLE4D_EXPAND_AXIS = {0: [EXAMPLE4D], 1: [[d0] for d0 in EXAMPLE4D], 2: [[[d1] for d1 in d0] for d0 in EXAMPLE4D], 3: [[[[d2] for d2 in d1] for d1 in d0] for d0 in EXAMPLE4D], 4: [[[[[d3] for d3 in d2] for d2 in d1] for d1 in d0] for d0 in EXAMPLE4D]}

    @parameterized.parameters([dict(rt_input=[[1, 2], [3]], axis=0, expected=[[[1, 2], [3]]], expected_shape=[1, 2, None]), dict(rt_input=[[1, 2], [3]], axis=1, expected=[[[1, 2]], [[3]]], expected_shape=[2, 1, None]), dict(rt_input=[[1, 2], [3]], axis=2, expected=[[[1], [2]], [[3]]], expected_shape=[2, None, 1]), dict(rt_input=[[1, 2], [3, 4], [5, 6]], ragged_rank=0, axis=0, expected=[[[1, 2], [3, 4], [5, 6]]], expected_shape=[1, 3, 2]), dict(rt_input=[[1, 2], [3, 4], [5, 6]], ragged_rank=0, axis=1, expected=[[[1, 2]], [[3, 4]], [[5, 6]]], expected_shape=[3, 1, 2]), dict(rt_input=[[1, 2], [3, 4], [5, 6]], ragged_rank=0, axis=2, expected=[[[1], [2]], [[3], [4]], [[5], [6]]], expected_shape=[3, 2, 1]), dict(rt_input=EXAMPLE4D, ragged_rank=2, axis=0, expected=EXAMPLE4D_EXPAND_AXIS[0], expected_shape=[1, 3, None, None, 2]), dict(rt_input=EXAMPLE4D, ragged_rank=2, axis=1, expected=EXAMPLE4D_EXPAND_AXIS[1], expected_shape=[3, 1, None, None, 2]), dict(rt_input=EXAMPLE4D, ragged_rank=2, axis=2, expected=EXAMPLE4D_EXPAND_AXIS[2], expected_shape=[3, None, 1, None, 2]), dict(rt_input=EXAMPLE4D, ragged_rank=2, axis=3, expected=EXAMPLE4D_EXPAND_AXIS[3], expected_shape=[3, None, None, 1, 2]), dict(rt_input=EXAMPLE4D, ragged_rank=2, axis=4, expected=EXAMPLE4D_EXPAND_AXIS[4], expected_shape=[3, None, None, 2, 1])])
    def testRaggedExpandDims(self, rt_input, axis, expected, ragged_rank=None, expected_shape=None):
        if False:
            for i in range(10):
                print('nop')
        rt = ragged_factory_ops.constant(rt_input, ragged_rank=ragged_rank)
        expanded = ragged_array_ops.expand_dims(rt, axis=axis)
        self.assertEqual(expanded.shape.ndims, rt.shape.ndims + 1)
        if expected_shape is not None:
            self.assertEqual(expanded.shape.as_list(), expected_shape)
        self.assertAllEqual(expanded, expected)
if __name__ == '__main__':
    googletest.main()