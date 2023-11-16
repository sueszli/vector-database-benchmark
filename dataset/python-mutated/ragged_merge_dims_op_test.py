"""Tests for RaggedTensor.merge_dims."""
from absl.testing import parameterized
from tensorflow.python.eager import context
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import googletest
from tensorflow.python.util import nest

@test_util.run_all_in_graph_and_eager_modes
class RaggedMergeDimsOpTest(test_util.TensorFlowTestCase, parameterized.TestCase):

    @parameterized.named_parameters([{'testcase_name': '2DAxis0To1', 'rt': [[1, 2], [], [3, 4, 5]], 'outer_axis': 0, 'inner_axis': 1, 'expected': [1, 2, 3, 4, 5]}, {'testcase_name': '3DAxis0To1', 'rt': [[[1, 2], [], [3, 4, 5]], [[6], [7, 8], []]], 'outer_axis': 0, 'inner_axis': 1, 'expected': [[1, 2], [], [3, 4, 5], [6], [7, 8], []]}, {'testcase_name': '3DAxis1To2', 'rt': [[[1, 2], [], [3, 4, 5]], [[6], [7, 8], []]], 'outer_axis': 1, 'inner_axis': 2, 'expected': [[1, 2, 3, 4, 5], [6, 7, 8]]}, {'testcase_name': '3DAxis0To2', 'rt': [[[1, 2], [], [3, 4, 5]], [[6], [7, 8], []]], 'outer_axis': 0, 'inner_axis': 2, 'expected': [1, 2, 3, 4, 5, 6, 7, 8]}, {'testcase_name': '3DAxis0To1WithDenseValues', 'rt': [[[1, 2], [3, 4], [5, 6]], [[7, 8]]], 'ragged_ranks': (1, 2), 'outer_axis': 0, 'inner_axis': 1, 'expected': [[1, 2], [3, 4], [5, 6], [7, 8]]}, {'testcase_name': '3DAxis1To2WithDenseValues', 'rt': [[[1, 2], [3, 4], [5, 6]], [[7, 8]]], 'ragged_ranks': (1, 2), 'outer_axis': 1, 'inner_axis': 2, 'expected': [[1, 2, 3, 4, 5, 6], [7, 8]]}, {'testcase_name': '4DAxis0To1', 'rt': [[[[1, 2], [], [3, 4, 5]], [[6], [7, 8], []]], [[[9], [0]]]], 'outer_axis': 0, 'inner_axis': 1, 'expected': [[[1, 2], [], [3, 4, 5]], [[6], [7, 8], []], [[9], [0]]]}, {'testcase_name': '4DAxis1To2', 'rt': [[[[1, 2], [], [3, 4, 5]], [[6], [7, 8], []]], [[[9], [0]]]], 'outer_axis': 1, 'inner_axis': 2, 'expected': [[[1, 2], [], [3, 4, 5], [6], [7, 8], []], [[9], [0]]]}, {'testcase_name': '4DAxis2To3', 'rt': [[[[1, 2], [], [3, 4, 5]], [[6], [7, 8], []]], [[[9], [0]]]], 'outer_axis': 2, 'inner_axis': 3, 'expected': [[[1, 2, 3, 4, 5], [6, 7, 8]], [[9, 0]]]}, {'testcase_name': '4DAxis1To3', 'rt': [[[[1, 2], [], [3, 4, 5]], [[6], [7, 8], []]], [[[9], [0]]]], 'outer_axis': 1, 'inner_axis': 3, 'expected': [[1, 2, 3, 4, 5, 6, 7, 8], [9, 0]]}, {'testcase_name': '4DAxis1ToNeg1', 'rt': [[[[1, 2], [], [3, 4, 5]], [[6], [7, 8], []]], [[[9], [0]]]], 'outer_axis': 1, 'inner_axis': -1, 'expected': [[1, 2, 3, 4, 5, 6, 7, 8], [9, 0]]}, {'testcase_name': '4DAxis1To2WithDenseValues', 'rt': [[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]]]], 'ragged_ranks': (1, 2, 3), 'outer_axis': 1, 'inner_axis': 2, 'expected': [[[1, 2], [3, 4], [5, 6], [7, 8]], [[9, 10], [11, 12]]]}, {'testcase_name': '4DAxis2To3WithDenseValues', 'rt': [[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]]]], 'ragged_ranks': (1, 2, 3), 'outer_axis': 2, 'inner_axis': 3, 'expected': [[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12]]]}, {'testcase_name': '4DAxis1To3WithDenseValues', 'rt': [[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]]]], 'ragged_ranks': (1, 2, 3), 'outer_axis': 1, 'inner_axis': 3, 'expected': [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12]]}, {'testcase_name': '5DAxis2To3WithDenseValues', 'rt': [[[[[1, 2], [3, 4]]], [[[5, 6], [7, 8]]]], [[[[9, 10], [11, 12]]]]], 'ragged_ranks': (1, 2, 3, 4), 'outer_axis': 2, 'inner_axis': 3, 'expected': [[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]]]]}, {'testcase_name': '5DAxis3To4WithDenseValues', 'rt': [[[[[1, 2], [3, 4]]], [[[5, 6], [7, 8]]]], [[[[9, 10], [11, 12]]]]], 'ragged_ranks': (1, 2, 3, 4), 'outer_axis': 3, 'inner_axis': 4, 'expected': [[[[1, 2, 3, 4]], [[5, 6, 7, 8]]], [[[9, 10, 11, 12]]]]}, {'testcase_name': '5DAxis1To3WithDenseValues', 'rt': [[[[[1, 2], [3, 4]]], [[[5, 6], [7, 8]]]], [[[[9, 10], [11, 12]]]]], 'ragged_ranks': (1, 2, 3, 4), 'outer_axis': 1, 'inner_axis': 3, 'expected': [[[1, 2], [3, 4], [5, 6], [7, 8]], [[9, 10], [11, 12]]]}, {'testcase_name': 'OuterEqualsInner', 'rt': [[1], [2], [3, 4]], 'outer_axis': 0, 'inner_axis': 0, 'expected': [[1], [2], [3, 4]]}, {'testcase_name': 'OuterEqualsInnerWithNegativeAxis', 'rt': [[1], [2], [3, 4]], 'outer_axis': 1, 'inner_axis': -1, 'expected': [[1], [2], [3, 4]]}])
    def testRaggedMergeDims(self, rt, outer_axis, inner_axis, expected, ragged_ranks=(None,)):
        if False:
            while True:
                i = 10
        for ragged_rank in ragged_ranks:
            x = ragged_factory_ops.constant(rt, ragged_rank=ragged_rank)
            actual = x.merge_dims(outer_axis, inner_axis)
            self.assertAllEqual(expected, actual)
            if outer_axis >= 0 and inner_axis >= 0:
                self.assertEqual(actual.shape.rank, x.shape.rank - (inner_axis - outer_axis))
            if outer_axis >= 0 and inner_axis >= 0:
                actual_with_neg_axis = x.merge_dims(outer_axis - x.shape.rank, inner_axis - x.shape.rank)
                self.assertAllEqual(expected, actual_with_neg_axis)
            if not context.executing_eagerly() and outer_axis >= 0 and (inner_axis >= 0):
                x_with_placeholders = nest.map_structure(lambda t: array_ops.placeholder_with_default(t, None), x, expand_composites=True)
                actual_with_placeholders = x_with_placeholders.merge_dims(outer_axis, inner_axis)
                self.assertAllEqual(expected, actual_with_placeholders)

    @parameterized.parameters([{'rt': [[1]], 'outer_axis': {}, 'inner_axis': 1, 'exception': TypeError, 'message': 'outer_axis must be an int'}, {'rt': [[1]], 'outer_axis': 1, 'inner_axis': {}, 'exception': TypeError, 'message': 'inner_axis must be an int'}, {'rt': [[1]], 'outer_axis': 1, 'inner_axis': 3, 'exception': ValueError, 'message': 'inner_axis=3 out of bounds: expected -2<=inner_axis<2'}, {'rt': [[1]], 'outer_axis': 1, 'inner_axis': -3, 'exception': ValueError, 'message': 'inner_axis=-3 out of bounds: expected -2<=inner_axis<2'}, {'rt': [[1]], 'outer_axis': 1, 'inner_axis': 0, 'exception': ValueError, 'message': 'Expected outer_axis .* to be less than or equal to .*'}, {'rt': [[1]], 'outer_axis': -1, 'inner_axis': -2, 'exception': ValueError, 'message': 'Expected outer_axis .* to be less than or equal to .*'}])
    def testRaggedMergeDimsError(self, rt, outer_axis, inner_axis, exception, message=None, ragged_rank=None):
        if False:
            for i in range(10):
                print('nop')
        x = ragged_factory_ops.constant(rt, ragged_rank=ragged_rank)
        with self.assertRaisesRegex(exception, message):
            self.evaluate(x.merge_dims(outer_axis, inner_axis))
if __name__ == '__main__':
    googletest.main()