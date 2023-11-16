"""Tests for the segment_id_ops.row_splits_to_segment_ids() op."""
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.ops.ragged import segment_id_ops
from tensorflow.python.platform import googletest

@test_util.run_all_in_graph_and_eager_modes
class RaggedSplitsToSegmentIdsOpTest(test_util.TensorFlowTestCase):

    def testDocStringExample(self):
        if False:
            for i in range(10):
                print('nop')
        splits = [0, 3, 3, 5, 6, 9]
        expected = [0, 0, 0, 2, 2, 3, 4, 4, 4]
        segment_ids = segment_id_ops.row_splits_to_segment_ids(splits)
        self.assertAllEqual(segment_ids, expected)

    def testEmptySplits(self):
        if False:
            for i in range(10):
                print('nop')
        segment_ids = segment_id_ops.row_splits_to_segment_ids([0])
        self.assertAllEqual(segment_ids, [])

    def testErrors(self):
        if False:
            print('Hello World!')
        self.assertRaisesRegex(ValueError, 'Invalid row_splits: \\[\\]', segment_id_ops.row_splits_to_segment_ids, [])
        self.assertRaisesRegex(ValueError, 'splits must have dtype int32 or int64', segment_id_ops.row_splits_to_segment_ids, constant_op.constant([0.5]))
        self.assertRaisesRegex(ValueError, 'Shape \\(\\) must have rank 1', segment_id_ops.row_splits_to_segment_ids, 0)
        self.assertRaisesRegex(ValueError, 'Shape \\(1, 1\\) must have rank 1', segment_id_ops.row_splits_to_segment_ids, [[0]])
if __name__ == '__main__':
    googletest.main()