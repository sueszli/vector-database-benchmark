"""Tests for ragged.rank op."""
from absl.testing import parameterized
from tensorflow.python.framework import test_util
from tensorflow.python.ops.ragged import ragged_array_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import googletest

@test_util.run_all_in_graph_and_eager_modes
class RaggedRankOpTest(test_util.TensorFlowTestCase, parameterized.TestCase):

    @parameterized.parameters([dict(test_input=1, expected_rank=0), dict(test_input=[1], expected_rank=1), dict(test_input=[1, 2, 3, 4], expected_rank=1), dict(test_input=[[1], [2], [3]], expected_rank=2), dict(test_input=[[[1], [2, 3]], [[4], [5, 6, 7]]], expected_rank=3), dict(test_input=[[[1], [2, 3], [10, 20]], [[4], [5, 6, 7]]], expected_rank=3, ragged_rank=2), dict(test_input=[[[[1], [2]]], [[[3, 4], [5, 6]], [[7, 8], [9, 10]]]], expected_rank=4), dict(test_input=[[[[1, 2]]], [[[5, 6], [7, 8]], [[9, 10], [11, 12]]]], expected_rank=4, ragged_rank=2)])
    def testRaggedRank(self, test_input, expected_rank, ragged_rank=None):
        if False:
            while True:
                i = 10
        test_input = ragged_factory_ops.constant(test_input, ragged_rank=ragged_rank)
        self.assertAllEqual(ragged_array_ops.rank(test_input), expected_rank)
if __name__ == '__main__':
    googletest.main()