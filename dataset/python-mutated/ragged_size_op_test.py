"""Tests for ragged.size."""
from absl.testing import parameterized
from tensorflow.python.framework import test_util
from tensorflow.python.ops.ragged import ragged_array_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import googletest

@test_util.run_all_in_graph_and_eager_modes
class RaggedSizeOpTest(test_util.TensorFlowTestCase, parameterized.TestCase):

    @parameterized.parameters([{'size': 1, 'test_input': 1}, {'size': 0, 'test_input': []}, {'size': 0, 'test_input': [], 'ragged_rank': 1}, {'size': 3, 'test_input': [1, 1, 1]}, {'size': 3, 'test_input': [[1, 1], [1]]}, {'size': 5, 'test_input': [[[1, 1, 1], [1]], [[1]]]}, {'size': 6, 'test_input': [[[1, 1], [1, 1]], [[1, 1]]], 'ragged_rank': 1}])
    def testRaggedSize(self, test_input, size, ragged_rank=None):
        if False:
            print('Hello World!')
        input_rt = ragged_factory_ops.constant(test_input, ragged_rank=ragged_rank)
        self.assertAllEqual(ragged_array_ops.size(input_rt), size)
if __name__ == '__main__':
    googletest.main()