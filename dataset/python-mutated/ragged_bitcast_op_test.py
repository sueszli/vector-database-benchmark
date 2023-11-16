"""Tests for ragged_array_ops.bitcast."""
from absl.testing import parameterized
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops.ragged import ragged_array_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import googletest

@test_util.run_all_in_graph_and_eager_modes
class RaggedSplitOpTest(test_util.TensorFlowTestCase, parameterized.TestCase):

    @parameterized.parameters([dict(descr='int32 to int32 cast', inputs=ragged_factory_ops.constant_value([[1, 2], [3]], dtype=dtypes.int32), outputs=ragged_factory_ops.constant_value([[1, 2], [3]], dtype=dtypes.int32)), dict(descr='int32 to uint32 cast', inputs=ragged_factory_ops.constant_value([[1, 2], [-1]], dtype=dtypes.int32), outputs=ragged_factory_ops.constant_value([[1, 2], [4294967295]], dtype=dtypes.uint32)), dict(descr='uint32 to int32 cast', inputs=ragged_factory_ops.constant_value([[1, 2], [4294967295]], dtype=dtypes.uint32), outputs=ragged_factory_ops.constant_value([[1, 2], [-1]], dtype=dtypes.int32)), dict(descr='int32 to int64 cast', inputs=ragged_factory_ops.constant_value([[[1, 0], [2, 0]], [[3, 0]]], dtype=dtypes.int32, ragged_rank=1), outputs=ragged_factory_ops.constant_value([[1, 2], [3]], dtype=dtypes.int64)), dict(descr='int64 to int32 cast', inputs=ragged_factory_ops.constant_value([[1, 2], [3]], dtype=dtypes.int64), outputs=ragged_factory_ops.constant_value([[[1, 0], [2, 0]], [[3, 0]]], dtype=dtypes.int32, ragged_rank=1))])
    def testBitcast(self, descr, inputs, outputs, name=None):
        if False:
            for i in range(10):
                print('nop')
        result = ragged_array_ops.bitcast(inputs, outputs.dtype, name)
        self.assertEqual(result.dtype, outputs.dtype)
        self.assertEqual(result.ragged_rank, outputs.ragged_rank)
        self.assertAllEqual(result, outputs)

    @parameterized.parameters([dict(descr='Upcast requires uniform inner dimension', inputs=ragged_factory_ops.constant_value([[[1, 0], [2, 0]], [[3, 0]]], dtype=dtypes.int32, ragged_rank=2), cast_to_dtype=dtypes.int64, exception=ValueError, message='`input.flat_values` is required to have rank >= 2')])
    def testBitcastError(self, descr, inputs, cast_to_dtype, exception, message, name=None):
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(exception, message):
            result = ragged_array_ops.bitcast(inputs, cast_to_dtype, name)
            self.evaluate(result)
if __name__ == '__main__':
    googletest.main()