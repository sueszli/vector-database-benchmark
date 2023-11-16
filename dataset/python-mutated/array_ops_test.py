"""Tests for array_ops that only work in V1."""
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.platform import test as test_lib

class ReverseV2Test(test_util.TensorFlowTestCase):

    @test_util.run_deprecated_v1
    def testUnknownDims(self):
        if False:
            for i in range(10):
                print('nop')
        reverse_v2 = array_ops.reverse_v2
        data_t = array_ops.placeholder(dtypes.float32)
        axis_known_t = array_ops.placeholder(dtypes.int32, shape=[3])
        reverse_known_t = reverse_v2(data_t, axis_known_t)
        self.assertIsNone(reverse_known_t.get_shape().ndims)
        axis_unknown_t = array_ops.placeholder(dtypes.int32)
        reverse_unknown_t = reverse_v2(data_t, axis_unknown_t)
        self.assertIs(None, reverse_unknown_t.get_shape().ndims)
        data_2d_t = array_ops.placeholder(dtypes.float32, shape=[None, None])
        axis_2d_t = array_ops.placeholder(dtypes.int32, shape=[3])
        reverse_2d_t = reverse_v2(data_2d_t, axis_2d_t)
        self.assertEqual(2, reverse_2d_t.get_shape().ndims)

class SequenceMaskTest(test_util.TensorFlowTestCase):

    @test_util.run_deprecated_v1
    def testUnknownShape(self):
        if False:
            return 10
        lengths = array_ops.placeholder(dtype=dtypes.int32)
        res = array_ops.sequence_mask(lengths)
        self.assertEqual(res.shape, None)

class BatchGatherNdTest(test_util.TensorFlowTestCase):

    @test_util.run_deprecated_v1
    def testUnknownIndices(self):
        if False:
            i = 10
            return i + 15
        'Tests whether indices with unknown rank works correctly.'
        params = constant_op.constant(((0, 1, 2),))
        indices = array_ops.placeholder(dtypes.int32)
        gather_nd_t = array_ops.gather_nd(params, indices, batch_dims=1)
        shape = gather_nd_t.get_shape()
        self.assertIsNone(shape.ndims)
        self.assertIsNone(tensor_shape.dimension_value(shape[0]))

class SliceAssignTest(test_util.TensorFlowTestCase):

    @test_util.run_v1_only('Variables need initialization only in V1')
    def testUninitialized(self):
        if False:
            return 10
        with self.assertRaisesRegex(errors.FailedPreconditionError, 'Attempting to use uninitialized value Variable'):
            v = variable_v1.VariableV1([1, 2])
            self.evaluate(v[:].assign([1, 2]))
if __name__ == '__main__':
    test_lib.main()