"""Testing the impact of graph node _tftrt_op_max_batch_size annotation on TRTEngineOp attributes."""
import unittest
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test

class MaxBatchSizesTestBase(trt_test.TfTrtIntegrationTestBase):

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        if cls is MaxBatchSizesTestBase:
            raise unittest.SkipTest('MaxBatchSizesTestBase defines base class for other tests.')
        super(MaxBatchSizesTestBase, cls).setUpClass()

    @property
    def tensor_shapes(self):
        if False:
            return 10
        return [[1, 512, 1, 1], [64, 2, 2, 2], [32, 4, 2, 2], [16, 8, 2, 2]]

    @property
    def max_batch_sizes(self):
        if False:
            for i in range(10):
                print('nop')
        return [shape[0] for shape in self.tensor_shapes]

    def GetParams(self):
        if False:
            i = 10
            return i + 15
        'Gets the build parameters for the test.'
        return self.BuildParams(self.GraphFn, dtype=dtypes.float32, input_shapes=[self.tensor_shapes[0]], output_shapes=[self.tensor_shapes[-1]])

    def ShouldRunTest(self, run_params):
        if False:
            i = 10
            return i + 15
        return (not run_params.dynamic_engine, 'test static engine only.')

    def GetMaxBatchSize(self, run_params):
        if False:
            i = 10
            return i + 15
        'Returns the max_batch_size that the converter should use for tests.'
        if run_params.dynamic_engine:
            return None
        return min(self.max_batch_sizes)

    def ExpectedEnginesToBuild(self, run_params):
        if False:
            for i in range(10):
                print('nop')
        'Checks that the expected engine is built.\n\n    Args:\n      run_params: the run parameters.\n\n    Returns:\n      the expected engines to build.\n\n    There shall be engines generated for each maximum batch size.\n    '
        return [f'TRTEngineOp_{seq_id:03d}' for seq_id in range(len(self.max_batch_sizes))]

    def ExpectedMaxBatchSizes(self, run_params):
        if False:
            i = 10
            return i + 15
        'Checks that the expected maximum batch sizes for the generated engines.\n\n    Args:\n      run_params: the run parameters.\n\n    Returns:\n      the expected maximum batch sizes for the generated engines.\n\n    There shall be engines generated for each maximum batch size.\n    '
        return self.max_batch_sizes

class AnnotateMaxBatchSizesTest(MaxBatchSizesTestBase):

    def GraphFn(self, inp):
        if False:
            return 10
        'Builds a tf.Graph for the test.'
        tensor = inp * 2.0
        tensor = array_ops.reshape(tensor, [-1] + self.tensor_shapes[1][1:])
        with ops.get_default_graph()._attr_scope({'_tftrt_op_max_batch_size': attr_value_pb2.AttrValue(i=self.max_batch_sizes[1])}):
            tensor = tensor + 3.0
        tensor = array_ops.reshape(tensor, [-1] + self.tensor_shapes[2][1:])
        with ops.get_default_graph()._attr_scope({'_tftrt_op_max_batch_size': attr_value_pb2.AttrValue(i=self.max_batch_sizes[2])}):
            tensor = tensor * 4.0
        tensor = array_ops.reshape(tensor, [-1] + self.tensor_shapes[3][1:])
        with ops.get_default_graph()._attr_scope({'_tftrt_op_max_batch_size': attr_value_pb2.AttrValue(i=self.max_batch_sizes[3])}):
            tensor += tensor + 5.0
        return array_ops.identity(tensor, name='output_0')

class StaticBatchSizeTest(MaxBatchSizesTestBase):

    def GraphFn(self, inp):
        if False:
            for i in range(10):
                print('nop')
        'Builds a tf.Graph for the test.'
        tensor = inp * 2.0
        tensor = array_ops.reshape(tensor, self.tensor_shapes[1])
        tensor = tensor + 3.0
        tensor = array_ops.reshape(tensor, self.tensor_shapes[2])
        tensor = tensor * 4.0
        tensor = array_ops.reshape(tensor, self.tensor_shapes[3])
        tensor += tensor + 5.0
        return array_ops.identity(tensor, name='output_0')
if __name__ == '__main__':
    test.main()