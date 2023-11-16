"""Script to test TF-TRT INT8 conversion without calibration on Mnist model."""
import numpy as np
from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.platform import test

class DynamicInputShapesTest(trt_test.TfTrtIntegrationTestBase):

    def GraphFn(self, x):
        if False:
            print('Hello World!')
        conv_filter1 = constant_op.constant(np.ones([3, 3, 1, 8]), name='weights1', dtype=dtypes.float32)
        bias1 = constant_op.constant(np.random.randn(8), dtype=dtypes.float32)
        x = nn.conv2d(input=x, filter=conv_filter1, strides=[1, 1, 1, 1], padding='SAME', name='conv')
        x = nn.bias_add(x, bias1)
        x = nn.relu(x)
        conv_filter2 = constant_op.constant(np.ones([3, 3, 8, 1]), name='weights2', dtype=dtypes.float32)
        bias2 = constant_op.constant(np.random.randn(1), dtype=dtypes.float32)
        x = nn.conv2d(input=x, filter=conv_filter2, strides=[1, 1, 1, 1], padding='SAME', name='conv')
        x = nn.bias_add(x, bias2)
        return array_ops.identity(x, name='output')

    def GetParams(self):
        if False:
            for i in range(10):
                print('nop')
        input_dims = [[[1, 5, 5, 1]], [[10, 5, 5, 1]], [[3, 5, 5, 1]], [[1, 5, 5, 1]], [[1, 3, 1, 1]], [[2, 9, 9, 1]], [[1, 224, 224, 1]], [[1, 128, 224, 1]]]
        expected_output_dims = input_dims
        return trt_test.TfTrtIntegrationTestParams(graph_fn=self.GraphFn, input_specs=[tensor_spec.TensorSpec([None, None, None, 1], dtypes.float32, 'input')], output_specs=[tensor_spec.TensorSpec([None, None, None, 1], dtypes.float32, 'output')], input_dims=input_dims, expected_output_dims=expected_output_dims)

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self.DisableNonTrtOptimizers()

    def ExpectedEnginesToBuild(self, run_params):
        if False:
            for i in range(10):
                print('nop')
        return ['TRTEngineOp_000']

    def ShouldRunTest(self, run_params):
        if False:
            i = 10
            return i + 15
        return (run_params.dynamic_engine and (not trt_test.IsQuantizationMode(run_params.precision_mode)), 'test dynamic engine and non-INT8')

    def ExpectedAbsoluteTolerance(self, run_params):
        if False:
            while True:
                i = 10
        'The absolute tolerance to compare floating point results.'
        return 0.001 if run_params.precision_mode == 'FP32' else 0.1

    def ExpectedRelativeTolerance(self, run_params):
        if False:
            i = 10
            return i + 15
        'The relative tolerance to compare floating point results.'
        return 0.001 if run_params.precision_mode == 'FP32' else 0.1
if __name__ == '__main__':
    test.main()