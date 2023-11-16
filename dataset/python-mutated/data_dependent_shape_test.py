"""Model script to test TF-TensorRT integration with data dependent shapes"""
import os
from unittest import SkipTest
from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test

class TrtModeTestBase(trt_test.TfTrtIntegrationTestBase):
    """Creates a network that has data dependent shapes."""

    def GraphFn(self, x):
        if False:
            return 10
        x = math_ops.floor(x * 10)
        (y, idx) = array_ops.unique(x)
        y = y * 2 + y
        padding = array_ops.constant([0])
        n = array_ops.shape(x) - array_ops.shape(y)
        padding = array_ops.concat([padding, n], 0)
        padding = array_ops.expand_dims(padding, 0)
        y = array_ops.pad(y, padding)
        return array_ops.identity(y, name='output_0')

    def ShouldRunTest(self, run_params):
        if False:
            while True:
                i = 10
        return (run_params.dynamic_engine and run_params.is_v2 and run_params.dynamic_shape and run_params.use_calibration, 'test v2 dynamic engine and calibration')

    def setUp(self):
        if False:
            return 10
        super().setUp()
        os.environ['TF_TRT_ALLOW_ENGINE_NATIVE_SEGMENT_EXECUTION'] = 'True'

    def tearDown(self):
        if False:
            while True:
                i = 10
        super().tearDown()
        os.environ['TF_TRT_ALLOW_ENGINE_NATIVE_SEGMENT_EXECUTION'] = 'False'

    def GetParams(self):
        if False:
            i = 10
            return i + 15
        return self.BuildParams(self.GraphFn, dtypes.float32, [[12]], [[12]])

    def ExpectedEnginesToBuild(self, run_params):
        if False:
            for i in range(10):
                print('nop')
        return ['TRTEngineOp_000', 'TRTEngineOp_001']
if __name__ == '__main__':
    test.main()