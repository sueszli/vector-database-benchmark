"""Test conversion of graphs involving INT32 tensors and operations."""
import numpy as np
from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.platform import test

class ExcludeUnsupportedInt32Test(trt_test.TfTrtIntegrationTestBase):
    """Test exclusion of ops which are not supported in INT32 mode by TF-TRT"""

    def _ConstOp(self, shape, dtype):
        if False:
            print('Hello World!')
        return constant_op.constant(np.random.randn(*shape), dtype=dtype)

    def GraphFn(self, x):
        if False:
            for i in range(10):
                print('nop')
        dtype = x.dtype
        b = self._ConstOp((4, 10), dtype)
        x = math_ops.matmul(x, b)
        b = self._ConstOp((10,), dtype)
        x = nn.bias_add(x, b)
        return array_ops.identity(x, name='output_0')

    def GetParams(self):
        if False:
            for i in range(10):
                print('nop')
        return self.BuildParams(self.GraphFn, dtypes.int32, [[100, 4]], [[100, 10]])

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self.DisableNonTrtOptimizers()

    def GetMaxBatchSize(self, run_params):
        if False:
            return 10
        'Returns the max_batch_size that the converter should use for tests.'
        if run_params.dynamic_engine:
            return None
        return 100

    def ExpectedEnginesToBuild(self, run_params):
        if False:
            return 10
        'Return the expected engines to build.'
        return []

class CalibrationInt32Support(trt_test.TfTrtIntegrationTestBase):
    """Test execution of calibration with int32 input"""

    def GraphFn(self, inp):
        if False:
            for i in range(10):
                print('nop')
        inp_transposed = array_ops.transpose(inp, [0, 3, 2, 1], name='transpose_0')
        return array_ops.identity(inp_transposed, name='output_0')

    def GetParams(self):
        if False:
            while True:
                i = 10
        return self.BuildParams(self.GraphFn, dtypes.int32, [[3, 4, 5, 6]], [[3, 6, 5, 4]])

    def ShouldRunTest(self, run_params):
        if False:
            return 10
        return (trt_test.IsQuantizationWithCalibration(run_params), 'test calibration and INT8')

    def ExpectedEnginesToBuild(self, run_params):
        if False:
            print('Hello World!')
        return ['TRTEngineOp_000']
if __name__ == '__main__':
    test.main()