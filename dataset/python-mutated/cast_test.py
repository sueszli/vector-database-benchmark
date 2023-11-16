"""Test conversion of graphs involving INT32 tensors and operations."""
import numpy as np
from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test

class CastInt32ToFp32Test(trt_test.TfTrtIntegrationTestBase):
    """Tests cast to FP32 are splitted in FP16 mode."""

    def _ConstOp(self, shape, dtype):
        if False:
            i = 10
            return i + 15
        return constant_op.constant(np.random.randn(*shape), dtype=dtype)

    def GraphFn(self, x):
        if False:
            for i in range(10):
                print('nop')
        b_f = self._ConstOp((1, 10), dtypes.float16)
        x_f = math_ops.cast(x, dtypes.float16)
        x_f = math_ops.mul(x_f, b_f)
        x_f = math_ops.cast(x_f, dtypes.float32)
        b_f = self._ConstOp((1, 10), dtypes.float32)
        x_f = math_ops.add(x_f, b_f)
        return array_ops.identity(x_f, name='output_0')

    def GetParams(self):
        if False:
            for i in range(10):
                print('nop')
        return self.BuildParams(self.GraphFn, dtypes.float32, [[1, 10]], [[1, 10]])

    def ExpectedEnginesToBuild(self, run_params):
        if False:
            while True:
                i = 10
        'Returns the expected engines to build.'
        return {'TRTEngineOp_000': ['AddV2', 'Cast', 'Const', 'Mul']}

    def ExpectedAbsoluteTolerance(self, run_params):
        if False:
            print('Hello World!')
        'The absolute tolerance to compare floating point results.'
        return 0.001 if run_params.precision_mode == 'FP32' else 0.01

    def ExpectedRelativeTolerance(self, run_params):
        if False:
            print('Hello World!')
        'The relative tolerance to compare floating point results.'
        return 0.001 if run_params.precision_mode == 'FP32' else 0.01
if __name__ == '__main__':
    test.main()