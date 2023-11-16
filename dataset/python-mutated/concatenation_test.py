"""Model script to test TF-TensorRT integration."""
import numpy as np
from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.platform import test

class ConcatenationTest(trt_test.TfTrtIntegrationTestBase):
    """Testing Concatenation in TF-TRT conversion."""

    def GraphFn(self, x):
        if False:
            while True:
                i = 10
        dtype = x.dtype
        a = constant_op.constant(np.random.randn(3, 1, 1), dtype=dtype)
        r1 = x / a
        a = constant_op.constant(np.random.randn(3, 1, 1), dtype=dtype)
        r2 = a / x
        a = constant_op.constant(np.random.randn(1, 3, 1), dtype=dtype)
        r3 = a + x
        a = constant_op.constant(np.random.randn(1, 3, 1), dtype=dtype)
        r4 = x * a
        a = constant_op.constant(np.random.randn(3, 1, 1), dtype=dtype)
        r5 = x - a
        a = constant_op.constant(np.random.randn(3, 1, 1), dtype=dtype)
        r6 = a - x
        a = constant_op.constant(np.random.randn(3, 1), dtype=dtype)
        r7 = x - a
        a = constant_op.constant(np.random.randn(3, 1), dtype=dtype)
        r8 = a - x
        a = constant_op.constant(np.random.randn(3, 1, 1), dtype=dtype)
        r9 = gen_math_ops.maximum(x, a)
        a = constant_op.constant(np.random.randn(3, 1), dtype=dtype)
        r10 = gen_math_ops.minimum(a, x)
        a = constant_op.constant(np.random.randn(3), dtype=dtype)
        r11 = x * a
        a = constant_op.constant(np.random.randn(1), dtype=dtype)
        r12 = a * x
        concat1 = array_ops.concat([r1, r2, r3, r4, r5, r6], axis=-1)
        concat2 = array_ops.concat([r7, r8, r9, r10, r11, r12], axis=3)
        x = array_ops.concat([concat1, concat2], axis=-1)
        return gen_array_ops.reshape(x, [2, -1], name='output_0')

    def GetParams(self):
        if False:
            print('Hello World!')
        return self.BuildParams(self.GraphFn, dtypes.float32, [[2, 3, 3, 1]], [[2, 126]])

    def ExpectedEnginesToBuild(self, run_params):
        if False:
            return 10
        'Return the expected engines to build.'
        return ['TRTEngineOp_000']
if __name__ == '__main__':
    test.main()