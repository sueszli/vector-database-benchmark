"""Model script to test TF-TensorRT integration."""
import numpy as np
from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.platform import test

@test_util.run_all_without_tensor_float_32('Uses matmul')
class MemoryAlignmentTest(trt_test.TfTrtIntegrationTestBase):
    """Testing conversion of BatchMatMul in TF-TRT conversion."""

    def GraphFn(self, inp):
        if False:
            return 10
        dtype = inp.dtype
        e1 = constant_op.constant(np.random.randn(1, 1, 3, 5), name='kernel_1', dtype=dtype)
        e2 = constant_op.constant(np.random.randn(1, 1, 5, 10), name='kernel_2', dtype=dtype)
        conv = nn.conv2d(input=inp, filter=e1, strides=[1, 1, 1, 1], padding='VALID', name='conv')
        out = nn.conv2d(input=conv, filter=e2, strides=[1, 1, 1, 1], padding='VALID', name='conv_2')
        return array_ops.squeeze(out, name='output_0')

    def GetParams(self):
        if False:
            while True:
                i = 10
        return self.BuildParams(self.GraphFn, dtypes.float32, [[2, 15, 15, 3]], [[2, 15, 15, 10]])

    def ExpectedEnginesToBuild(self, run_params):
        if False:
            i = 10
            return i + 15
        'Return the expected engines to build.'
        return ['TRTEngineOp_000']

    def ExpectedAbsoluteTolerance(self, run_params):
        if False:
            while True:
                i = 10
        'The absolute tolerance to compare floating point results.'
        return 1e-06 if run_params.precision_mode == 'FP32' else 0.01

    def ExpectedRelativeTolerance(self, run_params):
        if False:
            print('Hello World!')
        'The relative tolerance to compare floating point results.'
        return 0.1
if __name__ == '__main__':
    test.main()