"""Script to test TF-TensorRT integration."""
from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import nn
from tensorflow.python.platform import test

class ConstBroadcastTest(trt_test.TfTrtIntegrationTestBase):
    """Test for Constant broadcasting in TF-TRT."""

    def GraphFn(self, x):
        if False:
            return 10
        'Return the expected graph to convert.'
        dtype = x.dtype
        filt1 = constant_op.constant(0.3, shape=(3, 3, 2, 1), dtype=dtype, name='filt1')
        y1 = nn.conv2d(x, filt1, strides=[1, 1, 1, 1], padding='SAME', name='y1')
        z1 = nn.relu(y1, name='z1')
        filt2 = constant_op.constant(0.3, shape=(3, 3, 1, 1), dtype=dtype, name='filt2')
        y2 = nn.conv2d(z1, filt2, strides=[1, 1, 1, 1], padding='SAME', name='y2')
        z2 = nn.relu(y2, name='z')
        filt3 = constant_op.constant(0.3, shape=(3, 3, 1, 1), dtype=dtype, name='filt3')
        y3 = nn.conv2d(z2, filt3, strides=[1, 1, 1, 1], padding='SAME', name='y3')
        return nn.relu(y3, name='output_0')

    def GetParams(self):
        if False:
            i = 10
            return i + 15
        return self.BuildParams(self.GraphFn, dtypes.float32, [[5, 12, 12, 2]], [[5, 12, 12, 1]])

    def ExpectedEnginesToBuild(self, run_params):
        if False:
            i = 10
            return i + 15
        'Return the expected engines to build.'
        return ['TRTEngineOp_000']

    def ExpectedAbsoluteTolerance(self, run_params):
        if False:
            i = 10
            return i + 15
        'The absolute tolerance to compare floating point results.'
        return 0.0001 if run_params.precision_mode == 'FP32' else 0.01

    def ExpectedRelativeTolerance(self, run_params):
        if False:
            i = 10
            return i + 15
        'The relative tolerance to compare floating point results.'
        return 0.0001 if run_params.precision_mode == 'FP32' else 0.01
if __name__ == '__main__':
    test.main()