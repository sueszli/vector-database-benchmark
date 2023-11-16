"""Model script to test TF-TensorRT integration."""
import os
import numpy as np
from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test

class VGGBlockTest(trt_test.TfTrtIntegrationTestBase):
    """Single vgg layer test in TF-TRT conversion."""

    def GraphFn(self, x):
        if False:
            i = 10
            return i + 15
        dtype = x.dtype
        (x, _, _) = nn_impl.fused_batch_norm(x, [1.0, 1.0], [0.0, 0.0], mean=[0.5, 0.5], variance=[1.0, 1.0], is_training=False)
        e = constant_op.constant(np.random.randn(1, 1, 2, 6), name='weights', dtype=dtype)
        conv = nn.conv2d(input=x, filter=e, strides=[1, 2, 2, 1], padding='SAME', name='conv')
        b = constant_op.constant(np.random.randn(6), name='bias', dtype=dtype)
        t = nn.bias_add(conv, b, name='biasAdd')
        relu = nn.relu(t, 'relu')
        idty = array_ops.identity(relu, 'ID')
        v = nn_ops.max_pool(idty, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID', name='max_pool')
        return array_ops.squeeze(v, name='output_0')

    def GetParams(self):
        if False:
            return 10
        return self.BuildParams(self.GraphFn, dtypes.float32, [[5, 8, 8, 2]], [[5, 2, 2, 6]])

    def ExpectedEnginesToBuild(self, run_params):
        if False:
            i = 10
            return i + 15
        'Return the expected engines to build.'
        return ['TRTEngineOp_000']

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        os.environ['TF_TRT_ALLOW_ENGINE_NATIVE_SEGMENT_EXECUTION'] = 'True'
if __name__ == '__main__':
    test.main()