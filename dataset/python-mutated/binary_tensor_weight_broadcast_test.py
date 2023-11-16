"""Model script to test TF-TensorRT integration."""
import os
import numpy as np
from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging

class BinaryTensorWeightBroadcastTest(trt_test.TfTrtIntegrationTestBase):
    """Tests for scale & elementwise layers in TF-TRT."""

    def _ConstOp(self, shape):
        if False:
            for i in range(10):
                print('nop')
        return constant_op.constant(np.random.randn(*shape), dtype=dtypes.float32)

    def GraphFn(self, x):
        if False:
            while True:
                i = 10
        for weights_shape in [(1,), (24, 1, 1), (24, 24, 20), (20,), (1, 24, 1, 1), (1, 24, 24, 1), (1, 24, 24, 20), (24, 20)]:
            a = self._ConstOp(weights_shape)
            f = x + a
            x = self.trt_incompatible_op(f)
            a = self._ConstOp(weights_shape)
            f = a + x
            x = self.trt_incompatible_op(f)
        return gen_array_ops.reshape(x, [5, -1], name='output_0')

    def GetParams(self):
        if False:
            print('Hello World!')
        return self.BuildParams(self.GraphFn, dtypes.float32, [[10, 24, 24, 20]], [[5, 23040]])

    def ExpectedEnginesToBuild(self, run_params):
        if False:
            i = 10
            return i + 15
        'Return the expected engines to build.'
        num_engines = 17 if run_params.dynamic_shape else 16
        return [f'TRTEngineOp_{i:03d}' for i in range(num_engines)]

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        os.environ['TF_TRT_ALLOW_ENGINE_NATIVE_SEGMENT_EXECUTION'] = 'True'
        gpus = config.list_physical_devices('GPU')
        logging.info('Found the following GPUs:')
        for gpu in gpus:
            logging.info(f'\t- {gpu}')
            config.set_memory_growth(gpu, True)
if __name__ == '__main__':
    test.main()