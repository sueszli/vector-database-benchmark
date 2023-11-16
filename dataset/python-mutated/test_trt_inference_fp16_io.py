import os
import tempfile
import unittest
import numpy as np
import paddle
from paddle.inference import Config, PrecisionType, create_predictor
from paddle.jit import to_static
from paddle.static import InputSpec
from paddle.vision.models import alexnet

class TestEnableLowPrecisionIO:

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.temp_dir = tempfile.TemporaryDirectory()
        net = alexnet(True)
        model = to_static(net, input_spec=[InputSpec(shape=[None, 3, 224, 224], name='x')])
        paddle.jit.save(model, os.path.join(self.temp_dir.name, 'alexnet/inference'))

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        self.temp_dir.cleanup()

    def get_fp32_output(self):
        if False:
            print('Hello World!')
        predictor = self.init_predictor(low_precision_io=False)
        inputs = [paddle.to_tensor(0.1 * np.ones([1, 3, 224, 224]).astype(np.float32))]
        outputs = predictor.run(inputs)
        return outputs[0]

    def get_fp16_output(self):
        if False:
            i = 10
            return i + 15
        predictor = self.init_predictor(low_precision_io=True)
        inputs = [paddle.to_tensor(0.1 * np.ones([1, 3, 224, 224]).astype(np.float16))]
        outputs = predictor.run(inputs)
        return outputs[0]

    def test_output(self):
        if False:
            i = 10
            return i + 15
        if paddle.is_compiled_with_cuda():
            fp32_output = self.get_fp32_output()
            fp16_output = self.get_fp16_output()

class TestEnableLowPrecisionIOWithGPU(TestEnableLowPrecisionIO, unittest.TestCase):

    def init_predictor(self, low_precision_io: bool):
        if False:
            for i in range(10):
                print('nop')
        config = Config(os.path.join(self.temp_dir.name, 'alexnet/inference.pdmodel'), os.path.join(self.temp_dir.name, 'alexnet/inference.pdiparams'))
        config.enable_use_gpu(256, 0, PrecisionType.Half)
        config.enable_memory_optim()
        config.enable_low_precision_io(low_precision_io)
        config.disable_glog_info()
        predictor = create_predictor(config)
        return predictor

class TestEnableLowPrecisionIOWithTRTAllGraph(TestEnableLowPrecisionIO, unittest.TestCase):

    def init_predictor(self, low_precision_io: bool):
        if False:
            return 10
        config = Config(os.path.join(self.temp_dir.name, 'alexnet/inference.pdmodel'), os.path.join(self.temp_dir.name, 'alexnet/inference.pdiparams'))
        config.enable_use_gpu(256, 0, PrecisionType.Half)
        config.enable_tensorrt_engine(workspace_size=1 << 30, max_batch_size=1, min_subgraph_size=3, precision_mode=PrecisionType.Half, use_static=False, use_calib_mode=False)
        config.enable_tuned_tensorrt_dynamic_shape()
        config.enable_memory_optim()
        config.enable_low_precision_io(low_precision_io)
        config.disable_glog_info()
        predictor = create_predictor(config)
        return predictor

class TestEnableLowPrecisionIOWithTRTSubGraph(TestEnableLowPrecisionIO, unittest.TestCase):

    def init_predictor(self, low_precision_io: bool):
        if False:
            return 10
        config = Config(os.path.join(self.temp_dir.name, 'alexnet/inference.pdmodel'), os.path.join(self.temp_dir.name, 'alexnet/inference.pdiparams'))
        config.enable_use_gpu(256, 0, PrecisionType.Half)
        config.enable_tensorrt_engine(workspace_size=1 << 30, max_batch_size=1, min_subgraph_size=3, precision_mode=PrecisionType.Half, use_static=False, use_calib_mode=False)
        config.enable_tuned_tensorrt_dynamic_shape()
        config.enable_memory_optim()
        config.enable_low_precision_io(low_precision_io)
        config.exp_disable_tensorrt_ops(['flatten_contiguous_range'])
        config.disable_glog_info()
        predictor = create_predictor(config)
        return predictor
if __name__ == '__main__':
    unittest.main()