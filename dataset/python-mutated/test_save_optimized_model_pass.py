import os
import tempfile
import unittest
import numpy as np
import paddle
from paddle.inference import Config, PrecisionType, create_predictor
from paddle.jit import to_static
from paddle.static import InputSpec
from paddle.vision.models import alexnet

class TestSaveOptimizedModelPass:

    def setUp(self):
        if False:
            return 10
        self.temp_dir = tempfile.TemporaryDirectory()
        net = alexnet(True)
        model = to_static(net, input_spec=[InputSpec(shape=[None, 3, 224, 224], name='x')])
        paddle.jit.save(model, os.path.join(self.temp_dir.name, 'alexnet/inference'))

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        self.temp_dir.cleanup()

    def get_baseline(self):
        if False:
            for i in range(10):
                print('nop')
        predictor = self.init_predictor(save_optimized_model=True)
        inputs = [paddle.to_tensor(0.1 * np.ones([1, 3, 224, 224]).astype(np.float32))]
        outputs = predictor.run(inputs)
        return outputs[0]

    def get_test_output(self):
        if False:
            i = 10
            return i + 15
        predictor = self.init_predictor(save_optimized_model=False)
        inputs = [paddle.to_tensor(0.1 * np.ones([1, 3, 224, 224]).astype(np.float32))]
        outputs = predictor.run(inputs)
        return outputs[0]

    def test_output(self):
        if False:
            print('Hello World!')
        if paddle.is_compiled_with_cuda():
            baseline = self.get_baseline()
            test_output = self.get_test_output()
            np.testing.assert_allclose(baseline.numpy().flatten(), test_output.numpy().flatten())

class TestSaveOptimizedModelPassWithGPU(TestSaveOptimizedModelPass, unittest.TestCase):

    def init_predictor(self, save_optimized_model: bool):
        if False:
            for i in range(10):
                print('nop')
        if save_optimized_model is True:
            config = Config(os.path.join(self.temp_dir.name, 'alexnet/inference.pdmodel'), os.path.join(self.temp_dir.name, 'alexnet/inference.pdiparams'))
            config.enable_use_gpu(256, 0, PrecisionType.Half)
            config.enable_memory_optim()
            config.switch_ir_optim(True)
            config.set_optim_cache_dir(os.path.join(self.temp_dir.name, 'alexnet'))
            config.enable_save_optim_model(True)
        else:
            config = Config(os.path.join(self.temp_dir.name, 'alexnet/_optimized.pdmodel'), os.path.join(self.temp_dir.name, 'alexnet/_optimized.pdiparams'))
            config.enable_use_gpu(256, 0, PrecisionType.Half)
            config.enable_memory_optim()
            config.switch_ir_optim(False)
        predictor = create_predictor(config)
        return predictor

class TestSaveOptimizedModelPassWithTRT(TestSaveOptimizedModelPass, unittest.TestCase):

    def init_predictor(self, save_optimized_model: bool):
        if False:
            while True:
                i = 10
        if save_optimized_model is True:
            config = Config(os.path.join(self.temp_dir.name, 'alexnet/inference.pdmodel'), os.path.join(self.temp_dir.name, 'alexnet/inference.pdiparams'))
            config.enable_use_gpu(256, 0)
            config.enable_tensorrt_engine(workspace_size=1 << 30, max_batch_size=1, min_subgraph_size=3, precision_mode=PrecisionType.Half, use_static=True, use_calib_mode=False)
            config.set_trt_dynamic_shape_info({'x': [1, 3, 224, 224], 'flatten_0.tmp_0': [1, 9216]}, {'x': [1, 3, 224, 224], 'flatten_0.tmp_0': [1, 9216]}, {'x': [1, 3, 224, 224], 'flatten_0.tmp_0': [1, 9216]})
            config.exp_disable_tensorrt_ops(['flatten_contiguous_range'])
            config.enable_memory_optim()
            config.switch_ir_optim(True)
            config.set_optim_cache_dir(os.path.join(self.temp_dir.name, 'alexnet'))
            config.enable_save_optim_model(True)
        else:
            config = Config(os.path.join(self.temp_dir.name, 'alexnet/_optimized.pdmodel'), os.path.join(self.temp_dir.name, 'alexnet/_optimized.pdiparams'))
            config.enable_use_gpu(256, 0)
            config.enable_tensorrt_engine(workspace_size=1 << 30, max_batch_size=1, min_subgraph_size=3, precision_mode=PrecisionType.Half, use_static=True, use_calib_mode=False)
            config.enable_memory_optim()
            config.switch_ir_optim(False)
        predictor = create_predictor(config)
        return predictor
if __name__ == '__main__':
    unittest.main()