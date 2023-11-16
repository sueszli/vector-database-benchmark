import os
import tempfile
import unittest
import numpy as np
import paddle
from paddle.inference import Config, DataType, create_predictor
paddle.set_default_dtype('float64')

class TestNet(paddle.nn.Layer):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.fc1 = paddle.nn.Linear(4, 4)
        self.fc2 = paddle.nn.Linear(4, 4)

    def forward(self, x):
        if False:
            print('Hello World!')
        out = self.fc1(x)
        out = self.fc2(out)
        out = paddle.nn.functional.relu(out)
        return out

@unittest.skipIf(not paddle.is_compiled_with_cuda(), 'should compile with cuda.')
class TestDoubleOnGPU(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.temp_dir = tempfile.TemporaryDirectory()
        net = TestNet()
        model = paddle.jit.to_static(net, input_spec=[paddle.static.InputSpec(shape=[None, 4], dtype='float64')])
        paddle.jit.save(model, os.path.join(self.temp_dir.name, 'test_inference_datatype_model/inference'))

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        self.temp_dir.cleanup()

    def init_predictor(self):
        if False:
            for i in range(10):
                print('nop')
        config = Config(os.path.join(self.temp_dir.name, 'test_inference_datatype_model/inference.pdmodel'), os.path.join(self.temp_dir.name, 'test_inference_datatype_model/inference.pdiparams'))
        config.enable_use_gpu(256, 0)
        config.enable_memory_optim()
        config.switch_ir_optim(False)
        predictor = create_predictor(config)
        return predictor

    def test_output(self):
        if False:
            return 10
        predictor = self.init_predictor()
        input = np.ones((3, 4)).astype(np.float64)
        input_names = predictor.get_input_names()
        input_tensor = predictor.get_input_handle(input_names[0])
        input_tensor.reshape(input.shape)
        input_tensor.copy_from_cpu(input.copy())
        assert input_tensor.type() == DataType.FLOAT64
        predictor.run()
        output_names = predictor.get_output_names()
        output_tensor = predictor.get_output_handle(output_names[0])
        assert output_tensor.type() == DataType.FLOAT64
        output_data = output_tensor.copy_to_cpu()
if __name__ == '__main__':
    unittest.main()