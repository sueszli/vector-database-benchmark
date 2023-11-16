import os
import tempfile
import unittest
import numpy as np
import paddle
from paddle.base.framework import _dygraph_place_guard
from paddle.jit.layer import Layer
from paddle.static import InputSpec
paddle.seed(1)

class Net(paddle.nn.Layer):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self.fc1 = paddle.nn.Linear(4, 4)
        self.fc2 = paddle.nn.Linear(4, 4)
        self._bias = 0.4

    @paddle.jit.to_static(input_spec=[InputSpec([None, 4], dtype='float32')])
    def forward(self, x):
        if False:
            print('Hello World!')
        out = self.fc1(x)
        out = self.fc2(out)
        out = paddle.nn.functional.relu(out)
        out = paddle.mean(out)
        return out

    @paddle.jit.to_static(input_spec=[InputSpec([None, 4], dtype='float32')])
    def infer(self, input):
        if False:
            for i in range(10):
                print('nop')
        out = self.fc2(input)
        out = out + self._bias
        out = paddle.mean(out)
        return out

class TestMultiLoad(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        if False:
            print('Hello World!')
        self.temp_dir.cleanup()

    def test_multi_load(self):
        if False:
            for i in range(10):
                print('nop')
        x = paddle.full([2, 4], 2)
        model = Net()
        paddle.jit.enable_to_static(False)
        forward_out1 = model.forward(x)
        infer_out1 = model.infer(x)
        paddle.jit.enable_to_static(True)
        model_path = os.path.join(self.temp_dir.name, 'multi_program')
        paddle.jit.save(model, model_path, combine_params=True)
        place = paddle.CPUPlace()
        if paddle.is_compiled_with_cuda():
            place = paddle.CUDAPlace(0)
        jit_layer = Layer()
        jit_layer.load(model_path, place)
        forward_out2 = jit_layer.forward(x)
        infer_out2 = jit_layer.infer(x)
        np.testing.assert_allclose(forward_out1, forward_out2[0], rtol=1e-05)
        np.testing.assert_allclose(infer_out1, infer_out2[0], rtol=1e-05)

class SaveLinear(paddle.nn.Layer):

    def __init__(self):
        if False:
            return 10
        super().__init__()
        self.linear = paddle.nn.Linear(80, 80)

    @paddle.jit.to_static(input_spec=[InputSpec(shape=[None, 80], dtype='float32')])
    def forward(self, x):
        if False:
            i = 10
            return i + 15
        out = self.linear(x)
        return out

class TestMKLOutput(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        self.temp_dir.cleanup()

    def test_mkl_output(self):
        if False:
            for i in range(10):
                print('nop')
        with _dygraph_place_guard(place=paddle.CPUPlace()):
            net = SaveLinear()
            model_path = os.path.join(self.temp_dir.name, 'save_linear')
            paddle.jit.save(net, model_path, combine_params=True)
            layer = Layer()
            layer.load(model_path, paddle.CPUPlace())
            x = paddle.ones([498, 80])
            out = layer.forward(x)
            out = paddle.unsqueeze(out[0], 0)
            np.testing.assert_equal(out.shape, [1, 498, 80])
if __name__ == '__main__':
    unittest.main()