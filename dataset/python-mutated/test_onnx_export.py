import unittest
import numpy as np
import paddle

class LinearNet(paddle.nn.Layer):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self._linear = paddle.nn.Linear(128, 10)

    def forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        return self._linear(x)

class Logic(paddle.nn.Layer):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()

    def forward(self, x, y, z):
        if False:
            for i in range(10):
                print('nop')
        if z:
            return x
        else:
            return y

class TestExportWithTensor(unittest.TestCase):

    def test_with_tensor(self):
        if False:
            return 10
        self.x_spec = paddle.static.InputSpec(shape=[None, 128], dtype='float32')
        model = LinearNet()
        paddle.onnx.export(model, 'linear_net', input_spec=[self.x_spec])

class TestExportWithTensor1(unittest.TestCase):

    def test_with_tensor(self):
        if False:
            return 10
        self.x = paddle.to_tensor(np.random.random((1, 128)))
        model = LinearNet()
        paddle.onnx.export(model, 'linear_net', input_spec=[self.x])

class TestExportPrunedGraph(unittest.TestCase):

    def test_prune_graph(self):
        if False:
            return 10
        model = Logic()
        self.x = paddle.to_tensor(np.array([1]))
        self.y = paddle.to_tensor(np.array([-1]))
        paddle.jit.to_static(model)
        out = model(self.x, self.y, z=True)
        paddle.onnx.export(model, 'pruned', input_spec=[self.x, self.y, True], output_spec=[out], input_names_after_prune=[self.x.name])
if __name__ == '__main__':
    unittest.main()