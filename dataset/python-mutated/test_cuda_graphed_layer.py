import unittest
import numpy as np
import paddle
from paddle import nn
from paddle.device.cuda.cuda_graphed_layer import CUDAGraphedLayer
seed = 102

class Model(nn.Layer):

    def __init__(self, in_size, out_size, dropout=0):
        if False:
            return 10
        paddle.seed(seed)
        super().__init__()
        self.linear = nn.Linear(in_size, out_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        x = self.linear(x)
        x = self.relu(x)
        return x

class DropoutModel(nn.Layer):

    def __init__(self, in_size, out_size, dropout=0.5):
        if False:
            print('Hello World!')
        paddle.seed(seed)
        super().__init__()
        self.linear = nn.Linear(in_size, out_size)
        self.dropout_1 = paddle.nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.dropout_2 = paddle.nn.Dropout(dropout)

    def forward(self, x):
        if False:
            i = 10
            return i + 15
        x = self.linear(x)
        x = self.dropout_1(x)
        x = self.relu(x)
        x = self.dropout_2(x)
        return x

@unittest.skipIf(not paddle.is_compiled_with_cuda() or float(paddle.version.cuda()) < 11.0, 'only support cuda >= 11.0')
class TestSimpleModel(unittest.TestCase):

    def train(self, model):
        if False:
            i = 10
            return i + 15
        paddle.seed(seed)
        ans = []
        for _ in range(10):
            x = paddle.randn([3, 10], dtype='float32')
            x.stop_gradient = False
            loss = model(x).mean()
            loss.backward()
            ans.append(x.grad.numpy())
        return np.array(ans)

    def test_layer(self):
        if False:
            for i in range(10):
                print('nop')
        model = Model(10, 20)
        cuda_graphed_model = CUDAGraphedLayer(Model(10, 20))
        dropout_model = DropoutModel(10, 20)
        cuda_graphed_dropout_model = CUDAGraphedLayer(DropoutModel(10, 20))
        np.testing.assert_array_equal(self.train(model), self.train(cuda_graphed_model))
        np.testing.assert_array_equal(self.train(dropout_model), self.train(cuda_graphed_dropout_model))
if __name__ == '__main__':
    unittest.main()