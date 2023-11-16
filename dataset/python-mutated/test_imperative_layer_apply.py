import unittest
import numpy as np
import paddle
from paddle import base, nn

class LeNetDygraph(paddle.nn.Layer):

    def __init__(self, num_classes=10, classifier_activation='softmax'):
        if False:
            print('Hello World!')
        super().__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(nn.Conv2D(1, 6, 3, stride=1, padding=1), nn.ReLU(), paddle.nn.MaxPool2D(2, 2), nn.Conv2D(6, 16, 5, stride=1, padding=0), nn.ReLU(), paddle.nn.MaxPool2D(2, 2))
        if num_classes > 0:
            self.fc = nn.Sequential(nn.Linear(400, 120), nn.Linear(120, 84), nn.Linear(84, 10), nn.Softmax())

    def forward(self, inputs):
        if False:
            i = 10
            return i + 15
        x = self.features(inputs)
        if self.num_classes > 0:
            x = paddle.flatten(x, 1, -1)
            x = self.fc(x)
        return x

def init_weights(layer):
    if False:
        print('Hello World!')
    if type(layer) == nn.Linear:
        new_weight = paddle.tensor.fill_constant(layer.weight.shape, layer.weight.dtype, value=0.9)
        layer.weight.set_value(new_weight)
        new_bias = paddle.tensor.fill_constant(layer.bias.shape, layer.bias.dtype, value=-0.1)
        layer.bias.set_value(new_bias)
    elif type(layer) == nn.Conv2D:
        new_weight = paddle.tensor.fill_constant(layer.weight.shape, layer.weight.dtype, value=0.7)
        layer.weight.set_value(new_weight)
        new_bias = paddle.tensor.fill_constant(layer.bias.shape, layer.bias.dtype, value=-0.2)
        layer.bias.set_value(new_bias)

class TestLayerApply(unittest.TestCase):

    def test_apply_init_weight(self):
        if False:
            print('Hello World!')
        with base.dygraph.guard():
            net = LeNetDygraph()
            net.apply(init_weights)
            for layer in net.sublayers():
                if type(layer) == nn.Linear:
                    np.testing.assert_allclose(layer.weight.numpy(), 0.9)
                    np.testing.assert_allclose(layer.bias.numpy(), -0.1)
                elif type(layer) == nn.Conv2D:
                    np.testing.assert_allclose(layer.weight.numpy(), 0.7)
                    np.testing.assert_allclose(layer.bias.numpy(), -0.2)
if __name__ == '__main__':
    unittest.main()