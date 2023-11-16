import unittest
import numpy as np
import paddle
from paddle import base, nn

class LeNetDygraph(paddle.nn.Layer):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self.features = nn.Sequential(nn.Conv2D(1, 6, 3, stride=1, padding=1), nn.ReLU(), paddle.nn.MaxPool2D(2, 2), nn.Conv2D(6, 16, 5, stride=1, padding=0), nn.ReLU(), paddle.nn.MaxPool2D(2, 2))

    def forward(self, inputs):
        if False:
            for i in range(10):
                print('nop')
        x = self.features(inputs)
        return x

class TestLayerChildren(unittest.TestCase):

    def func_apply_init_weight(self):
        if False:
            print('Hello World!')
        with base.dygraph.guard():
            net = LeNetDygraph()
            net.eval()
            net_layers = nn.Sequential(*list(net.children()))
            net_layers.eval()
            x = paddle.rand([2, 1, 28, 28])
            y1 = net(x)
            y2 = net_layers(x)
            np.testing.assert_allclose(y1.numpy(), y2.numpy())
            return (y1, y2)

    def test_func_apply_init_weight(self):
        if False:
            print('Hello World!')
        paddle.seed(102)
        (self.new_y1, self.new_y2) = self.func_apply_init_weight()
        paddle.seed(102)
        (self.ori_y1, self.ori_y2) = self.func_apply_init_weight()
        np.testing.assert_array_equal(self.ori_y1.numpy(), self.new_y1.numpy())
        np.testing.assert_array_equal(self.ori_y2.numpy(), self.new_y2.numpy())
if __name__ == '__main__':
    unittest.main()