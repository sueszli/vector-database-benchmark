import unittest
import numpy as np
import paddle
from paddle import base
from paddle.base import dygraph

class TestImperativeLayerTrainable(unittest.TestCase):

    def test_set_trainable(self):
        if False:
            while True:
                i = 10
        with base.dygraph.guard():
            label = np.random.uniform(-1, 1, [10, 10]).astype(np.float32)
            label = dygraph.to_variable(label)
            linear = paddle.nn.Linear(10, 10)
            y = linear(label)
            self.assertFalse(y.stop_gradient)
            linear.weight.trainable = False
            linear.bias.trainable = False
            self.assertFalse(linear.weight.trainable)
            self.assertTrue(linear.weight.stop_gradient)
            y = linear(label)
            self.assertTrue(y.stop_gradient)
            with self.assertRaises(ValueError):
                linear.weight.trainable = '1'
if __name__ == '__main__':
    unittest.main()