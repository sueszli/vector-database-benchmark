import os
import unittest
import numpy as np
import paddle
from paddle import base
paddle.enable_static()
np.random.seed(123)
os.environ['CPU_NUM'] = '1'
base.core._set_eager_deletion_mode(0.0, 1.0, True)

class RecurrentNet(paddle.nn.Layer):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()
        self.cell = paddle.nn.SimpleRNNCell(16, 32)
        self.rnn = paddle.nn.RNN(self.cell)

    def forward(self, inputs, prev_h):
        if False:
            for i in range(10):
                print('nop')
        (outputs, final_states) = self.rnn(inputs, prev_h)
        return (outputs, final_states)

class TestDy2StRecurrentOpBackward(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        paddle.disable_static()
        paddle.seed(100)

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        paddle.enable_static()

    def test_recurrent_backward(self):
        if False:
            for i in range(10):
                print('nop')
        net = RecurrentNet()
        inputs = paddle.rand((4, 23, 16))
        inputs.stop_gradient = False
        prev_h = paddle.randn((4, 32))
        prev_h.stop_gradient = False
        (outputs, final_states) = net(inputs, prev_h)
        outputs.backward()
        dy_grad = inputs.gradient()
        inputs.clear_gradient()
        net = paddle.jit.to_static(net)
        (outputs, final_states) = net(inputs, prev_h)
        outputs.backward()
        st_grad = inputs.gradient()
        np.testing.assert_allclose(dy_grad, st_grad)
if __name__ == '__main__':
    unittest.main()