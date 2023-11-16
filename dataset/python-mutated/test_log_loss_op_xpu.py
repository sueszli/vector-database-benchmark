import unittest
import numpy as np
from op_test import OpTest
import paddle

def sigmoid_array(x):
    if False:
        for i in range(10):
            print('nop')
    return 1 / (1 + np.exp(-x))

class TestXPULogLossOp(OpTest):

    def setUp(self):
        if False:
            print('Hello World!')
        self.op_type = 'log_loss'
        samples_num = 100
        x = np.random.random((samples_num, 1)).astype('float32')
        predicted = sigmoid_array(x)
        labels = np.random.randint(0, 2, (samples_num, 1)).astype('float32')
        epsilon = 1e-07
        self.inputs = {'Predicted': predicted, 'Labels': labels}
        self.attrs = {'epsilon': epsilon}
        loss = -labels * np.log(predicted + epsilon) - (1 - labels) * np.log(1 - predicted + epsilon)
        self.outputs = {'Loss': loss}

    def test_check_output(self):
        if False:
            return 10
        if paddle.is_compiled_with_xpu():
            paddle.enable_static()
            place = paddle.XPUPlace(0)
            self.check_output_with_place(place, check_dygraph=False)

    def test_check_grad(self):
        if False:
            print('Hello World!')
        if paddle.is_compiled_with_xpu():
            paddle.enable_static()
            place = paddle.XPUPlace(0)
            self.check_grad(['Predicted'], 'Loss', check_dygraph=False)
if __name__ == '__main__':
    unittest.main()