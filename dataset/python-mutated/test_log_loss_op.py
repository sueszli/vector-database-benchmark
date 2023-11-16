import unittest
import numpy as np
from op_test import OpTest
from paddle.nn import functional as F

def sigmoid_array(x):
    if False:
        print('Hello World!')
    return 1 / (1 + np.exp(-x))

class TestLogLossOp(OpTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.op_type = 'log_loss'
        self.python_api = F.log_loss
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
            i = 10
            return i + 15
        self.check_output()

    def test_check_grad(self):
        if False:
            while True:
                i = 10
        self.check_grad(['Predicted'], 'Loss', max_relative_error=0.03)
if __name__ == '__main__':
    unittest.main()