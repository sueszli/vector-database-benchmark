import unittest
import numpy as np
from op_test import OpTest

class TestHingeLossOp(OpTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.op_type = 'hinge_loss'
        samples_num = 100
        logits = np.random.uniform(-10, 10, (samples_num, 1)).astype('float32')
        labels = np.random.randint(0, 2, (samples_num, 1)).astype('float32')
        self.inputs = {'Logits': logits, 'Labels': labels}
        loss = np.maximum(1.0 - (2 * labels - 1) * logits, 0)
        self.outputs = {'Loss': loss}

    def test_check_output(self):
        if False:
            while True:
                i = 10
        self.check_output()

    def test_check_grad(self):
        if False:
            i = 10
            return i + 15
        self.check_grad(['Logits'], 'Loss')
if __name__ == '__main__':
    unittest.main()