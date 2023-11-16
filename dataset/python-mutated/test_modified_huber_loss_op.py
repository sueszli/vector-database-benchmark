import unittest
import numpy as np
from op_test import OpTest

def modified_huber_loss_forward(val):
    if False:
        for i in range(10):
            print('nop')
    if val < -1:
        return -4.0 * val
    elif val < 1:
        return (1.0 - val) * (1.0 - val)
    else:
        return 0.0

class TestModifiedHuberLossOp(OpTest):

    def setUp(self):
        if False:
            print('Hello World!')
        self.op_type = 'modified_huber_loss'
        samples_num = 100
        x_np = np.random.uniform(-2.0, 2.0, (samples_num, 1)).astype('float32')
        y_np = np.random.choice([0, 1], samples_num).reshape((samples_num, 1)).astype('float32')
        product_res = x_np * (2.0 * y_np - 1.0)
        for (pos, val) in np.ndenumerate(product_res):
            while abs(val - 1.0) < 0.05:
                x_np[pos] = np.random.uniform(-2.0, 2.0)
                y_np[pos] = np.random.choice([0, 1])
                product_res[pos] = x_np[pos] * (2 * y_np[pos] - 1)
                val = product_res[pos]
        self.inputs = {'X': x_np, 'Y': y_np}
        loss = np.vectorize(modified_huber_loss_forward)(product_res)
        self.outputs = {'IntermediateVal': product_res.astype('float32'), 'Out': loss.reshape((samples_num, 1)).astype('float32')}

    def test_check_output(self):
        if False:
            print('Hello World!')
        self.check_output()

    def test_check_grad(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_grad(['X'], 'Out')
if __name__ == '__main__':
    unittest.main()