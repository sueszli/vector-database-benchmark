import unittest
import numpy as np
from op_test import OpTest

class TestL1NormOp(OpTest):
    """Test l1_norm"""

    def setUp(self):
        if False:
            return 10
        self.op_type = 'l1_norm'
        self.max_relative_error = 0.005
        X = np.random.uniform(-1, 1, (13, 19)).astype('float32')
        X[np.abs(X) < self.max_relative_error] = 0.1
        self.inputs = {'X': X}
        self.outputs = {'Out': np.sum(np.abs(X))}

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_output()

    def test_check_grad(self):
        if False:
            while True:
                i = 10
        self.check_grad(['X'], 'Out')
if __name__ == '__main__':
    unittest.main()