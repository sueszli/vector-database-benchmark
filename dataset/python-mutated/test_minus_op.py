import unittest
import numpy as np
from op_test import OpTest
import paddle

class TestMinusOp(OpTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_type = 'minus'
        self.inputs = {'X': np.random.random((32, 84)).astype('float32'), 'Y': np.random.random((32, 84)).astype('float32')}
        self.outputs = {'Out': self.inputs['X'] - self.inputs['Y']}

    def test_check_output(self):
        if False:
            return 10
        self.check_output(check_dygraph=False)

    def test_check_grad(self):
        if False:
            print('Hello World!')
        self.check_grad(['X', 'Y'], 'Out', check_dygraph=False)
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()