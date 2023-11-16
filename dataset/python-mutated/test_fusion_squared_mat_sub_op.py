import unittest
import numpy as np
from op_test import OpTest

class TestFusionSquaredMatSubOp(OpTest):

    def setUp(self):
        if False:
            print('Hello World!')
        self.op_type = 'fusion_squared_mat_sub'
        self.m = 11
        self.n = 12
        self.k = 4
        self.scalar = 0.5
        self.set_conf()
        matx = np.random.random((self.m, self.k)).astype('float32')
        maty = np.random.random((self.k, self.n)).astype('float32')
        self.inputs = {'X': matx, 'Y': maty}
        self.outputs = {'Out': (np.dot(matx, maty) ** 2 - np.dot(matx ** 2, maty ** 2)) * self.scalar}
        self.attrs = {'scalar': self.scalar}

    def set_conf(self):
        if False:
            print('Hello World!')
        pass

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        self.check_output(check_dygraph=False)

class TestFusionSquaredMatSubOpCase1(TestFusionSquaredMatSubOp):

    def set_conf(self):
        if False:
            return 10
        self.scalar = -0.3
if __name__ == '__main__':
    unittest.main()