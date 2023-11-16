import unittest
import numpy as np
from op_test import OpTest

class TestFillZerosLikeOp(OpTest):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.op_type = 'fill_zeros_like'
        self.dtype = np.float32
        self.init_dtype()
        self.inputs = {'X': np.random.random((219, 232)).astype(self.dtype)}
        self.outputs = {'Out': np.zeros_like(self.inputs['X'])}

    def init_dtype(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def test_check_output(self):
        if False:
            print('Hello World!')
        self.check_output()

class TestFillZerosLikeOpFp16(TestFillZerosLikeOp):

    def init_dtype(self):
        if False:
            while True:
                i = 10
        self.dtype = np.float16
if __name__ == '__main__':
    unittest.main()