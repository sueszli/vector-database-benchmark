import unittest
import numpy as np
from op_test import OpTest
from paddle.base.framework import convert_np_dtype_to_dtype_

class TestFillZerosLike2Op(OpTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.op_type = 'fill_zeros_like2'
        self.dtype = np.float32
        self.init_dtype()
        self.inputs = {'X': np.random.random((219, 232)).astype(self.dtype)}
        self.outputs = {'Out': np.zeros_like(self.inputs['X'])}
        self.attrs = {'dtype': convert_np_dtype_to_dtype_(self.dtype)}

    def init_dtype(self):
        if False:
            return 10
        pass

    def test_check_output(self):
        if False:
            print('Hello World!')
        self.check_output()

class TestFillZerosLike2OpFp16(TestFillZerosLike2Op):

    def init_dtype(self):
        if False:
            while True:
                i = 10
        self.dtype = np.float16

class TestFillZerosLike2OpFp64(TestFillZerosLike2Op):

    def init_dtype(self):
        if False:
            while True:
                i = 10
        self.dtype = np.float64
if __name__ == '__main__':
    unittest.main()