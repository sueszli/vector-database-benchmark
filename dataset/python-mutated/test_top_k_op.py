import unittest
import numpy as np
from op_test import OpTest
import paddle

class TestTopkOp(OpTest):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.variable_k = False
        self.set_args()
        self.op_type = 'top_k'
        self.dtype = np.float64
        self.check_cinn = True
        self.init_dtype()
        k = self.top_k
        input = np.random.random((self.row, k)).astype(self.dtype)
        output = np.ndarray((self.row, k))
        indices = np.ndarray((self.row, k)).astype('int64')
        self.inputs = {'X': input}
        if self.variable_k:
            self.inputs['K'] = np.array([k]).astype('int32')
        else:
            self.attrs = {'k': k}
        for rowid in range(self.row):
            row = input[rowid]
            output[rowid] = np.sort(row)[::-1][:k]
            indices[rowid] = row.argsort()[::-1][:k]
        self.outputs = {'Out': output, 'Indices': indices}

    def init_dtype(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def set_args(self):
        if False:
            while True:
                i = 10
        self.row = 100
        self.top_k = 1

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_output(check_cinn=self.check_cinn)

    def test_check_grad(self):
        if False:
            i = 10
            return i + 15
        self.check_grad({'X'}, 'Out', check_cinn=self.check_cinn)
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()