import random
import unittest
import numpy as np
from op_test import OpTest

class TestPartialSumOp(OpTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.op_type = 'partial_sum'
        self.init_kernel_type()
        self.init_para()
        if self.length is -1:
            end_index = self.column
        else:
            end_index = self.start_index + self.length
        self.var_names = ['x' + str(num) for num in range(self.var_num)]
        self.vars = [np.random.random((self.batch_size, self.column)).astype(self.dtype) for num in range(self.var_num)]
        self.inputs = {'X': list(zip(self.var_names, self.vars))}
        self.attrs = {'start_index': self.start_index, 'length': self.length}
        y = self.vars[0][:, self.start_index:end_index]
        for i in range(1, self.var_num):
            y = y + self.vars[i][:, self.start_index:end_index]
        self.outputs = {'Out': y}

    def init_kernel_type(self):
        if False:
            print('Hello World!')
        self.dtype = np.float64

    def init_para(self):
        if False:
            for i in range(10):
                print('nop')
        self.batch_size = random.randint(10, 20)
        self.column = random.randint(101, 200)
        self.start_index = random.randint(0, self.column - 1)
        self.length = random.randint(0, self.column - self.start_index)
        self.var_num = random.randint(1, 3)

    def test_check_output(self):
        if False:
            print('Hello World!')
        self.check_output()

    def test_check_grad(self):
        if False:
            i = 10
            return i + 15
        for var_name in self.var_names:
            self.check_grad([var_name], 'Out')

class TestPartialSumOp2(TestPartialSumOp):

    def init_para(self):
        if False:
            for i in range(10):
                print('nop')
        self.batch_size = random.randint(1, 10)
        self.column = random.randint(101, 200)
        self.start_index = random.randint(0, self.column - 1)
        self.length = -1
        self.var_num = 3

class TestPartialSumOp3(TestPartialSumOp):

    def init_para(self):
        if False:
            print('Hello World!')
        self.batch_size = random.randint(1, 10)
        self.column = random.randint(101, 200)
        self.start_index = self.column - 1
        self.length = 1
        self.var_num = 2

class TestPartialSumOp4(TestPartialSumOp):

    def init_para(self):
        if False:
            i = 10
            return i + 15
        self.batch_size = random.randint(1, 10)
        self.column = random.randint(101, 200)
        self.start_index = self.column - 1
        self.length = 1
        self.var_num = 1
if __name__ == '__main__':
    unittest.main()