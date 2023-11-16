import unittest
import numpy as np
from op_test import OpTest

class TestFlattenOp(OpTest):

    def setUp(self):
        if False:
            print('Hello World!')
        self.op_type = 'flatten2'
        self.init_test_case()
        self.inputs = {'X': np.random.random(self.in_shape).astype('float64')}
        self.init_attrs()
        self.outputs = {'Out': self.inputs['X'].reshape(self.new_shape), 'XShape': np.random.random(self.in_shape).astype('float32')}

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_output(no_check_set=['XShape'])

    def test_check_grad(self):
        if False:
            print('Hello World!')
        self.check_grad(['X'], 'Out')

    def init_test_case(self):
        if False:
            while True:
                i = 10
        self.in_shape = (3, 2, 4, 5)
        self.axis = 1
        self.new_shape = (3, 40)

    def init_attrs(self):
        if False:
            print('Hello World!')
        self.attrs = {'axis': self.axis}

class TestFlattenOp_ZeroDim(TestFlattenOp):

    def init_test_case(self):
        if False:
            while True:
                i = 10
        self.in_shape = ()
        self.axis = 0
        self.new_shape = 1

class TestFlattenOp1(TestFlattenOp):

    def init_test_case(self):
        if False:
            for i in range(10):
                print('nop')
        self.in_shape = (3, 2, 5, 4)
        self.axis = 0
        self.new_shape = (1, 120)

class TestFlattenOpWithDefaultAxis(TestFlattenOp):

    def init_test_case(self):
        if False:
            return 10
        self.in_shape = (10, 2, 2, 3)
        self.new_shape = (10, 12)

    def init_attrs(self):
        if False:
            print('Hello World!')
        self.attrs = {}

class TestFlattenOpSixDims(TestFlattenOp):

    def init_test_case(self):
        if False:
            while True:
                i = 10
        self.in_shape = (3, 2, 3, 2, 4, 4)
        self.axis = 4
        self.new_shape = (36, 16)
if __name__ == '__main__':
    unittest.main()