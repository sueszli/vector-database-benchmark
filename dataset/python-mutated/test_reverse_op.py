import unittest
import numpy as np
from op_test import OpTest
import paddle

class TestReverseOp(OpTest):

    def initTestCase(self):
        if False:
            for i in range(10):
                print('nop')
        self.x = np.random.random((3, 40)).astype('float64')
        self.axis = [0]

    def setUp(self):
        if False:
            print('Hello World!')
        self.initTestCase()
        self.op_type = 'reverse'
        self.python_api = paddle.reverse
        self.inputs = {'X': self.x}
        self.attrs = {'axis': self.axis}
        out = self.x
        for a in self.axis:
            out = np.flip(out, axis=a)
        self.outputs = {'Out': out}

    def test_check_output(self):
        if False:
            print('Hello World!')
        self.check_output(check_cinn=True)

    def test_check_grad(self):
        if False:
            i = 10
            return i + 15
        self.check_grad(['X'], 'Out', check_cinn=True)

class TestCase0(TestReverseOp):

    def initTestCase(self):
        if False:
            i = 10
            return i + 15
        self.x = np.random.random((3, 40)).astype('float64')
        self.axis = [1]

class TestCase0_neg(TestReverseOp):

    def initTestCase(self):
        if False:
            while True:
                i = 10
        self.x = np.random.random((3, 40)).astype('float64')
        self.axis = [-1]

class TestCase1(TestReverseOp):

    def initTestCase(self):
        if False:
            return 10
        self.x = np.random.random((3, 40)).astype('float64')
        self.axis = [0, 1]

class TestCase1_neg(TestReverseOp):

    def initTestCase(self):
        if False:
            for i in range(10):
                print('nop')
        self.x = np.random.random((3, 40)).astype('float64')
        self.axis = [0, -1]

class TestCase2(TestReverseOp):

    def initTestCase(self):
        if False:
            for i in range(10):
                print('nop')
        self.x = np.random.random((3, 4, 10)).astype('float64')
        self.axis = [0, 2]

class TestCase2_neg(TestReverseOp):

    def initTestCase(self):
        if False:
            return 10
        self.x = np.random.random((3, 4, 10)).astype('float64')
        self.axis = [0, -2]

class TestCase3(TestReverseOp):

    def initTestCase(self):
        if False:
            for i in range(10):
                print('nop')
        self.x = np.random.random((3, 4, 10)).astype('float64')
        self.axis = [1, 2]

class TestCase3_neg(TestReverseOp):

    def initTestCase(self):
        if False:
            while True:
                i = 10
        self.x = np.random.random((3, 4, 10)).astype('float64')
        self.axis = [-1, -2]
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()