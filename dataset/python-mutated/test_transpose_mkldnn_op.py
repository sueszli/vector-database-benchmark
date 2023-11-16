import unittest
import numpy as np
from op_test import OpTest

class TestTransposeMKLDNN(OpTest):

    def setUp(self):
        if False:
            print('Hello World!')
        self.init_op_type()
        self.initTestCase()
        self.inputs = {'X': np.random.random(self.shape).astype('float32')}
        self.attrs = {'axis': list(self.axis), 'use_mkldnn': self.use_mkldnn}
        self.outputs = {'XShape': np.random.random(self.shape).astype('float32'), 'Out': self.inputs['X'].transpose(self.axis)}

    def init_op_type(self):
        if False:
            return 10
        self.op_type = 'transpose2'
        self.use_mkldnn = True

    def test_check_output(self):
        if False:
            while True:
                i = 10
        self.check_output(no_check_set=['XShape'], check_dygraph=False)

    def test_check_grad(self):
        if False:
            return 10
        self.check_grad(['X'], 'Out', check_dygraph=False)

    def initTestCase(self):
        if False:
            i = 10
            return i + 15
        self.shape = (30, 4)
        self.axis = (1, 0)

class TestCase0MKLDNN(TestTransposeMKLDNN):

    def initTestCase(self):
        if False:
            i = 10
            return i + 15
        self.shape = (100,)
        self.axis = (0,)

class TestCase1a(TestTransposeMKLDNN):

    def initTestCase(self):
        if False:
            return 10
        self.shape = (3, 4, 10)
        self.axis = (0, 2, 1)

class TestCase1b(TestTransposeMKLDNN):

    def initTestCase(self):
        if False:
            i = 10
            return i + 15
        self.shape = (3, 4, 10)
        self.axis = (2, 1, 0)

class TestCase2(TestTransposeMKLDNN):

    def initTestCase(self):
        if False:
            print('Hello World!')
        self.shape = (2, 3, 4, 5)
        self.axis = (0, 2, 3, 1)

class TestCase3(TestTransposeMKLDNN):

    def initTestCase(self):
        if False:
            for i in range(10):
                print('nop')
        self.shape = (2, 3, 4, 5, 6)
        self.axis = (4, 2, 3, 1, 0)

class TestCase4(TestTransposeMKLDNN):

    def initTestCase(self):
        if False:
            i = 10
            return i + 15
        self.shape = (2, 3, 4, 5, 6, 1)
        self.axis = (4, 2, 3, 1, 0, 5)

class TestCase_ZeroDim(TestTransposeMKLDNN):

    def initTestCase(self):
        if False:
            while True:
                i = 10
        self.shape = ()
        self.axis = ()
if __name__ == '__main__':
    unittest.main()