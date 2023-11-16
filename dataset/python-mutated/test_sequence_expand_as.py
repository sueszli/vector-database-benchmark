import unittest
import numpy as np
from op_test import OpTest
import paddle
from paddle.base import Program, program_guard

class TestSequenceExpandAs(OpTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.op_type = 'sequence_expand_as'
        self.set_data()
        self.compute()

    def set_data(self):
        if False:
            i = 10
            return i + 15
        x_data = np.random.uniform(0.1, 1, [3, 40]).astype('float64')
        y_data = np.random.uniform(0.1, 1, [8, 1]).astype('float64')
        y_lod = [[1, 3, 4]]
        self.inputs = {'X': x_data, 'Y': (y_data, y_lod)}

    def compute(self):
        if False:
            for i in range(10):
                print('nop')
        x = self.inputs['X']
        (x_data, x_lod) = x if type(x) == tuple else (x, None)
        (y_data, y_lod) = self.inputs['Y']
        assert len(y_lod) == 1 and len(y_lod[0]) == x_data.shape[0]
        repeats = []
        for i in range(len(y_lod[0])):
            repeat_num = y_lod[0][i]
            if repeat_num == 0:
                continue
            repeats.extend([i for _ in range(repeat_num)])
        out_data = x_data[repeats]
        self.outputs = {'Out': (out_data, y_lod)}

    def test_check_output(self):
        if False:
            while True:
                i = 10
        self.check_output(check_dygraph=False)

    def test_check_grad(self):
        if False:
            while True:
                i = 10
        self.check_grad(['X'], 'Out', check_dygraph=False)

class TestSequenceExpandAsCase1(TestSequenceExpandAs):

    def set_data(self):
        if False:
            while True:
                i = 10
        x_data = np.random.uniform(0.1, 1, [5, 20]).astype('float64')
        x_lod = [[2, 3]]
        y_data = np.random.uniform(0.1, 1, [10, 1]).astype('float64')
        y_lod = [[2, 2, 0, 3, 3]]
        self.inputs = {'X': (x_data, x_lod), 'Y': (y_data, y_lod)}

class TestSequenceExpandAsCase2(TestSequenceExpandAs):

    def set_data(self):
        if False:
            for i in range(10):
                print('nop')
        x_data = np.random.uniform(0.1, 1, [5, 20]).astype('float64')
        x_lod = [[2, 3]]
        y_data = np.random.uniform(0.1, 1, [10, 1]).astype('float64')
        y_lod = [[0, 4, 0, 6, 0]]
        self.inputs = {'X': (x_data, x_lod), 'Y': (y_data, y_lod)}

class TestSequenceExpandAsCase3(TestSequenceExpandAs):

    def set_data(self):
        if False:
            return 10
        x_data = np.random.uniform(0.1, 1, [1, 2, 50]).astype('float64')
        x_lod = [[1]]
        y_data = np.random.uniform(0.1, 1, [2, 2, 2]).astype('float64')
        y_lod = [[2]]
        self.inputs = {'X': (x_data, x_lod), 'Y': (y_data, y_lod)}

class TestSequenceExpandAsOpError(unittest.TestCase):

    def test_errors(self):
        if False:
            i = 10
            return i + 15
        with program_guard(Program(), Program()):
            x1 = np.random.random((2, 4)).astype('float32')
            self.assertRaises(TypeError, paddle.static.nn.sequence_lod.sequence_expand_as, x1)
            x2 = paddle.static.data(name='x2', shape=[None, 4], dtype='bool')
            self.assertRaises(TypeError, paddle.static.nn.sequence_lod.sequence_expand_as, x2)
            x3 = paddle.static.data(name='x3', shape=[None, 4], dtype='float32')
            y = np.random.random((2, 4)).astype('float32')
            self.assertRaises(TypeError, paddle.static.nn.sequence_lod.sequence_expand_as, x3, y)
if __name__ == '__main__':
    unittest.main()