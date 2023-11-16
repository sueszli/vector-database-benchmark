import unittest
import numpy as np
from op_test import OpTest
import paddle

class TestSequenceReverseBase(OpTest):

    def initParameters(self):
        if False:
            return 10
        pass

    def setUp(self):
        if False:
            while True:
                i = 10
        self.size = (10, 3, 4)
        self.lod = [2, 3, 5]
        self.dtype = 'float32'
        self.initParameters()
        self.op_type = 'sequence_reverse'
        self.x = np.random.random(self.size).astype(self.dtype)
        self.y = self.get_output()
        self.inputs = {'X': (self.x, [self.lod])}
        self.outputs = {'Y': (self.y, [self.lod])}

    def get_output(self):
        if False:
            return 10
        tmp_x = np.reshape(self.x, newshape=[self.x.shape[0], -1])
        tmp_y = np.ndarray(tmp_x.shape).astype(self.dtype)
        prev_idx = 0
        for cur_len in self.lod:
            idx_range = range(prev_idx, prev_idx + cur_len)
            tmp_y[idx_range, :] = np.flip(tmp_x[idx_range, :], 0)
            prev_idx += cur_len
        return np.reshape(tmp_y, newshape=self.x.shape).astype(self.dtype)

    def test_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_output(0, check_dygraph=False)

    def test_grad(self):
        if False:
            i = 10
            return i + 15
        self.check_grad(['X'], 'Y', check_dygraph=False)

class TestSequenceReserve1(TestSequenceReverseBase):

    def initParameters(self):
        if False:
            while True:
                i = 10
        self.size = (12, 10)
        self.lod = [4, 5, 3]

class TestSequenceReverse2(TestSequenceReverseBase):

    def initParameters(self):
        if False:
            i = 10
            return i + 15
        self.size = (12, 10)
        self.lod = [12]

class TestSequenceReverse3(TestSequenceReverseBase):

    def initParameters(self):
        if False:
            while True:
                i = 10
        self.size = (12, 10)
        self.lod = [3, 0, 6, 3]

class TestSequenceReverse4(TestSequenceReverseBase):

    def initParameters(self):
        if False:
            i = 10
            return i + 15
        self.size = (12, 10)
        self.lod = [0, 2, 10, 0]

class TestSequenceReverseOpError(unittest.TestCase):

    def test_error(self):
        if False:
            while True:
                i = 10

        def test_variable():
            if False:
                for i in range(10):
                    print('nop')
            x_data = np.random.random((2, 4)).astype('float32')
            paddle.static.nn.sequence_lod.sequence_reverse(x=x_data)
        self.assertRaises(TypeError, test_variable)

        def test_dtype():
            if False:
                while True:
                    i = 10
            x2_data = paddle.static.data(name='x2', shape=[-1, 4], dtype='float16')
            paddle.static.nn.sequence_lod.sequence_reverse(x=x2_data)
        self.assertRaises(TypeError, test_dtype)
if __name__ == '__main__':
    unittest.main()