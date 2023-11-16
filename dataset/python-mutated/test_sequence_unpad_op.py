import unittest
import numpy as np
from op_test import OpTest
import paddle

class TestSequenceUnpadOp(OpTest):

    def init(self):
        if False:
            for i in range(10):
                print('nop')
        self.length = [2, 3, 4]
        self.x_shape = (3, 40)
        self.dtype = 'float64'

    def compute(self):
        if False:
            for i in range(10):
                print('nop')
        assert len(self.length) == self.x_shape[0]
        x = np.random.random(self.x_shape).astype(self.dtype)
        out_lod = [self.length]
        out = x[0, 0:self.length[0]]
        for i in range(1, x.shape[0]):
            out = np.append(out, x[i, 0:self.length[i]], axis=0)
        out_shape = (sum(self.length),)
        if len(self.x_shape) == 2:
            out_shape = out_shape + (1,)
        else:
            out_shape = out_shape + self.x_shape[2:]
        self.inputs = {'X': x, 'Length': np.array(self.length).astype('int64')}
        self.outputs = {'Out': (out.reshape(out_shape), out_lod)}

    def setUp(self):
        if False:
            return 10
        self.op_type = 'sequence_unpad'
        self.init()
        self.compute()

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_output(check_dygraph=False)

    def test_check_grad(self):
        if False:
            print('Hello World!')
        self.check_grad(['X'], 'Out', check_dygraph=False)

class TestSequenceUnpadOp2(TestSequenceUnpadOp):

    def init(self):
        if False:
            print('Hello World!')
        self.length = [2, 3, 4]
        self.x_shape = (3, 5, 4, 3)
        self.dtype = 'float64'

class TestSequenceUnpadOp3(TestSequenceUnpadOp):

    def init(self):
        if False:
            return 10
        self.length = [5, 2, 3, 4]
        self.x_shape = (4, 5, 3, 3, 6)
        self.dtype = 'float64'

class TestSequenceUnpadOp4(TestSequenceUnpadOp):

    def init(self):
        if False:
            while True:
                i = 10
        self.length = [5, 0, 0, 4]
        self.x_shape = (4, 5, 3, 3, 6)
        self.dtype = 'float64'

class TestSequenceUnpadOp5(TestSequenceUnpadOp):

    def init(self):
        if False:
            return 10
        self.length = [0, 4, 3, 0]
        self.x_shape = (4, 5, 3, 3, 6)
        self.dtype = 'float64'

class TestSequenceUnpadOpError(unittest.TestCase):

    def test_error(self):
        if False:
            print('Hello World!')

        def test_x_variable():
            if False:
                for i in range(10):
                    print('nop')
            x = np.random.random((10, 5)).astype('float64')
            len = paddle.static.data(name='length2', shape=[10], dtype='int64')
            paddle.static.nn.sequence_lod.sequence_pad(x=x, length=len)
        self.assertRaises(TypeError, test_x_variable)

        def test_length_variable():
            if False:
                while True:
                    i = 10
            x1 = paddle.static.data(name='x1', shape=[10, 5], dtype='float32')
            len1 = np.random.random(10).astype('int64')
            paddle.static.nn.sequence_lod.sequence_pad(x=x1, length=len1)
        self.assertRaises(TypeError, test_length_variable)

        def test_x_dtype():
            if False:
                print('Hello World!')
            x2 = paddle.static.data(name='x2', shape=[10, 5], dtype='float16')
            len2 = paddle.static.data(name='length2', shape=[10], dtype='int64')
            paddle.static.nn.sequence_lod.sequence_pad(x=x2, length=len2)
        self.assertRaises(TypeError, test_x_dtype)

        def test_length_dtype():
            if False:
                return 10
            x3 = paddle.static.data(name='x3', shape=[10, 5], dtype='float64')
            len3 = paddle.static.data(name='length3', shape=[10], dtype='int32')
            paddle.static.nn.sequence_lod.sequence_pad(x=x3, length=len3)
        self.assertRaises(TypeError, test_length_dtype)
if __name__ == '__main__':
    unittest.main()