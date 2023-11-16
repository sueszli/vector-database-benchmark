import unittest
import numpy as np
from op_test import OpTest
import paddle

class TestSequenceReshape(OpTest):

    def init_data(self):
        if False:
            for i in range(10):
                print('nop')
        self.dimension = 12
        self.x_lod = [[4, 1, 3, 3]]
        self.x = np.random.uniform(0.1, 1, [11, 24]).astype('float64')

    def setUp(self):
        if False:
            return 10
        self.init_data()
        self.op_type = 'sequence_reshape'
        self.inputs = {'X': (self.x, self.x_lod)}
        self.attrs = {'new_dim': self.dimension}
        (out, out_lod) = self.compute_output(self.x, self.x_lod, self.dimension)
        self.outputs = {'Out': (out, out_lod)}

    def compute_output(self, x, x_lod, dimension):
        if False:
            return 10
        x_width = x.shape[1]
        out_lod = [[]]
        for i in range(len(x_lod[0])):
            seq_len = x_lod[0][i]
            offset = seq_len * x_width / dimension
            assert int(offset) * dimension == seq_len * x_width
            out_lod[0].append(int(offset))
        out = np.zeros(shape=(sum(out_lod[0]), dimension)).astype('float64')
        out.ravel()[:] = x.ravel()[:]
        return (out, out_lod)

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_output()

    def test_check_grad(self):
        if False:
            return 10
        self.check_grad(['X'], 'Out')

class TestSequenceReshape_reduce(TestSequenceReshape):

    def init_data(self):
        if False:
            print('Hello World!')
        self.dimension = 24
        self.x_lod = [[4, 2, 2, 4]]
        self.x = np.random.uniform(0.1, 1, [12, 12]).astype('float64')

class TestSequenceReshape_same(TestSequenceReshape):

    def init_data(self):
        if False:
            for i in range(10):
                print('nop')
        self.dimension = 12
        self.x_lod = [[4, 2, 2, 4]]
        self.x = np.random.uniform(0.1, 1, [12, 12]).astype('float64')

class TestSequenceReshape_reduce_seq_len0(TestSequenceReshape):

    def init_data(self):
        if False:
            i = 10
            return i + 15
        self.dimension = 24
        self.x_lod = [[0, 6, 0, 2, 4]]
        self.x = np.random.uniform(0.1, 1, [12, 12]).astype('float64')

class TestSequenceReshape_reduce_seq_len0_case1(TestSequenceReshape):

    def init_data(self):
        if False:
            for i in range(10):
                print('nop')
        self.dimension = 24
        self.x_lod = [[0, 2, 8, 2, 0]]
        self.x = np.random.uniform(0.1, 1, [12, 12]).astype('float64')

class TestSequenceReshapeOpError(unittest.TestCase):

    def test_error(self):
        if False:
            print('Hello World!')

        def test_variable():
            if False:
                return 10
            x = np.random.random((2, 4)).astype('float32')
            paddle.static.nn.sequence_lod.sequence_reshape(x=x, new_dim=4)
        self.assertRaises(TypeError, test_variable)

        def test_dtype():
            if False:
                while True:
                    i = 10
            x1 = paddle.static.data(name='x1', shape=[-1, 2, 6], dtype='float16', lod_level=1)
            paddle.static.nn.sequence_lod.sequence_reshape(x=x1, new_dim=4)
        self.assertRaises(TypeError, test_dtype)
if __name__ == '__main__':
    unittest.main()