import unittest
import numpy as np
from op_test import OpTest

class TestMatchMatrixTensorOp(OpTest):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.init_op_type()
        self.set_data()
        self.compute()

    def init_op_type(self):
        if False:
            i = 10
            return i + 15
        self.op_type = 'match_matrix_tensor'

    def set_data(self):
        if False:
            while True:
                i = 10
        (ix, iy, h, dim_t) = [5, 8, 20, 4]
        x_lod = [[1, 2, 2]]
        y_lod = [[3, 1, 4]]
        self.init_data(ix, x_lod, iy, y_lod, h, dim_t)

    def init_data(self, ix, x_lod, iy, y_lod, h, dim_t):
        if False:
            while True:
                i = 10
        x_data = np.random.random((ix, h)).astype('float32')
        y_data = np.random.random((iy, h)).astype('float32')
        w_data = np.random.random((h, dim_t, h)).astype('float32')
        self.inputs = {'X': (x_data, x_lod), 'Y': (y_data, y_lod), 'W': w_data}
        self.attrs = {'dim_t': dim_t}

    def compute(self):
        if False:
            return 10
        (x_data, x_lod) = self.inputs['X']
        (y_data, y_lod) = self.inputs['Y']
        w_data = self.inputs['W'].transpose(1, 0, 2)
        out = np.zeros((0, 1), dtype=x_data.dtype)
        tmp = np.zeros((0, 1), dtype=x_data.dtype)
        out_lod = [[]]
        tmp_lod = [[]]
        (x_offset, y_offset) = (0, 0)
        for idx in range(len(x_lod[0])):
            x_len = x_lod[0][idx]
            y_len = y_lod[0][idx]
            x_sub = x_data[x_offset:x_offset + x_len, :]
            y_sub = y_data[y_offset:y_offset + y_len, :]
            tmp_sub = np.dot(x_sub, w_data)
            tmp = np.vstack((tmp, tmp_sub.reshape(tmp_sub.size, 1)))
            out_sub = np.dot(tmp_sub, y_sub.T).transpose(1, 0, 2)
            out_lod[0].append(out_sub.size)
            out = np.vstack((out, out_sub.reshape(out_sub.size, 1)))
            x_offset += x_len
            y_offset += y_len
        self.outputs = {'Out': (out, out_lod), 'Tmp': tmp}

    def test_check_output(self):
        if False:
            print('Hello World!')
        self.check_output(check_dygraph=False)

    def test_check_grad(self):
        if False:
            while True:
                i = 10
        self.check_grad(['X', 'Y'], 'Out', check_dygraph=False)

class TestMatchMatrixTensorOpCase1(TestMatchMatrixTensorOp):

    def set_data(self):
        if False:
            while True:
                i = 10
        (ix, iy, h, dim_t) = [5, 8, 25, 4]
        x_lod = [[5]]
        y_lod = [[8]]
        self.init_data(ix, x_lod, iy, y_lod, h, dim_t)

class TestMatchMatrixTensorOpCase2(TestMatchMatrixTensorOp):

    def set_data(self):
        if False:
            print('Hello World!')
        (ix, iy, h, dim_t) = [105, 120, 1, 4]
        x_lod = [[30, 45, 30]]
        y_lod = [[45, 15, 60]]
        self.init_data(ix, x_lod, iy, y_lod, h, dim_t)

class TestMatchMatrixTensorOpCase3(TestMatchMatrixTensorOp):

    def set_data(self):
        if False:
            i = 10
            return i + 15
        (ix, iy, h, dim_t) = [5, 9, 32, 1]
        x_lod = [[1, 2, 2]]
        y_lod = [[3, 2, 4]]
        self.init_data(ix, x_lod, iy, y_lod, h, dim_t)

class TestMatchMatrixTensorOpCase4(TestMatchMatrixTensorOp):

    def set_data(self):
        if False:
            for i in range(10):
                print('nop')
        (ix, iy, h, dim_t) = [8, 12, 16, 5]
        x_lod = [[1, 2, 3, 1, 1]]
        y_lod = [[3, 2, 4, 1, 2]]
        self.init_data(ix, x_lod, iy, y_lod, h, dim_t)
if __name__ == '__main__':
    unittest.main()