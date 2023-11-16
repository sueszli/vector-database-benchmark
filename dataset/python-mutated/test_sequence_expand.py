import unittest
import numpy as np
from op_test import OpTest

class TestSequenceExpand(OpTest):

    def set_data(self):
        if False:
            while True:
                i = 10
        x_data = np.random.uniform(0.1, 1, [3, 40]).astype('float64')
        y_data = np.random.uniform(0.1, 1, [8, 1]).astype('float64')
        y_lod = [[1, 3, 4]]
        self.inputs = {'X': x_data, 'Y': (y_data, y_lod)}

    def compute(self):
        if False:
            i = 10
            return i + 15
        x = self.inputs['X']
        (x_data, x_lod) = x if type(x) == tuple else (x, None)
        (y_data, y_lod) = self.inputs['Y']
        if hasattr(self, 'attrs'):
            ref_level = self.attrs['ref_level']
        else:
            ref_level = len(y_lod) - 1
        out = np.zeros(shape=(0,) + x_data.shape[1:], dtype=x_data.dtype)
        if x_lod is None:
            x_idx = [1] * x_data.shape[0]
        else:
            x_idx = x_lod[0]
            out_lod = [[]]
        offset = 0
        for i in range(len(y_lod[ref_level])):
            repeat_num = y_lod[ref_level][i]
            x_len = x_idx[i]
            if repeat_num > 0:
                x_sub = x_data[offset:offset + x_len, :]
                stacked_x_sub = x_sub
                for r in range(repeat_num - 1):
                    stacked_x_sub = np.vstack((stacked_x_sub, x_sub))
                out = np.vstack((out, stacked_x_sub))
                if x_lod is not None:
                    for j in range(repeat_num):
                        out_lod[0].append(x_len)
            offset += x_len
        if x_lod is None:
            self.outputs = {'Out': out}
        else:
            self.outputs = {'Out': (out, out_lod)}

    def setUp(self):
        if False:
            print('Hello World!')
        self.op_type = 'sequence_expand'
        self.set_data()
        self.compute()

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        self.check_output(check_dygraph=False)

    def test_check_grad(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_grad(['X'], 'Out', check_dygraph=False)

class TestSequenceExpandCase1(TestSequenceExpand):

    def set_data(self):
        if False:
            print('Hello World!')
        x_data = np.random.uniform(0.1, 1, [5, 20]).astype('float64')
        y_data = np.random.uniform(0.1, 1, [13, 1]).astype('float64')
        y_lod = [[2, 3], [2, 2, 3, 3, 3]]
        self.inputs = {'X': x_data, 'Y': (y_data, y_lod)}
        self.attrs = {'ref_level': 1}

class TestSequenceExpandCase2(TestSequenceExpand):

    def set_data(self):
        if False:
            return 10
        x_data = np.random.uniform(0.1, 1, [1, 2, 50]).astype('float64')
        x_lod = [[1]]
        y_data = np.random.uniform(0.1, 1, [2, 2, 2]).astype('float64')
        y_lod = [[2], [1, 1]]
        self.inputs = {'X': (x_data, x_lod), 'Y': (y_data, y_lod)}
        self.attrs = {'ref_level': 0}

class TestSequenceExpandCase3(TestSequenceExpand):

    def set_data(self):
        if False:
            i = 10
            return i + 15
        x_data = np.random.uniform(0.1, 1, [4, 25]).astype('float64')
        x_lod = [[1, 1, 1, 1]]
        y_data = np.random.uniform(0.1, 1, [8, 1]).astype('float64')
        y_lod = [[2, 2, 2, 2]]
        self.inputs = {'X': (x_data, x_lod), 'Y': (y_data, y_lod)}

class TestSequenceExpandCase4(TestSequenceExpand):

    def set_data(self):
        if False:
            i = 10
            return i + 15
        data = np.random.uniform(0.1, 1, [5 * 20, 1])
        x_data = np.array(data).reshape([5, 20]).astype('float64')
        x_lod = [[2, 3]]
        y_data = np.random.uniform(0.1, 1, [5, 1]).astype('float64')
        y_lod = [[2], [2, 3]]
        self.inputs = {'X': (x_data, x_lod), 'Y': (y_data, y_lod)}

class TestSequenceExpandCase5(TestSequenceExpand):

    def set_data(self):
        if False:
            while True:
                i = 10
        x_data = np.random.uniform(0.1, 1, [6, 20]).astype('float64')
        y_data = np.random.uniform(0.1, 1, [13, 1]).astype('float64')
        y_lod = [[2, 4], [2, 2, 3, 0, 3, 3]]
        self.inputs = {'X': x_data, 'Y': (y_data, y_lod)}
        self.attrs = {'ref_level': 1}

class TestSequenceExpandCase6(TestSequenceExpand):

    def set_data(self):
        if False:
            i = 10
            return i + 15
        x_data = np.random.uniform(0.1, 1, [4, 25]).astype('float64')
        x_lod = [[1, 1, 0, 1, 1]]
        y_data = np.random.uniform(0.1, 1, [8, 1]).astype('float64')
        y_lod = [[0, 2, 4, 2, 0]]
        self.inputs = {'X': (x_data, x_lod), 'Y': (y_data, y_lod)}
if __name__ == '__main__':
    unittest.main()