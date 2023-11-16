import unittest
import numpy as np
from op_test import OpTest

class TestSequenceSliceOp(OpTest):

    def set_data(self):
        if False:
            print('Hello World!')
        self.init_test_case()
        x = np.random.random(self.x_dim).astype('float32')
        lod = self.x_lod
        offset = np.array(self.offset).astype('int64')
        length = np.array(self.length).astype('int64')
        self.inputs = {'X': (x, lod), 'Offset': offset, 'Length': length}
        outs = []
        out_lod = [[]]
        lod_offset = 0
        for i in range(len(offset)):
            sub_x = x[lod_offset + offset[i, 0]:lod_offset + offset[i, 0] + length[i, 0], :]
            outs.append(sub_x)
            out_lod[0].append(len(sub_x))
            lod_offset += lod[0][i]
        outs = np.concatenate(outs, axis=0)
        self.outputs = {'Out': (outs, out_lod)}

    def init_test_case(self):
        if False:
            print('Hello World!')
        self.x_dim = (100, 3, 2)
        self.x_lod = [[20, 20, 20, 20, 20]]
        self.offset = [[1], [2], [3], [4], [5]]
        self.length = [[10], [8], [6], [4], [2]]

    def setUp(self):
        if False:
            print('Hello World!')
        self.op_type = 'sequence_slice'
        self.set_data()

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_output()

    def test_check_grad(self):
        if False:
            i = 10
            return i + 15
        self.check_grad(['X'], 'Out')

class TestSequenceSliceOpSeqlen0Case0(TestSequenceSliceOp):

    def init_test_case(self):
        if False:
            print('Hello World!')
        self.x_dim = (100, 3, 2)
        self.x_lod = [[20, 30, 0, 30, 20]]
        self.offset = [[1], [2], [0], [4], [5]]
        self.length = [[10], [8], [0], [4], [2]]

class TestSequenceSliceOpSeqlen0Case1(TestSequenceSliceOp):

    def init_test_case(self):
        if False:
            return 10
        self.x_dim = (100, 3, 2)
        self.x_lod = [[0, 70, 0, 30, 0]]
        self.offset = [[0], [2], [0], [4], [0]]
        self.length = [[0], [8], [0], [4], [0]]

class TestSequenceSliceOpSeqlen0Case2(TestSequenceSliceOp):

    def init_test_case(self):
        if False:
            for i in range(10):
                print('nop')
        self.x_dim = (100, 3, 2)
        self.x_lod = [[0, 100, 0, 0, 0]]
        self.offset = [[0], [2], [0], [0], [0]]
        self.length = [[0], [8], [0], [0], [0]]
if __name__ == '__main__':
    unittest.main()