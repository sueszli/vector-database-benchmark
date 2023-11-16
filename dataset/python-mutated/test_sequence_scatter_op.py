import unittest
import numpy as np
from op_test import OpTest

class TestSequenceScatterOp(OpTest):

    def init_lod(self):
        if False:
            print('Hello World!')
        return [[30, 50, 40]]

    def setUp(self):
        if False:
            print('Hello World!')
        self.op_type = 'sequence_scatter'
        X_data = np.random.uniform(0.1, 1.0, [3, 6]).astype('float64')
        Ids_data = np.random.randint(0, 6, (120, 1)).astype('int64')
        Ids_lod = self.init_lod()
        Updates_data = np.random.uniform(0.1, 1.0, [120, 1]).astype('float64')
        Updates_lod = Ids_lod
        Out_data = np.copy(X_data)
        offset = 0
        for i in range(3):
            for j in range(Ids_lod[0][i]):
                Out_data[i][Ids_data[offset + j]] += Updates_data[offset + j]
            offset += Ids_lod[0][i]
        self.inputs = {'X': X_data, 'Ids': (Ids_data, Ids_lod), 'Updates': (Updates_data, Updates_lod)}
        self.outputs = {'Out': Out_data}

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        self.check_output(check_dygraph=False)

    def test_check_grad(self):
        if False:
            print('Hello World!')
        self.check_grad(['Updates'], 'Out', in_place=True, check_dygraph=False)

class TestSequenceScatterOpSeqLen0(TestSequenceScatterOp):

    def init_lod(self):
        if False:
            i = 10
            return i + 15
        return [[60, 60, 0]]

class TestSequenceScatterOpSeqLen0Case1(TestSequenceScatterOp):

    def init_lod(self):
        if False:
            return 10
        return [[0, 60, 60]]

class TestSequenceScatterOpSeqLen0Case2(TestSequenceScatterOp):

    def init_lod(self):
        if False:
            i = 10
            return i + 15
        return [[60, 0, 60]]

class TestSequenceScatterOpSeqLen0Case3(TestSequenceScatterOp):

    def init_lod(self):
        if False:
            while True:
                i = 10
        return [[120, 0, 0]]

class TestSequenceScatterOpSeqLen0Case4(TestSequenceScatterOp):

    def init_lod(self):
        if False:
            for i in range(10):
                print('nop')
        return [[0, 120, 0]]

class TestSequenceScatterOpSeqLen0Case5(TestSequenceScatterOp):

    def init_lod(self):
        if False:
            print('Hello World!')
        return [[0, 0, 120]]
if __name__ == '__main__':
    unittest.main()