import sys
import unittest
import numpy as np
from op_test import OpTest
sys.path.append('../../test/sequence')
from test_sequence_conv import seqconv

class TestSeqConvEltAddRelu(OpTest):

    def set_conf(self):
        if False:
            print('Hello World!')
        pass

    def setUp(self):
        if False:
            print('Hello World!')
        self.op_type = 'fusion_seqconv_eltadd_relu'
        self.lod = [[6, 4]]
        self.in_fea_size = 16
        self.out_fea_size = 8
        self.context_length = 4
        self.context_stride = 1
        self.context_start = 0
        self.set_conf()
        assert self.context_stride == 1
        T = sum(self.lod[0])
        x = np.random.uniform(-1, 1, [T, self.in_fea_size]).astype('float32')
        w = np.random.uniform(-1, 1, [self.in_fea_size * self.context_length, self.out_fea_size]).astype('float32')
        b = np.random.uniform(-2, 1, [1, self.out_fea_size]).astype('float32')
        out = seqconv(x, self.lod, w, self.context_length, self.context_start)
        out = np.maximum(out + b, 0)
        self.inputs = {'X': (x, self.lod), 'Filter': w, 'Bias': b}
        self.attrs = {'contextStart': self.context_start, 'contextLength': self.context_length, 'contextStride': self.context_stride}
        self.outputs = {'Out': out}

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_output(check_dygraph=False)

class TestSeqConvEltAddReluBS1(TestSeqConvEltAddRelu):

    def set_conf(self):
        if False:
            print('Hello World!')
        self.lod = [[10]]

class TestSeqConvEltAddReluBS1Case2(TestSeqConvEltAddRelu):

    def set_conf(self):
        if False:
            while True:
                i = 10
        self.lod = [[2]]

class TestSeqConvEltAddReluCase1(TestSeqConvEltAddRelu):

    def set_conf(self):
        if False:
            for i in range(10):
                print('nop')
        self.lod = [[3, 5, 1, 6]]
        self.context_length = 3
        self.context_start = -2

class TestSeqConvEltAddReluCase2(TestSeqConvEltAddRelu):

    def set_conf(self):
        if False:
            while True:
                i = 10
        self.lod = [[10, 1, 2, 4, 1, 5, 6]]
        self.in_fea_size = 2
        self.context_length = 4
        self.context_start = -1

class TestSeqConvEltAddReluCase3(TestSeqConvEltAddRelu):

    def set_conf(self):
        if False:
            for i in range(10):
                print('nop')
        self.lod = [[10, 1, 2, 4, 1, 5, 6]]
        self.context_length = 5
        self.context_start = -4
if __name__ == '__main__':
    unittest.main()