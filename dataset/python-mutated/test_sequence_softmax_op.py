import unittest
import numpy as np
from op_test import OpTest
from test_softmax_op import stable_softmax
from paddle.base import core

class TestSequenceSoftmaxOp(OpTest):

    def setUp(self):
        if False:
            return 10
        self.op_type = 'sequence_softmax'
        self.use_cudnn = False
        self.init_op_type()
        self.dtype = 'float32' if core.is_compiled_with_rocm() else 'float64'
        x = np.random.uniform(0.1, 1, (110, 1)).astype(self.dtype)
        self.init_lod()
        out = np.zeros((110, 1)).astype(self.dtype)
        offset = 0
        for i in range(len(self.lod[0])):
            if self.lod[0][i] == 0:
                continue
            sub_x = x[offset:offset + self.lod[0][i], :]
            sub_x = sub_x.reshape(1, self.lod[0][i])
            sub_out = stable_softmax(sub_x)
            out[offset:offset + self.lod[0][i], :] = sub_out.reshape(self.lod[0][i], 1)
            offset += self.lod[0][i]
        self.inputs = {'X': (x, self.lod)}
        self.outputs = {'Out': out}
        self.attrs = {'use_cudnn': self.use_cudnn}

    def init_lod(self):
        if False:
            while True:
                i = 10
        self.lod = [[40, 10, 30, 30]]

    def init_op_type(self):
        if False:
            i = 10
            return i + 15
        pass

    def test_check_output(self):
        if False:
            print('Hello World!')
        if self.use_cudnn:
            place = core.CUDAPlace(0)
            self.check_output_with_place(place, atol=1e-05)
        else:
            self.check_output()

    def test_check_grad(self):
        if False:
            return 10
        if self.use_cudnn:
            place = core.CUDAPlace(0)
            self.check_grad_with_place(place, ['X'], 'Out')
        else:
            self.check_grad(['X'], 'Out')

@unittest.skipIf(not core.is_compiled_with_cuda(), 'core is not compiled with CUDA')
class TestSequenceSoftmaxCUDNNOp(TestSequenceSoftmaxOp):

    def init_op_type(self):
        if False:
            print('Hello World!')
        self.use_cudnn = True

class TestSequenceSoftmaxOpSeqLen0Case0(TestSequenceSoftmaxOp):

    def init_lod(self):
        if False:
            while True:
                i = 10
        self.lod = [[40, 0, 40, 30]]

class TestSequenceSoftmaxOpSeqLen0Case1(TestSequenceSoftmaxOp):

    def init_lod(self):
        if False:
            i = 10
            return i + 15
        self.lod = [[0, 40, 70, 0]]

class TestSequenceSoftmaxOpSeqLen0Case2(TestSequenceSoftmaxOp):

    def init_lod(self):
        if False:
            print('Hello World!')
        self.lod = [[0, 0, 0, 110]]
if __name__ == '__main__':
    unittest.main()