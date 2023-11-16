import unittest
import numpy as np
from op_test import OpTest
from paddle.base import core

def np_cal_batchfc(input, w, bias):
    if False:
        i = 10
        return i + 15
    (slot_pairs_num, batch_size, in_dim) = input.shape
    (_, _, out_dim) = w.shape
    res = np.zeros((slot_pairs_num, batch_size, out_dim))
    for slot in range(slot_pairs_num):
        res[slot, :] = np.dot(input[slot, :], w[slot, :])
    for slot in range(slot_pairs_num):
        for bindx in range(out_dim):
            res[slot, :, bindx] += bias[slot, bindx]
    return res

class TestBatchFCOp(OpTest):

    def config(self):
        if False:
            return 10
        self.slot_pairs_num = 10
        self.batch_size = 5
        self.in_dim = 10
        self.out_dim = 12
        self.dtype = 'float64'

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.config()
        self.input = np.random.random((self.slot_pairs_num, self.batch_size, self.in_dim)).astype(self.dtype)
        self.w = np.random.random((self.slot_pairs_num, self.in_dim, self.out_dim)).astype(self.dtype)
        self.bias = np.random.random((self.slot_pairs_num, self.out_dim)).astype(self.dtype)
        self.op_type = 'batch_fc'
        np_out = np_cal_batchfc(self.input, self.w, self.bias)
        np_out = np_out.astype(self.dtype)
        self.inputs = {'Input': self.input, 'W': self.w, 'Bias': self.bias}
        self.outputs = {'Out': np_out}

    def test_check_output_gpu(self):
        if False:
            i = 10
            return i + 15
        if core.is_compiled_with_cuda():
            self.check_output_with_place(core.CUDAPlace(0))

    def test_check_grad_gpu(self):
        if False:
            for i in range(10):
                print('nop')
        if core.is_compiled_with_cuda():
            self.check_grad_with_place(core.CUDAPlace(0), ['Bias', 'W', 'Input'], 'Out')

class TestBatchFCOp1(OpTest):

    def config(self):
        if False:
            for i in range(10):
                print('nop')
        self.slot_pairs_num = 10
        self.batch_size = 5
        self.in_dim = 10
        self.out_dim = 12
        self.dtype = 'float64'

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.config()
        self.input = np.random.random((self.slot_pairs_num, self.batch_size, self.in_dim)).astype(self.dtype)
        self.w = np.random.random((self.slot_pairs_num, self.in_dim, self.out_dim)).astype(self.dtype)
        self.bias = np.random.random((self.slot_pairs_num, self.out_dim)).astype(self.dtype)
        self.op_type = 'batch_fc'
        np_out = np_cal_batchfc(self.input, self.w, self.bias)
        np_out = np_out.astype(self.dtype)
        self.inputs = {'Input': self.input, 'W': self.w, 'Bias': self.bias}
        self.outputs = {'Out': np_out}

    def test_check_output_cpu(self):
        if False:
            return 10
        try:
            self.check_output_with_place(place=core.CPUPlace())
        except:
            print('do not support cpu test, skip')

    def test_check_grad_cpu(self):
        if False:
            return 10
        try:
            self.check_grad_with_place(core.CPUPlace(), ['Bias', 'W', 'Input'], 'Out')
        except:
            print('do not support cpu test, skip')
if __name__ == '__main__':
    unittest.main()