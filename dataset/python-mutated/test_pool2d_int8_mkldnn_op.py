import unittest
import numpy as np
from op_test import OpTest
from test_pool2d_op import TestPool2D_Op, max_pool2D_forward_naive
from paddle.base import core

class TestPool2DMKLDNNInt8_Op(TestPool2D_Op):

    def init_kernel_type(self):
        if False:
            i = 10
            return i + 15
        self.use_mkldnn = True

    def init_data_type(self):
        if False:
            i = 10
            return i + 15
        self.dtype = np.int8

    def setUp(self):
        if False:
            return 10
        TestPool2D_Op.setUp(self)
        assert self.dtype in [np.int8, np.uint8], 'Dtype should be int8 or uint8'
        input = np.random.randint(0, 100, self.shape).astype(self.dtype)
        output = self.pool2D_forward_naive(input, self.ksize, self.strides, self.paddings, self.global_pool, self.ceil_mode, self.exclusive, self.adaptive, self.dtype).astype(self.dtype)
        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(input)}
        self.outputs = {'Out': output}

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_output_with_place(core.CPUPlace(), atol=1e-05, check_dygraph=False)

    def test_check_grad(self):
        if False:
            return 10
        pass

class TestCase1Avg(TestPool2DMKLDNNInt8_Op):

    def init_test_case(self):
        if False:
            while True:
                i = 10
        self.shape = [2, 3, 7, 7]
        self.ksize = [3, 3]
        self.strides = [1, 1]
        self.paddings = [0, 0]

    def init_global_pool(self):
        if False:
            for i in range(10):
                print('nop')
        self.global_pool = False

    def init_exclusive(self):
        if False:
            while True:
                i = 10
        self.exclusive = True

class TestCase2Avg(TestPool2DMKLDNNInt8_Op):

    def init_test_case(self):
        if False:
            i = 10
            return i + 15
        self.shape = [2, 3, 7, 7]
        self.ksize = [3, 3]
        self.strides = [1, 1]
        self.paddings = [1, 1]

    def init_global_pool(self):
        if False:
            i = 10
            return i + 15
        self.global_pool = False

    def init_exclusive(self):
        if False:
            while True:
                i = 10
        self.exclusive = False

class TestCase0Max(TestPool2DMKLDNNInt8_Op):

    def init_pool_type(self):
        if False:
            for i in range(10):
                print('nop')
        self.pool_type = 'max'
        self.pool2D_forward_naive = max_pool2D_forward_naive

class TestCase1Max(TestCase1Avg):

    def init_pool_type(self):
        if False:
            for i in range(10):
                print('nop')
        self.pool_type = 'max'
        self.pool2D_forward_naive = max_pool2D_forward_naive

class TestCase2Max(TestCase2Avg):

    def init_pool_type(self):
        if False:
            return 10
        self.pool_type = 'max'
        self.pool2D_forward_naive = max_pool2D_forward_naive

def create_test_s8_u8_class(parent):
    if False:
        while True:
            i = 10

    class TestS8Case(parent):

        def init_data_type(self):
            if False:
                return 10
            self.dtype = np.int8

    class TestU8Case(parent):

        def init_data_type(self):
            if False:
                while True:
                    i = 10
            self.dtype = np.uint8
    cls_name_s8 = '{}_{}'.format(parent.__name__, 'mkldnn_s8')
    cls_name_u8 = '{}_{}'.format(parent.__name__, 'mkldnn_u8')
    TestS8Case.__name__ = cls_name_s8
    TestU8Case.__name__ = cls_name_u8
    globals()[cls_name_s8] = TestS8Case
    globals()[cls_name_u8] = TestU8Case
create_test_s8_u8_class(TestPool2DMKLDNNInt8_Op)
create_test_s8_u8_class(TestCase1Avg)
create_test_s8_u8_class(TestCase2Avg)
create_test_s8_u8_class(TestCase0Max)
create_test_s8_u8_class(TestCase1Max)
create_test_s8_u8_class(TestCase2Max)
if __name__ == '__main__':
    unittest.main()