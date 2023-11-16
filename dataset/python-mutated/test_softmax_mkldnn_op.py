import unittest
import numpy as np
from mkldnn_op_test import check_if_mkldnn_primitives_exist_in_bwd
from op_test import OpTest
from test_softmax_op import TestSoftmaxOp, TestSoftmaxOp2, TestSoftmaxOp3, TestSoftmaxOp4, TestSoftmaxOp5, TestSoftmaxOp6, TestSoftmaxOp_ZeroDim1
import paddle
from paddle.base import core
paddle.enable_static()

def stable_softmax(x):
    if False:
        i = 10
        return i + 15
    'Compute the softmax of vector x in a numerically stable way.'
    shiftx = x - np.max(x).clip(-64.0)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)

class TestSoftmaxMKLDNNOp(TestSoftmaxOp):

    def get_x_shape(self):
        if False:
            print('Hello World!')
        return [10, 10]

    def get_axis(self):
        if False:
            for i in range(10):
                print('nop')
        return -1

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.op_type = 'softmax'
        self.use_cudnn = False
        self.use_mkldnn = False
        self.dtype = np.float32
        self.init_kernel_type()
        self.shape = self.get_x_shape()
        self.axis = self.get_axis()
        x = np.random.uniform(0.1, 1, self.shape).astype(self.dtype)
        out = np.apply_along_axis(stable_softmax, self.axis, x)
        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {'Out': out}
        self.attrs = {'axis': self.axis, 'use_cudnn': self.use_cudnn, 'use_mkldnn': self.use_mkldnn}

    def test_check_output(self):
        if False:
            while True:
                i = 10
        if self.use_cudnn:
            place = core.CUDAPlace(0)
            self.check_output_with_place(place, check_dygraph=False)
        else:
            self.check_output(check_dygraph=False)

    def test_check_grad(self):
        if False:
            return 10
        if self.use_cudnn or self.dtype == np.float16:
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                self.check_grad_with_place(place, ['X'], 'Out', max_relative_error=0.01, check_dygraph=False)
        else:
            self.check_grad(['X'], 'Out', max_relative_error=0.01, check_dygraph=False)

    def init_kernel_type(self):
        if False:
            i = 10
            return i + 15
        self.use_mkldnn = True

class TestSoftmaxMKLDNNOp2(TestSoftmaxOp2):

    def init_kernel_type(self):
        if False:
            i = 10
            return i + 15
        self.use_mkldnn = True
        self.dtype = np.float32

class TestSoftmaxMKLDNNOp3(TestSoftmaxOp3):

    def init_kernel_type(self):
        if False:
            for i in range(10):
                print('nop')
        self.use_mkldnn = True
        self.dtype = np.float32

class TestSoftmaxMKLDNNOp4(TestSoftmaxOp4):

    def init_kernel_type(self):
        if False:
            for i in range(10):
                print('nop')
        self.use_mkldnn = True
        self.dtype = np.float32

class TestSoftmaxMKLDNNOp5(TestSoftmaxOp5):

    def init_kernel_type(self):
        if False:
            return 10
        self.use_mkldnn = True
        self.dtype = np.float32

class TestSoftmaxMKLDNNOp6(TestSoftmaxOp6):

    def init_kernel_type(self):
        if False:
            while True:
                i = 10
        self.use_mkldnn = True
        self.dtype = np.float32

class TestSoftmaxMKLDNNOp_ZeroDim(TestSoftmaxOp_ZeroDim1):

    def init_kernel_type(self):
        if False:
            return 10
        self.use_mkldnn = True
        self.dtype = np.float32

class TestSoftmaxMKLDNNPrimitivesAlreadyExist(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        super().setUp()
        np.random.seed(123)
        self.op_type = 'softmax'
        self.x = np.random.uniform(-1, 1, 2).astype(np.float32)
        self.out = stable_softmax(self.x)
        self.out_grad = np.random.random_sample(self.x.shape).astype(np.float32)
        self.x_grad = self.__softmax_bwd(self.out, self.out_grad)

    def __softmax_bwd(self, out, out_grad):
        if False:
            while True:
                i = 10
        return out * (out_grad - np.dot(out, out_grad))

    def test_check(self):
        if False:
            while True:
                i = 10
        check_if_mkldnn_primitives_exist_in_bwd(self, self.op_type, self.x, self.out, self.out_grad, self.x_grad)
if __name__ == '__main__':
    from paddle import enable_static
    enable_static()
    unittest.main()