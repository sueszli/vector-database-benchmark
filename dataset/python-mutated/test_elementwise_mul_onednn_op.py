import unittest
import numpy as np
from op_test import skip_check_grad_ci
from test_elementwise_mul_op import ElementwiseMulOp
from paddle import enable_static

class TestOneDNNElementwiseMulOp(ElementwiseMulOp):

    def init_kernel_type(self):
        if False:
            return 10
        self.use_mkldnn = True

    def init_dtype(self):
        if False:
            for i in range(10):
                print('nop')
        self.dtype = np.float32

class TestOneDNNElementwiseMulOp2(TestOneDNNElementwiseMulOp):

    def init_input_output(self):
        if False:
            return 10
        self.x = np.random.random((100,)).astype(self.dtype)
        self.y = np.random.random((100,)).astype(self.dtype)
        self.out = np.multiply(self.x, self.y)

class TestOneDNNElementwiseMulOp3(TestOneDNNElementwiseMulOp):

    def init_input_output(self):
        if False:
            while True:
                i = 10
        self.x = np.random.uniform(0.1, 1, [2, 3, 4, 5]).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, [2, 3, 4, 5]).astype(self.dtype)
        self.out = np.multiply(self.x, self.y)

class TestOneDNNElementwiseMulOp4(TestOneDNNElementwiseMulOp):

    def init_input_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.x = np.random.uniform(1, 2, [2, 3, 4, 32]).astype(self.dtype)
        self.y = np.random.uniform(1, 2, [4, 32]).astype(self.dtype)
        self.out = np.multiply(self.x, self.y)

    def test_check_grad_normal(self):
        if False:
            i = 10
            return i + 15
        pass

    def test_check_grad_ingore_y(self):
        if False:
            for i in range(10):
                print('nop')
        pass

class TestOneDNNElementwiseMulOp5(TestOneDNNElementwiseMulOp):

    def init_input_output(self):
        if False:
            return 10
        self.x = np.random.uniform(1, 2, [2, 3, 4, 100]).astype(self.dtype)
        self.y = np.random.uniform(1, 2, [100]).astype(self.dtype)
        self.out = np.multiply(self.x, self.y)

    def test_check_grad_normal(self):
        if False:
            return 10
        pass

    def test_check_grad_ingore_y(self):
        if False:
            while True:
                i = 10
        pass

    def test_check_grad_ingore_x(self):
        if False:
            print('Hello World!')
        pass

class TestOneDNNElementwiseMulOpZeroDim(TestOneDNNElementwiseMulOp):

    def init_input_output(self):
        if False:
            return 10
        self.x = np.random.random((100,)).astype(self.dtype)
        self.y = np.array(3.0).astype(self.dtype)
        self.out = np.multiply(self.x, self.y)

    def test_check_grad_normal(self):
        if False:
            while True:
                i = 10
        pass

    def test_check_grad_ingore_y(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def test_check_grad_ingore_x(self):
        if False:
            return 10
        pass

class TestOneDNNElementwiseMulOpZeroDim2(TestOneDNNElementwiseMulOp):

    def init_input_output(self):
        if False:
            i = 10
            return i + 15
        self.x = np.array(3.0).astype(self.dtype)
        self.y = np.random.random((100,)).astype(self.dtype)
        self.out = np.multiply(self.x, self.y)

    def test_check_grad_normal(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def test_check_grad_ingore_y(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def test_check_grad_ingore_x(self):
        if False:
            print('Hello World!')
        pass

class TestOneDNNElementwiseMulOpZeroDim3(TestOneDNNElementwiseMulOp):

    def init_input_output(self):
        if False:
            return 10
        self.x = np.array(3.0).astype(self.dtype)
        self.y = np.array(3.0).astype(self.dtype)
        self.out = np.multiply(self.x, self.y)

    def test_check_grad_normal(self):
        if False:
            print('Hello World!')
        pass

    def test_check_grad_ingore_y(self):
        if False:
            while True:
                i = 10
        pass

    def test_check_grad_ingore_x(self):
        if False:
            while True:
                i = 10
        pass
' INT8 Tests '

@skip_check_grad_ci(reason="oneDNN's int8 elementwise_ops don't implemend grad kernel.")
class TestInt8(ElementwiseMulOp):

    def init_kernel_type(self):
        if False:
            i = 10
            return i + 15
        self.use_mkldnn = True
        self._cpu_only = True

    def init_dtype(self):
        if False:
            for i in range(10):
                print('nop')
        self.dtype = np.int8

    def init_input_output(self):
        if False:
            print('Hello World!')
        self.x = np.random.randint(0, 3, (12, 9)).astype('int8')
        self.y = np.random.randint(0, 3, (12, 9)).astype('int8')
        self.out = np.multiply(self.x, self.y)

    def init_scales(self):
        if False:
            for i in range(10):
                print('nop')
        self.attrs['scale_x'] = 1.0
        self.attrs['scale_y'] = 1.0
        self.attrs['scale_out'] = 1.0

    def test_check_output(self):
        if False:
            while True:
                i = 10
        self.init_scales()
        self.check_output(check_dygraph=not self.use_mkldnn)

    def test_check_grad_normal(self):
        if False:
            while True:
                i = 10
        pass

    def test_check_grad_ingore_x(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def test_check_grad_ingore_y(self):
        if False:
            while True:
                i = 10
        pass
if __name__ == '__main__':
    enable_static()
    unittest.main()