import unittest
import numpy as np
from op_test import skip_check_grad_ci
from test_elementwise_add_op import TestElementwiseAddOp
from paddle import enable_static

class TestOneDNNElementwiseAddOp(TestElementwiseAddOp):

    def init_kernel_type(self):
        if False:
            while True:
                i = 10
        self.use_mkldnn = True

    def init_dtype(self):
        if False:
            for i in range(10):
                print('nop')
        self.dtype = np.float32

class TestOneDNNElementwiseAddOp2(TestOneDNNElementwiseAddOp):

    def init_input_output(self):
        if False:
            print('Hello World!')
        self.x = np.random.random((100,)).astype(self.dtype)
        self.y = np.random.random((100,)).astype(self.dtype)
        self.out = np.add(self.x, self.y)

class TestOneDNNElementwiseAddOp3(TestOneDNNElementwiseAddOp):

    def init_input_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.x = np.random.uniform(0.1, 1, [2, 3, 4, 5]).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, [2, 3, 4, 5]).astype(self.dtype)
        self.out = np.add(self.x, self.y)

class TestOneDNNElementwiseAddOp4(TestOneDNNElementwiseAddOp):

    def init_input_output(self):
        if False:
            print('Hello World!')
        self.x = np.random.uniform(1, 2, [2, 3, 4, 32]).astype(self.dtype)
        self.y = np.random.uniform(1, 2, [4, 32]).astype(self.dtype)
        self.out = np.add(self.x, self.y)

    def test_check_grad_normal(self):
        if False:
            print('Hello World!')
        pass

    def test_check_grad_ingore_y(self):
        if False:
            while True:
                i = 10
        pass

class TestOneDNNElementwiseAddOp5(TestOneDNNElementwiseAddOp):

    def init_input_output(self):
        if False:
            while True:
                i = 10
        self.x = np.random.uniform(1, 2, [2, 3, 4, 100]).astype(self.dtype)
        self.y = np.random.uniform(1, 2, [100]).astype(self.dtype)
        self.out = np.add(self.x, self.y)

class TestOneDNNElementwiseAddOpBroadcastXintoY(TestOneDNNElementwiseAddOp):

    def init_input_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.x = np.random.uniform(1, 2, [2, 50, 1]).astype(self.dtype)
        self.y = np.random.uniform(1, 2, [2, 50, 160]).astype(self.dtype)
        self.out = np.add(self.x, self.y)

class TestOneDNNElementwiseAddOp_broadcast_3(TestOneDNNElementwiseAddOp):

    def init_input_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.x = np.random.rand(2, 10, 12, 3).astype(self.dtype)
        self.y = np.random.rand(10, 12).astype(self.dtype)
        self.out = self.x + self.y.reshape(1, 10, 12, 1)

    def init_axis(self):
        if False:
            return 10
        self.axis = 1

class TestElementwiseAddOp_xsize_lessthan_ysize_add(TestOneDNNElementwiseAddOp):

    def init_input_output(self):
        if False:
            print('Hello World!')
        self.x = np.random.rand(10, 12).astype(self.dtype)
        self.y = np.random.rand(2, 2, 10, 12).astype(self.dtype)
        self.out = self.x + self.y

    def init_axis(self):
        if False:
            while True:
                i = 10
        self.axis = 2

    def test_check_grad_normal(self):
        if False:
            i = 10
            return i + 15
        pass

    def test_check_grad_ingore_y(self):
        if False:
            return 10
        pass

    def test_check_grad_ingore_x(self):
        if False:
            i = 10
            return i + 15
        pass

class TestOneDNNlementwiseAddOpZeroDim(TestOneDNNElementwiseAddOp):

    def init_input_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.x = np.random.random((100,)).astype(self.dtype)
        self.y = np.array(3.0).astype(self.dtype)
        self.out = np.add(self.x, self.y)

class TestOneDNNlementwiseAddOpZeroDim2(TestOneDNNElementwiseAddOp):

    def init_input_output(self):
        if False:
            print('Hello World!')
        self.x = np.array(3.0).astype(self.dtype)
        self.y = np.random.random((100,)).astype(self.dtype)
        self.out = np.add(self.x, self.y)

class TestOneDNNlementwiseAddOpZeroDim3(TestOneDNNElementwiseAddOp):

    def init_input_output(self):
        if False:
            return 10
        self.x = np.array(3.0).astype(self.dtype)
        self.y = np.array(3.0).astype(self.dtype)
        self.out = np.add(self.x, self.y)
' INT8 Tests '

@skip_check_grad_ci(reason="oneDNN's int8 elementwise_ops don't implemend grad kernel.")
class TestInt8(TestElementwiseAddOp):

    def init_kernel_type(self):
        if False:
            return 10
        self.use_mkldnn = True
        self._cpu_only = True

    def init_dtype(self):
        if False:
            i = 10
            return i + 15
        self.dtype = np.int8

    def init_input_output(self):
        if False:
            while True:
                i = 10
        self.x = np.random.randint(0, 3, (12, 9)).astype('int8')
        self.y = np.random.randint(0, 3, (12, 9)).astype('int8')
        self.out = np.add(self.x, self.y)

    def init_scales(self):
        if False:
            print('Hello World!')
        self.attrs['scale_x'] = 1.0
        self.attrs['scale_y'] = 1.0
        self.attrs['scale_out'] = 1.0

    def test_check_output(self):
        if False:
            return 10
        self.init_scales()
        self.check_output(check_dygraph=not self.use_mkldnn)

    def test_check_grad_normal(self):
        if False:
            return 10
        pass

    def test_check_grad_ingore_x(self):
        if False:
            while True:
                i = 10
        pass

    def test_check_grad_ingore_y(self):
        if False:
            print('Hello World!')
        pass
if __name__ == '__main__':
    enable_static()
    unittest.main()