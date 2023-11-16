import unittest
import numpy as np
from op_test import OpTest, OpTestTool, convert_float_to_uint16
from paddle import enable_static
from paddle.base import core
from paddle.base.framework import _current_expected_place

@OpTestTool.skip_if(not isinstance(_current_expected_place(), core.CPUPlace), 'GPU is not supported')
class TestMKLDNNElementwiseDivOp(OpTest):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.op_type = 'elementwise_div'
        self.init_dtype()
        self.init_input_output()
        self.init_kernel_type()
        self.init_axis()
        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(self.x), 'Y': OpTest.np_dtype_to_base_dtype(self.y)}
        self.attrs = {'axis': self.axis, 'use_mkldnn': self.use_mkldnn}
        self.outputs = {'Out': self.out}

    def init_input_output(self):
        if False:
            print('Hello World!')
        self.x = np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
        self.out = np.divide(self.x, self.y)

    def test_check_grad_normal(self):
        if False:
            while True:
                i = 10
        self.check_grad(['X', 'Y'], 'Out', None, 0.005, False, 0.02)

    def test_check_grad_ignore_x(self):
        if False:
            return 10
        self.check_grad(['Y'], 'Out', set('X'), 0.005, False, 0.02)

    def test_check_grad_ignore_y(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_grad(['X'], 'Out', set('Y'), 0.005, False, 0.02)

    def init_axis(self):
        if False:
            i = 10
            return i + 15
        self.axis = -1

    def init_kernel_type(self):
        if False:
            for i in range(10):
                print('nop')
        self.use_mkldnn = True

    def init_dtype(self):
        if False:
            while True:
                i = 10
        self.dtype = np.float32

    def test_check_output(self):
        if False:
            while True:
                i = 10
        self.check_output()

class TestMKLDNNElementwiseDivOp2(TestMKLDNNElementwiseDivOp):

    def init_input_output(self):
        if False:
            return 10
        self.x = np.random.uniform(0.1, 1, [100]).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, [100]).astype(self.dtype)
        self.out = np.divide(self.x, self.y)

class TestMKLDNNElementwiseDivOp3(TestMKLDNNElementwiseDivOp):

    def init_input_output(self):
        if False:
            while True:
                i = 10
        self.x = np.random.uniform(0.1, 1, [2, 3, 4, 5]).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, [2, 3, 4, 5]).astype(self.dtype)
        self.out = np.divide(self.x, self.y)

class TestMKLDNNElementwiseDivOp4(TestMKLDNNElementwiseDivOp):

    def init_input_output(self):
        if False:
            while True:
                i = 10
        self.x = np.random.uniform(1, 2, [2, 3, 4, 32]).astype(self.dtype)
        self.y = np.random.uniform(1, 2, [4, 32]).astype(self.dtype)
        self.out = np.divide(self.x, self.y)

    def test_check_grad_normal(self):
        if False:
            print('Hello World!')
        pass

    def test_check_grad_ignore_x(self):
        if False:
            i = 10
            return i + 15
        pass

class TestMKLDNNElementwiseDivOp5(TestMKLDNNElementwiseDivOp):

    def init_input_output(self):
        if False:
            while True:
                i = 10
        self.x = np.random.uniform(1, 2, [2, 3, 4, 100]).astype(self.dtype)
        self.y = np.random.uniform(1, 2, [100]).astype(self.dtype)
        self.out = np.divide(self.x, self.y)

    def test_check_grad_normal(self):
        if False:
            i = 10
            return i + 15
        pass

    def test_check_grad_ignore_x(self):
        if False:
            print('Hello World!')
        pass

class TestMKLDNNElementwiseDivOpZeroDim(TestMKLDNNElementwiseDivOp):

    def init_input_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.x = np.random.uniform(0.1, 1, [100]).astype(self.dtype)
        self.y = np.array(3.0).astype(self.dtype)
        self.out = np.divide(self.x, self.y)

    def test_check_grad_normal(self):
        if False:
            while True:
                i = 10
        pass

    def test_check_grad_ignore_x(self):
        if False:
            print('Hello World!')
        pass

class TestMKLDNNElementwiseDivOpZeroDim2(TestMKLDNNElementwiseDivOp):

    def init_input_output(self):
        if False:
            print('Hello World!')
        self.x = np.array(3.0).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, [100]).astype(self.dtype)
        self.out = np.divide(self.x, self.y)

    def test_check_grad_normal(self):
        if False:
            i = 10
            return i + 15
        pass

    def test_check_grad_ignore_x(self):
        if False:
            print('Hello World!')
        pass

class TestMKLDNNElementwiseDivOpZeroDim3(TestMKLDNNElementwiseDivOp):

    def init_input_output(self):
        if False:
            return 10
        self.x = np.array(3.0).astype(self.dtype)
        self.y = np.array(3.0).astype(self.dtype)
        self.out = np.divide(self.x, self.y)

    def test_check_grad_normal(self):
        if False:
            return 10
        pass

    def test_check_grad_ignore_x(self):
        if False:
            print('Hello World!')
        pass

@OpTestTool.skip_if_not_cpu_bf16()
class TestBf16(TestMKLDNNElementwiseDivOp):

    def setUp(self):
        if False:
            print('Hello World!')
        self.op_type = 'elementwise_div'
        self.init_dtype()
        self.init_input_output()
        self.init_kernel_type()
        self.init_axis()
        self.x_bf16 = convert_float_to_uint16(self.x)
        self.y_bf16 = convert_float_to_uint16(self.y)
        self.inputs = {'X': self.x_bf16, 'Y': self.y_bf16}
        self.attrs = {'axis': self.axis, 'use_mkldnn': self.use_mkldnn}
        self.outputs = {'Out': convert_float_to_uint16(self.out)}

    def init_dtype(self):
        if False:
            return 10
        self.dtype = np.float32
        self.mkldnn_data_type = 'bfloat16'

    def init_input_output(self):
        if False:
            print('Hello World!')
        self.x = np.random.uniform(0.1, 1, [100]).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, [100]).astype(self.dtype)
        self.out = np.divide(self.x, self.y)

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_output_with_place(core.CPUPlace())

    def test_check_grad_normal(self):
        if False:
            i = 10
            return i + 15
        self.check_grad_with_place(core.CPUPlace(), ['X', 'Y'], 'Out', user_defined_grads=[np.divide(self.x, self.y), np.divide(np.multiply(-self.x, self.x), np.multiply(self.y, self.y))], user_defined_grad_outputs=[self.x_bf16])

    def test_check_grad_ignore_x(self):
        if False:
            while True:
                i = 10
        self.check_grad_with_place(core.CPUPlace(), ['Y'], 'Out', user_defined_grads=[np.divide(np.multiply(-self.x, self.y), np.multiply(self.y, self.y))], user_defined_grad_outputs=[self.y_bf16])

    def test_check_grad_ignore_y(self):
        if False:
            return 10
        self.check_grad_with_place(core.CPUPlace(), ['X'], 'Out', user_defined_grads=[np.divide(self.x, self.y)], user_defined_grad_outputs=[self.x_bf16])

class TestBf16Broadcasting(TestBf16):

    def init_input_output(self):
        if False:
            while True:
                i = 10
        self.x = np.random.uniform(1, 2, [2, 3, 4, 100]).astype(self.dtype)
        self.y = np.random.uniform(1, 2, [100]).astype(self.dtype)
        self.out = np.subtract(self.x, self.y)

    def test_check_grad_normal(self):
        if False:
            print('Hello World!')
        pass

    def test_check_grad_ignore_x(self):
        if False:
            i = 10
            return i + 15
        pass
if __name__ == '__main__':
    enable_static()
    unittest.main()