import unittest
import numpy as np
from op_test import OpTest, OpTestTool, convert_float_to_uint16
import paddle
from paddle.base import core

class TestClipOneDNNOp(OpTest):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.op_type = 'clip'
        self.init_shape()
        self.set_inputs()
        self.set_attrs()
        self.set_additional_inputs()
        self.adjust_op_settings()
        self.min = self.attrs['min'] if 'Min' not in self.inputs else self.inputs['Min']
        self.max = self.attrs['max'] if 'Max' not in self.inputs else self.inputs['Max']
        self.outputs = {'Out': np.clip(self.x_fp32, self.min, self.max)}

    def init_shape(self):
        if False:
            print('Hello World!')
        self.shape = [10, 10]

    def set_inputs(self):
        if False:
            return 10
        self.inputs = {'X': np.array(np.random.random(self.shape).astype(np.float32) * 25)}
        self.x_fp32 = self.inputs['X']

    def set_additional_inputs(self):
        if False:
            while True:
                i = 10
        pass

    def adjust_op_settings(self):
        if False:
            return 10
        pass

    def set_attrs(self):
        if False:
            return 10
        self.attrs = {'min': 7.2, 'max': 9.6, 'use_mkldnn': True}

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_output(check_dygraph=False)

    def test_check_grad(self):
        if False:
            return 10
        self.check_grad(['X'], 'Out', check_dygraph=False)

class TestClipOneDNNOp_ZeroDim(TestClipOneDNNOp):

    def init_shape(self):
        if False:
            i = 10
            return i + 15
        self.shape = []

class TestClipMinAsInputOneDNNOp(TestClipOneDNNOp):

    def set_additional_inputs(self):
        if False:
            for i in range(10):
                print('nop')
        self.inputs['Min'] = np.array([6.8]).astype('float32')

class TestClipMaxAsInputOneDNNOp(TestClipOneDNNOp):

    def set_additional_inputs(self):
        if False:
            print('Hello World!')
        self.inputs['Max'] = np.array([9.1]).astype('float32')

class TestClipMaxAndMinAsInputsOneDNNOp(TestClipOneDNNOp):

    def set_additional_inputs(self):
        if False:
            print('Hello World!')
        self.inputs['Max'] = np.array([8.5]).astype('float32')
        self.inputs['Min'] = np.array([7.1]).astype('float32')

def create_bf16_test_class(parent):
    if False:
        i = 10
        return i + 15

    @OpTestTool.skip_if_not_cpu_bf16()
    class TestClipBF16OneDNNOp(parent):

        def set_inputs(self):
            if False:
                for i in range(10):
                    print('nop')
            self.x_fp32 = np.random.random((10, 10)).astype(np.float32) * 25
            self.inputs = {'X': convert_float_to_uint16(self.x_fp32)}

        def adjust_op_settings(self):
            if False:
                i = 10
                return i + 15
            self.dtype = np.uint16
            self.attrs['mkldnn_data_type'] = 'bfloat16'

        def calculate_grads(self):
            if False:
                while True:
                    i = 10
            self.dout = self.outputs['Out']
            self.dx = np.zeros(self.x_fp32.shape).astype('float32')
            for i in range(self.dx.shape[0]):
                for j in range(self.dx.shape[1]):
                    if self.x_fp32[j][i] > self.min and self.x_fp32[j][i] < self.max:
                        self.dx[j][i] = self.dout[j][i]

        def test_check_output(self):
            if False:
                i = 10
                return i + 15
            self.check_output_with_place(core.CPUPlace(), check_dygraph=False)

        def test_check_grad(self):
            if False:
                while True:
                    i = 10
            self.calculate_grads()
            self.check_grad_with_place(core.CPUPlace(), ['X'], 'Out', user_defined_grads=[self.dx], user_defined_grad_outputs=[convert_float_to_uint16(self.dout)], check_dygraph=False)
    cls_name = '{}_{}'.format(parent.__name__, 'BF16')
    TestClipBF16OneDNNOp.__name__ = cls_name
    globals()[cls_name] = TestClipBF16OneDNNOp
create_bf16_test_class(TestClipOneDNNOp)
create_bf16_test_class(TestClipMinAsInputOneDNNOp)
create_bf16_test_class(TestClipMaxAsInputOneDNNOp)
create_bf16_test_class(TestClipMaxAndMinAsInputsOneDNNOp)
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()