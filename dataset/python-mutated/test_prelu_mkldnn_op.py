import unittest
import numpy as np
from op_test import OpTest, OpTestTool, convert_float_to_uint16
import paddle
from paddle.base import core

def ref_prelu(x, weight, mode):
    if False:
        return 10
    result = x.copy()
    if mode == 'all':
        result = np.where(x > 0, x, x * weight[0])
    elif mode == 'channel':
        if len(weight.shape) > 1:
            for i in range(x.shape[1]):
                result[:, i] = np.where(x[:, i] > 0, x[:, i], x[:, i] * weight[0, i])
        else:
            for i in range(x.shape[1]):
                result[:, i] = np.where(x[:, i] > 0, x[:, i], x[:, i] * weight[i])
    elif mode == 'element':
        result = np.where(x[:] > 0, x[:], x[:] * weight)
    return result

class TestPReluModeChannelOneDNNOp(OpTest):

    def init_attrs(self):
        if False:
            i = 10
            return i + 15
        self.mode = 'element'
        self.alpha = np.random.random((1, 4, 5, 5)).astype('float32')

    def set_dtype_attr(self):
        if False:
            return 10
        pass

    def set_inputs(self):
        if False:
            return 10
        self.inputs = {'X': self.x, 'Alpha': self.alpha}

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_type = 'prelu'
        self.x = np.random.random((2, 4, 5, 5)).astype('float32') + 1
        self.init_attrs()
        self.set_inputs()
        self.attrs = {'mode': self.mode, 'use_mkldnn': True}
        self.set_dtype_attr()
        self.outputs = {'Out': ref_prelu(self.x, self.alpha, self.mode)}

    def test_check_output(self):
        if False:
            while True:
                i = 10
        self.check_output(check_dygraph=False)

    def test_check_grad(self):
        if False:
            return 10
        self.check_grad(['X', 'Alpha'], 'Out', check_dygraph=False)

class TestPReluModeAllOneDNNOp(TestPReluModeChannelOneDNNOp):

    def init_attrs(self):
        if False:
            print('Hello World!')
        self.mode = 'all'
        self.alpha = np.random.random((1, 1, 1, 1)).astype('float32')

    def test_check_grad(self):
        if False:
            i = 10
            return i + 15
        self.check_grad(['X'], 'Out', check_dygraph=False)

class TestPReluModeElementOneDNNOp(TestPReluModeChannelOneDNNOp):

    def init_attrs(self):
        if False:
            return 10
        self.mode = 'element'
        self.alpha = np.random.random((1, 4, 5, 5)).astype('float32')

class TestPReluModeElement0DOneDNNOp(TestPReluModeChannelOneDNNOp):

    def init_attrs(self):
        if False:
            i = 10
            return i + 15
        self.mode = 'all'
        self.alpha = np.random.random(()).astype('float32')

    def setUp(self):
        if False:
            while True:
                i = 10
        self.op_type = 'prelu'
        self.x = np.random.random(()).astype('float32')
        self.init_attrs()
        self.set_inputs()
        self.attrs = {'mode': self.mode, 'use_mkldnn': True}
        self.set_dtype_attr()
        self.outputs = {'Out': self.x if self.x > 0 else self.x * self.alpha}

class TestPReluModeChannel3DOneDNNOp(TestPReluModeChannelOneDNNOp):

    def init_attrs(self):
        if False:
            while True:
                i = 10
        self.mode = 'channel'
        self.x = np.random.random((1, 100, 1)).astype('float32')
        self.alpha = np.random.random((1, 100, 1)).astype('float32')

class TestPReluModeChannelAlpha1DOneDNNOp(TestPReluModeChannelOneDNNOp):

    def init_attrs(self):
        if False:
            while True:
                i = 10
        self.mode = 'channel'
        self.x = np.random.random((1, 100, 1)).astype('float32')
        self.alpha = np.random.random(100).astype('float32')

class TestPReluModeAllAlpha1DOneDNNOp(TestPReluModeAllOneDNNOp):

    def init_attrs(self):
        if False:
            print('Hello World!')
        self.mode = 'channel'
        self.x = np.random.random((1, 1, 100)).astype('float32')
        self.alpha = np.random.random(1).astype('float32')

def create_bf16_test_class(parent):
    if False:
        i = 10
        return i + 15

    @OpTestTool.skip_if_not_cpu_bf16()
    class TestPReluBF16OneDNNOp(parent):

        def set_inputs(self):
            if False:
                i = 10
                return i + 15
            self.inputs = {'X': convert_float_to_uint16(self.x), 'Alpha': convert_float_to_uint16(self.alpha)}

        def set_dtype_attr(self):
            if False:
                print('Hello World!')
            self.attrs['mkldnn_data_type'] = 'bfloat16'

        def calculate_grads(self):
            if False:
                return 10
            dout = self.outputs['Out']
            self.dx = self.x.copy()
            self.dalpha = self.alpha.copy()
            if self.mode == 'all':
                self.dx = np.where(self.x > 0, dout, dout * self.alpha[0])
            elif self.mode == 'channel':
                if len(self.alpha.shape) > 1:
                    for i in range(self.x.shape[1]):
                        self.dx[:, i] = np.where(self.x[:, i] > 0, dout[:, i], dout[:, i] * self.alpha[0, i])
                else:
                    for i in range(self.x.shape[1]):
                        self.dx[:, i] = np.where(self.x[:, i] > 0, dout[:, i], dout[:, i] * self.alpha[i])
            elif self.mode == 'element':
                self.dx = np.where(self.x[:] > 0, dout[:], dout[:] * self.alpha)
            self.dalpha = np.where(self.x < 0, dout * self.x, 0)
            self.dout = dout

        def test_check_output(self):
            if False:
                for i in range(10):
                    print('nop')
            self.check_output_with_place(core.CPUPlace(), check_dygraph=False)

        def test_check_grad(self):
            if False:
                return 10
            self.calculate_grads()
            self.check_grad_with_place(core.CPUPlace(), ['X', 'Alpha'], 'Out', user_defined_grads=[self.dx, self.dalpha], user_defined_grad_outputs=[convert_float_to_uint16(self.dout)], check_dygraph=False)
    cls_name = '{}_{}'.format(parent.__name__, 'BF16')
    TestPReluBF16OneDNNOp.__name__ = cls_name
    globals()[cls_name] = TestPReluBF16OneDNNOp
create_bf16_test_class(TestPReluModeChannelOneDNNOp)
create_bf16_test_class(TestPReluModeElementOneDNNOp)
create_bf16_test_class(TestPReluModeChannel3DOneDNNOp)
create_bf16_test_class(TestPReluModeChannelAlpha1DOneDNNOp)
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()