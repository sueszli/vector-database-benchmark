import unittest
import numpy as np
from op_test import OpTest, OpTestTool

@OpTestTool.skip_if_not_cpu()
class TestFCINT8OneDNNOp(OpTest):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.op_type = 'fc'
        self._cpu_only = True
        self.configure()
        self.set_shape()
        self.generate_data()
        self.set_inputs()
        y_scales_size = self.bias_shape if self.per_channel_quantize_weight else 1
        self.attrs = {'use_mkldnn': True, 'Scale_in': self.x_scale, 'Scale_weights': [self.y_scale] * y_scales_size, 'Scale_out': self.out_scale, 'force_fp32_output': self.force_fp32_output, 'in_num_col_dims': self.in_num_col_dims}
        if self.force_fp32_output:
            out = self.out_float
        else:
            out = self.out
        self.outputs = {'Out': out}

    def configure(self):
        if False:
            for i in range(10):
                print('nop')
        self.use_bias = True
        self.force_fp32_output = False
        self.in_num_col_dims = 1
        self.per_channel_quantize_weight = False

    def set_shape(self):
        if False:
            while True:
                i = 10
        self.input_shape = (10, 5)
        self.weight_shape = (5, 10)
        self.bias_shape = 10

    def set_inputs(self):
        if False:
            return 10
        self.inputs = {'Input': self.x, 'W': self.y_float, 'Bias': self.bias}

    def quantize(self, tensor):
        if False:
            i = 10
            return i + 15
        scale = 63.0 / np.abs(np.amax(tensor))
        quantized = np.round(scale * tensor).astype('int8')
        return (scale, quantized)

    def generate_data(self):
        if False:
            for i in range(10):
                print('nop')
        self.x_float = np.random.random(self.input_shape).astype('float32') * 10
        (self.x_scale, self.x) = self.quantize(self.x_float)
        self.y_float = np.random.random(self.weight_shape).astype('float32') * 10
        (self.y_scale, self.y) = self.quantize(self.y_float)
        flatten_shape = [1, 1]
        for i in range(len(self.input_shape)):
            if i < self.in_num_col_dims:
                flatten_shape[0] *= self.input_shape[i]
            else:
                flatten_shape[1] *= self.input_shape[i]
        self.out_float = np.dot(self.x_float.reshape(flatten_shape), self.y_float)
        if self.use_bias:
            self.bias = np.random.random(self.bias_shape).astype('float32') * 10
            self.out_float += self.bias
        (self.out_scale, self.out) = self.quantize(self.out_float)

    def test_check_output(self):
        if False:
            while True:
                i = 10
        int_atol = 2
        self.check_output(int_atol)

class TestFCINT8NoBiasOneDNNOp(TestFCINT8OneDNNOp):

    def configure(self):
        if False:
            return 10
        self.use_bias = False
        self.force_fp32_output = False
        self.in_num_col_dims = 1
        self.per_channel_quantize_weight = False

    def set_inputs(self):
        if False:
            i = 10
            return i + 15
        self.inputs = {'Input': self.x, 'W': self.y_float}

class TestFCINT8ForceFP32OutputOneDNNOp(TestFCINT8NoBiasOneDNNOp):

    def configure(self):
        if False:
            while True:
                i = 10
        self.use_bias = False
        self.force_fp32_output = True
        self.in_num_col_dims = 1
        self.per_channel_quantize_weight = False

class TestFCINT8ForceFP32OutputPerChannelWeightOneDNNOp(TestFCINT8OneDNNOp):

    def configure(self):
        if False:
            while True:
                i = 10
        self.use_bias = True
        self.force_fp32_output = True
        self.in_num_col_dims = 1
        self.per_channel_quantize_weight = True

    def set_shape(self):
        if False:
            for i in range(10):
                print('nop')
        self.input_shape = (1, 8, 1, 1)
        self.weight_shape = (8, 10)
        self.bias_shape = 10
if __name__ == '__main__':
    import paddle
    paddle.enable_static()
    unittest.main()