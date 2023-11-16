from cinn.common import is_compiled_with_cudnn
from cinn.frontend import NetBuilder
from op_test import OpTest, OpTestTool
from op_test_helper import TestCaseHelper
import paddle
from paddle import nn

@OpTestTool.skip_if(not is_compiled_with_cudnn(), 'x86 test will be skipped due to timeout.')
class TestDepthwiseConv2dOp(OpTest):

    def setUp(self):
        if False:
            print('Hello World!')
        self.prepare_inputs()

    def prepare_inputs(self):
        if False:
            return 10
        self.x_np = self.random(shape=self.case['x_shape'], dtype=self.case['dtype'])
        self.w_np = self.random(shape=self.case['w_shape'], dtype=self.case['dtype'])

    def build_paddle_program(self, target):
        if False:
            return 10
        x = paddle.to_tensor(self.x_np, stop_gradient=False)
        weight = nn.initializer.Assign(self.w_np)
        if self.case['data_format'] == 'NCHW':
            c_axis = 1
        elif self.case['data_format'] == 'NHWC':
            c_axis = 3
        else:
            raise ValueError('Unknown data_format')
        conv = nn.Conv2D(in_channels=self.case['x_shape'][c_axis], out_channels=self.case['x_shape'][c_axis], kernel_size=self.case['kernel_size'], stride=self.case['stride'], padding=self.case['padding'], dilation=self.case['dilation'], groups=self.case['groups'], weight_attr=weight, bias_attr=False, data_format=self.case['data_format'])
        y = conv(x)
        self.paddle_outputs = [y]

    def build_cinn_program(self, target):
        if False:
            while True:
                i = 10
        builder = NetBuilder('depthwise_conv2d')
        x = builder.create_input(self.nptype2cinntype(self.case['dtype']), self.case['x_shape'], 'x')
        weight = builder.create_input(self.nptype2cinntype(self.case['dtype']), self.case['w_shape'], 'weight')
        if self.case['data_format'] == 'NCHW':
            y = builder.depthwise_conv2d(x, weight, strides=self.case['stride'], paddings=self.case['padding'], dilations=self.case['dilation'], groups=self.case['groups'], data_format=self.case['data_format'])
        elif self.case['data_format'] == 'NHWC':
            weight_t = builder.transpose(weight, [0, 2, 3, 1])
            y = builder.depthwise_conv2d(x, weight_t, strides=self.case['stride'], paddings=self.case['padding'], dilations=self.case['dilation'], groups=self.case['groups'], data_format=self.case['data_format'])
        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x, weight], [self.x_np, self.w_np], [y], passes=[])
        self.cinn_outputs = res

    def test_check_results(self):
        if False:
            for i in range(10):
                print('nop')
        max_relative_error = self.case['max_relative_error'] if 'max_relative_error' in self.case else 1e-05
        self.check_outputs_and_grads(max_relative_error=max_relative_error)

class TestDepthwiseConv2dOpShape(TestCaseHelper):

    def init_attrs(self):
        if False:
            for i in range(10):
                print('nop')
        self.class_name = 'TestDepthwiseConv2dCase'
        self.cls = TestDepthwiseConv2dOp
        self.inputs = [{'x_shape': [3, 16, 32, 32], 'w_shape': [16, 1, 3, 3], 'data_format': 'NCHW', 'groups': 16}, {'x_shape': [3, 16, 64, 64], 'w_shape': [16, 1, 3, 3], 'data_format': 'NCHW', 'groups': 16}, {'x_shape': [3, 32, 32, 16], 'w_shape': [16, 1, 3, 3], 'data_format': 'NHWC', 'groups': 16}, {'x_shape': [3, 64, 64, 16], 'w_shape': [16, 1, 3, 3], 'data_format': 'NHWC', 'groups': 16}]
        self.dtypes = [{'dtype': 'float32'}]
        self.attrs = [{'kernel_size': [3, 3], 'stride': [1, 1], 'padding': [0, 0], 'dilation': [1, 1]}]

class TestDepthwiseConv2dOpAttr(TestCaseHelper):

    def init_attrs(self):
        if False:
            print('Hello World!')
        self.class_name = 'TestDepthwiseConv2dCase'
        self.cls = TestDepthwiseConv2dOp
        self.inputs = [{'x_shape': [3, 16, 32, 32], 'w_shape': [16, 1, 3, 3], 'data_format': 'NCHW', 'groups': 16}]
        self.dtypes = [{'dtype': 'float32'}]
        self.attrs = [{'kernel_size': [5, 5], 'stride': [1, 1], 'padding': [0, 0], 'dilation': [1, 1]}, {'kernel_size': [3, 3], 'stride': [2, 2], 'padding': [0, 0], 'dilation': [1, 1]}, {'kernel_size': [3, 3], 'stride': [1, 1], 'padding': [1, 1], 'dilation': [1, 1]}, {'kernel_size': [3, 3], 'stride': [1, 1], 'padding': [0, 0], 'dilation': [2, 2]}]
if __name__ == '__main__':
    TestDepthwiseConv2dOpShape().run()
    TestDepthwiseConv2dOpAttr().run()