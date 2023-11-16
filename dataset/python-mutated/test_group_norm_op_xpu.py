import unittest
import numpy as np
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
from op_test import OpTest
from op_test_xpu import XPUOpTest
import paddle
paddle.enable_static()

def group_norm_naive(x, scale, bias, epsilon, groups, data_layout):
    if False:
        for i in range(10):
            print('nop')
    if data_layout == 'NHWC':
        x = np.transpose(x, (0, 3, 1, 2))
    (N, C, H, W) = x.shape
    G = groups
    x = x.reshape((N * G, -1))
    mean = np.mean(x, axis=1, keepdims=True)
    var = np.var(x, axis=1, keepdims=True)
    output = (x - mean) / np.sqrt(var + epsilon)
    output = output.reshape((N, C, H, W)) * scale.reshape((-1, 1, 1)) + bias.reshape((-1, 1, 1))
    if data_layout == 'NHWC':
        output = np.transpose(output, (0, 2, 3, 1))
    return (output, mean.reshape((N, G)), var.reshape((N, G)))

class XPUTestGroupNormOp(XPUOpTestWrapper):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.op_name = 'group_norm'
        self.use_dynamic_create_class = False

    class TestGroupNormOp(XPUOpTest):

        def init_test_case(self):
            if False:
                return 10
            self.data_format = 'NCHW'
            self.attrs = {'epsilon': 1e-05, 'groups': 2, 'data_layout': 'NCHW'}

        def setUp(self):
            if False:
                i = 10
                return i + 15
            'Test GroupNorm Op with supplied attributes'
            self.__class__.op_type = 'group_norm'
            self.dtype = self.in_type
            self.shape = (2, 100, 3, 5)
            self.init_test_case()
            input = np.random.random(self.shape).astype(self.dtype)
            if self.data_format == 'NHWC':
                input = np.transpose(input, (0, 2, 3, 1))
            scale = np.random.random([self.shape[1]]).astype(self.dtype)
            bias = np.random.random([self.shape[1]]).astype(self.dtype)
            (output, mean, var) = group_norm_naive(input, scale, bias, self.attrs['epsilon'], self.attrs['groups'], self.data_format)
            self.inputs = {'X': OpTest.np_dtype_to_base_dtype(input), 'Scale': OpTest.np_dtype_to_base_dtype(scale), 'Bias': OpTest.np_dtype_to_base_dtype(bias)}
            self.outputs = {'Y': output, 'Mean': mean, 'Variance': var}
            self.attrs['data_layout'] = self.data_format

        def test_check_output(self):
            if False:
                i = 10
                return i + 15
            self.check_output_with_place(paddle.XPUPlace(0))

        def test_check_grad(self):
            if False:
                while True:
                    i = 10
            self.check_grad_with_place(paddle.XPUPlace(0), ['X', 'Scale', 'Bias'], 'Y')

    class TestGroupNormOp2(TestGroupNormOp):

        def init_test_case(self):
            if False:
                print('Hello World!')
            self.data_format = 'NHWC'
            self.attrs = {'epsilon': 1e-05, 'groups': 2, 'data_layout': 'NHWC'}
support_types = get_xpu_op_support_types('group_norm')
for stype in support_types:
    create_test_class(globals(), XPUTestGroupNormOp, stype)
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()