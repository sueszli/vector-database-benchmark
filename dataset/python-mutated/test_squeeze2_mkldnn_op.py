import unittest
import numpy as np
from op_test import OpTest, OpTestTool, convert_float_to_uint16
import paddle
from paddle.base import core

@OpTestTool.skip_if(core.is_compiled_with_cuda(), 'CUDA has to be skipped because it forces dygraph')
class TestSqueeze2OneDNNOp(OpTest):

    def set_op_type(self):
        if False:
            return 10
        self.op_type = 'squeeze2'

    def init_test_case(self):
        if False:
            while True:
                i = 10
        self.ori_shape = (1, 3, 1, 40)
        self.axes = (0, 2)
        self.new_shape = (3, 40)

    def set_inputs(self):
        if False:
            return 10
        self.inputs = {'X': self.x}

    def init_attrs(self):
        if False:
            return 10
        self.attrs = {'axes': self.axes, 'use_mkldnn': True}

    def set_outputs(self):
        if False:
            i = 10
            return i + 15
        self.outputs = {'Out': self.x.reshape(self.new_shape), 'XShape': np.random.random(self.ori_shape).astype('float32')}

    def setUp(self):
        if False:
            print('Hello World!')
        self.set_op_type()
        self.init_test_case()
        self.x = np.random.random(self.ori_shape).astype('float32')
        self.set_inputs()
        self.init_attrs()
        self.set_outputs()

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_output_with_place(core.CPUPlace(), no_check_set=['XShape'])

    def test_check_grad(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_grad_with_place(core.CPUPlace(), ['X'], 'Out')

class TestSqueezeOneDNNOp(TestSqueeze2OneDNNOp):

    def set_op_type(self):
        if False:
            while True:
                i = 10
        self.op_type = 'squeeze'

    def set_outputs(self):
        if False:
            for i in range(10):
                print('nop')
        self.outputs = {'Out': self.x.reshape(self.new_shape)}

    def test_check_output(self):
        if False:
            print('Hello World!')
        self.check_output_with_place(core.CPUPlace())

class TestSqueeze2OneDNNOp_ZeroDim(TestSqueeze2OneDNNOp):

    def init_test_case(self):
        if False:
            print('Hello World!')
        self.ori_shape = [1]
        self.axes = ()
        self.new_shape = ()

class TestSqueezeOneDNNOp_ZeroDim(TestSqueezeOneDNNOp):

    def init_test_case(self):
        if False:
            print('Hello World!')
        self.ori_shape = [1]
        self.axes = ()
        self.new_shape = ()

class TestSqueeze2OneDNNOp1(TestSqueeze2OneDNNOp):

    def init_test_case(self):
        if False:
            print('Hello World!')
        self.ori_shape = (1, 20, 1, 5)
        self.axes = (0, -2)
        self.new_shape = (20, 5)

class TestSqueezeOneDNNOp1(TestSqueezeOneDNNOp):

    def init_test_case(self):
        if False:
            for i in range(10):
                print('nop')
        self.ori_shape = (1, 20, 1, 5)
        self.axes = (0, -2)
        self.new_shape = (20, 5)

class TestSqueeze2OneDNNOp2(TestSqueeze2OneDNNOp):

    def init_test_case(self):
        if False:
            i = 10
            return i + 15
        self.ori_shape = (1, 20, 1, 5)
        self.axes = ()
        self.new_shape = (20, 5)

class TestSqueezeOneDNNOp2(TestSqueezeOneDNNOp):

    def init_test_case(self):
        if False:
            return 10
        self.ori_shape = (1, 20, 1, 5)
        self.axes = ()
        self.new_shape = (20, 5)

class TestSqueeze2OneDNNOp3(TestSqueeze2OneDNNOp):

    def init_test_case(self):
        if False:
            i = 10
            return i + 15
        self.ori_shape = (25, 1, 1, 4, 1)
        self.axes = (1, -1)
        self.new_shape = (25, 1, 4)

class TestSqueeze2OneDNNOp4(TestSqueeze2OneDNNOp):

    def set_outputs(self):
        if False:
            while True:
                i = 10
        self.outputs = {'Out': self.x.reshape(self.new_shape)}

    def init_test_case(self):
        if False:
            while True:
                i = 10
        self.ori_shape = (25, 1, 1, 4, 1)
        self.axes = (1, -1)
        self.new_shape = (25, 1, 4)

class TestSqueezeOneDNNOp3(TestSqueezeOneDNNOp):

    def init_test_case(self):
        if False:
            i = 10
            return i + 15
        self.ori_shape = (25, 1, 1, 4, 1)
        self.axes = (1, -1)
        self.new_shape = (25, 1, 4)

def create_squeeze_bf16_test_classes(parent):
    if False:
        for i in range(10):
            print('nop')

    @OpTestTool.skip_if_not_cpu_bf16()
    class TestSqueeze2BF16OneDNNOp(parent):

        def set_inputs(self):
            if False:
                return 10
            self.dtype = np.uint16
            self.inputs = {'X': convert_float_to_uint16(self.x)}

        def calculate_grads(self):
            if False:
                return 10
            self.dout = self.outputs['Out']
            self.dx = np.reshape(self.dout, self.ori_shape)

        def test_check_grad(self):
            if False:
                i = 10
                return i + 15
            self.calculate_grads()
            self.check_grad_with_place(core.CPUPlace(), ['X'], 'Out', user_defined_grads=[self.dx], user_defined_grad_outputs=[self.dout])
    cls_name = '{}_{}'.format(parent.__name__, 'Squeeze2_BF16')
    TestSqueeze2BF16OneDNNOp.__name__ = cls_name
    globals()[cls_name] = TestSqueeze2BF16OneDNNOp

    class TestSqueezeBF16OneDNNOp(TestSqueeze2BF16OneDNNOp):

        def set_op_type(self):
            if False:
                return 10
            self.dtype = np.uint16
            self.op_type = 'squeeze'

        def set_outputs(self):
            if False:
                return 10
            self.outputs = {'Out': self.x.reshape(self.new_shape)}

        def test_check_output(self):
            if False:
                return 10
            self.check_output_with_place(core.CPUPlace())
    cls_name = '{}_{}'.format(parent.__name__, 'Squeeze_BF16')
    TestSqueezeBF16OneDNNOp.__name__ = cls_name
    globals()[cls_name] = TestSqueezeBF16OneDNNOp
create_squeeze_bf16_test_classes(TestSqueeze2OneDNNOp)
create_squeeze_bf16_test_classes(TestSqueeze2OneDNNOp1)
create_squeeze_bf16_test_classes(TestSqueeze2OneDNNOp2)
create_squeeze_bf16_test_classes(TestSqueeze2OneDNNOp3)
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()