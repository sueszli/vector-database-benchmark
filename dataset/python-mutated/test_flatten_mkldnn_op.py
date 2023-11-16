import unittest
import numpy as np
from op_test import OpTest, OpTestTool, convert_float_to_uint16
import paddle
from paddle.base import core

@OpTestTool.skip_if_not_cpu_bf16()
class TestFlattenOneDNNOp(OpTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.set_op_type()
        self.init_test_case()
        self.set_inputs()
        self.attrs = {'axis': self.axis, 'use_mkldnn': True}
        self.ori_shape = self.inputs['X'].shape
        self.outputs = {'Out': self.inputs['X'].copy().reshape(self.new_shape)}

    def set_inputs(self):
        if False:
            for i in range(10):
                print('nop')
        self.inputs = {'X': np.random.random(self.in_shape).astype('float32')}

    def set_op_type(self):
        if False:
            print('Hello World!')
        self.op_type = 'flatten'

    def test_check_output(self):
        if False:
            print('Hello World!')
        self.check_output_with_place(core.CPUPlace())

    def test_check_grad(self):
        if False:
            while True:
                i = 10
        self.check_grad_with_place(core.CPUPlace(), ['X'], 'Out')

    def init_test_case(self):
        if False:
            while True:
                i = 10
        self.in_shape = (3, 2, 2, 10)
        self.axis = 1
        self.new_shape = (3, 40)

class TestFlattenOneDNNOp1(TestFlattenOneDNNOp):

    def init_test_case(self):
        if False:
            for i in range(10):
                print('nop')
        self.in_shape = (3, 2, 2, 10)
        self.axis = 0
        self.new_shape = (1, 120)

class TestFlattenOneDNNOpSixDims(TestFlattenOneDNNOp):

    def init_test_case(self):
        if False:
            for i in range(10):
                print('nop')
        self.in_shape = (3, 2, 3, 2, 4, 4)
        self.axis = 4
        self.new_shape = (36, 16)

class TestFlatten2OneDNNOp(TestFlattenOneDNNOp):

    def set_op_type(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_type = 'flatten2'

class TestFlatten2OneDNNOp1(TestFlattenOneDNNOp1):

    def set_op_type(self):
        if False:
            return 10
        self.op_type = 'flatten2'

class TestFlatten2OneDNNOpSixDims(TestFlattenOneDNNOpSixDims):

    def set_op_type(self):
        if False:
            return 10
        self.op_type = 'flatten2'

def create_flatten_bf16_test_classes(parent):
    if False:
        i = 10
        return i + 15

    class TestFlatten2BF16OneDNNOp(parent):

        def set_inputs(self):
            if False:
                while True:
                    i = 10
            self.dtype = np.uint16
            self.inputs = {'X': np.random.random(self.in_shape).astype('uint16')}

        def calculate_grads(self):
            if False:
                print('Hello World!')
            self.dout = self.outputs['Out']
            self.dx = np.reshape(self.dout, self.ori_shape)

        def test_check_output(self):
            if False:
                print('Hello World!')
            self.check_output_with_place(core.CPUPlace(), no_check_set=['XShape'])

        def test_check_grad(self):
            if False:
                print('Hello World!')
            self.calculate_grads()
            self.check_grad_with_place(core.CPUPlace(), ['X'], 'Out', user_defined_grads=[self.dx], user_defined_grad_outputs=[self.dout])
    cls_name = '{}_{}'.format(parent.__name__, 'Flatten2_BF16')
    TestFlatten2BF16OneDNNOp.__name__ = cls_name
    globals()[cls_name] = TestFlatten2BF16OneDNNOp

    class TestFlattenBF16OneDNNOp(parent):

        def set_op_type(self):
            if False:
                for i in range(10):
                    print('nop')
            self.dtype = np.uint16
            self.op_type = 'flatten'

        def set_inputs(self):
            if False:
                for i in range(10):
                    print('nop')
            self.dtype = np.uint16
            self.inputs = {'X': np.random.random(self.in_shape).astype('uint16')}

        def set_outputs(self):
            if False:
                i = 10
                return i + 15
            self.outputs = {'Out': self.x.reshape(self.new_shape)}

        def calculate_grads(self):
            if False:
                i = 10
                return i + 15
            self.dout = self.outputs['Out']
            self.dx = np.reshape(self.dout, self.ori_shape)

        def test_check_output(self):
            if False:
                return 10
            self.check_output_with_place(core.CPUPlace())

        def test_check_grad(self):
            if False:
                return 10
            self.calculate_grads()
            self.check_grad_with_place(core.CPUPlace(), ['X'], 'Out', user_defined_grads=[self.dx], user_defined_grad_outputs=[convert_float_to_uint16(self.dout)])
    cls_name = '{}_{}'.format(parent.__name__, 'Flatten_BF16')
    TestFlattenBF16OneDNNOp.__name__ = cls_name
    globals()[cls_name] = TestFlattenBF16OneDNNOp
create_flatten_bf16_test_classes(TestFlatten2OneDNNOp)
create_flatten_bf16_test_classes(TestFlatten2OneDNNOp1)
create_flatten_bf16_test_classes(TestFlatten2OneDNNOpSixDims)
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()