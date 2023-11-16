import numpy as np
from cinn.common import is_compiled_with_cuda
from cinn.frontend import NetBuilder
from op_test import OpTest, OpTestTool
from op_test_helper import TestCaseHelper
import paddle

@OpTestTool.skip_if(not is_compiled_with_cuda(), 'x86 test will be skipped due to timeout.')
class TestLeftShiftOp(OpTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        print(f'\nRunning {self.__class__.__name__}: {self.case}')
        self.prepare_inputs()

    def prepare_inputs(self):
        if False:
            for i in range(10):
                print('nop')
        self.x_np = self.random(shape=self.case['x_shape'], dtype=self.case['x_dtype'], low=-100, high=100)
        self.y_np = self.random(shape=self.case['y_shape'], dtype=self.case['y_dtype'], low=0, high=16)

    def build_paddle_program(self, target):
        if False:
            while True:
                i = 10
        np_out = np.left_shift(self.x_np, self.y_np)
        out = paddle.to_tensor(np_out, stop_gradient=True)
        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        if False:
            return 10
        builder = NetBuilder('left_shift')
        x = builder.create_input(self.nptype2cinntype(self.case['x_dtype']), self.case['x_shape'], 'x')
        y = builder.create_input(self.nptype2cinntype(self.case['y_dtype']), self.case['y_shape'], 'y')
        out = builder.left_shift(x, y)
        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x, y], [self.x_np, self.y_np], [out])
        self.cinn_outputs = [res[0]]

    def test_check_results(self):
        if False:
            while True:
                i = 10
        self.check_outputs_and_grads()

class TestLeftShiftAll(TestCaseHelper):

    def init_attrs(self):
        if False:
            print('Hello World!')
        self.class_name = 'TestLeftShiftOpCase'
        self.cls = TestLeftShiftOp
        self.inputs = [{'x_shape': [1], 'y_shape': [1]}, {'x_shape': [1024], 'y_shape': [1024]}, {'x_shape': [512, 256], 'y_shape': [512, 256]}, {'x_shape': [128, 64, 32], 'y_shape': [128, 64, 32]}, {'x_shape': [16, 8, 4, 2], 'y_shape': [16, 8, 4, 2]}, {'x_shape': [16, 8, 4, 2, 1], 'y_shape': [16, 8, 4, 2, 1]}]
        self.dtypes = [{'x_dtype': 'int32', 'y_dtype': 'int32'}]
        self.attrs = []

class TestLeftShiftAllWithBroadcast(TestCaseHelper):

    def init_attrs(self):
        if False:
            for i in range(10):
                print('nop')
        self.class_name = 'TestLeftShiftOpCase'
        self.cls = TestLeftShiftOp
        self.inputs = [{'x_shape': [1], 'y_shape': [1]}, {'x_shape': [1024], 'y_shape': [1]}, {'x_shape': [512, 256], 'y_shape': [1, 1]}, {'x_shape': [128, 64, 32], 'y_shape': [1, 1, 1]}, {'x_shape': [16, 8, 4, 2], 'y_shape': [1, 1, 1, 1]}, {'x_shape': [16, 8, 4, 2, 1], 'y_shape': [1, 1, 1, 1, 1]}]
        self.dtypes = [{'x_dtype': 'int32', 'y_dtype': 'int32'}]
        self.attrs = []
if __name__ == '__main__':
    TestLeftShiftAll().run()
    TestLeftShiftAllWithBroadcast().run()