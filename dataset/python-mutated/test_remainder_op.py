import numpy as np
from cinn.common import is_compiled_with_cuda
from cinn.frontend import NetBuilder
from op_test import OpTest, OpTestTool
from op_test_helper import TestCaseHelper
import paddle

@OpTestTool.skip_if(not is_compiled_with_cuda(), 'x86 test will be skipped due to timeout.')
class TestRemainderOp(OpTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.prepare_inputs()

    def prepare_inputs(self):
        if False:
            return 10
        self.x_np = self.random(shape=self.case['x_shape'], dtype=self.case['dtype'], low=self.case['x_low'], high=self.case['x_high'])
        self.y_np = self.random(shape=self.case['y_shape'], dtype=self.case['dtype'], low=self.case['y_low'], high=self.case['y_high'])
        self.y_np = np.where(self.y_np == 0, 1, self.y_np)

    def build_paddle_program(self, target):
        if False:
            return 10
        x = paddle.to_tensor(self.x_np, stop_gradient=False)
        y = paddle.to_tensor(self.y_np, stop_gradient=False)
        out = paddle.remainder(x, y)
        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        if False:
            while True:
                i = 10
        builder = NetBuilder('remainder')
        x = builder.create_input(self.nptype2cinntype(self.x_np.dtype), self.x_np.shape, 'x')
        y = builder.create_input(self.nptype2cinntype(self.y_np.dtype), self.y_np.shape, 'y')
        out = builder.remainder(x, y)
        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x, y], [self.x_np, self.y_np], [out])
        self.cinn_outputs = res

    def test_check_results(self):
        if False:
            i = 10
            return i + 15
        self.check_outputs_and_grads()

class TestRemainderOpShape(TestCaseHelper):

    def init_attrs(self):
        if False:
            return 10
        self.class_name = 'TestRemainderOpCase'
        self.cls = TestRemainderOp
        self.inputs = [{'x_shape': [1], 'y_shape': [1]}, {'x_shape': [1024], 'y_shape': [1024]}, {'x_shape': [512, 256], 'y_shape': [512, 256]}, {'x_shape': [128, 64, 32], 'y_shape': [128, 64, 32]}, {'x_shape': [16, 8, 4, 2], 'y_shape': [16, 8, 4, 2]}, {'x_shape': [16, 8, 4, 2, 1], 'y_shape': [16, 8, 4, 2, 1]}]
        self.dtypes = [{'dtype': 'int32'}]
        self.attrs = [{'x_low': -10, 'x_high': 10, 'y_low': -10, 'y_high': 10}]

class TestRemainderOpDtype(TestCaseHelper):

    def init_attrs(self):
        if False:
            while True:
                i = 10
        self.class_name = 'TestRemainderOpCase'
        self.cls = TestRemainderOp
        self.inputs = [{'x_shape': [1024], 'y_shape': [1024]}]
        self.dtypes = [{'dtype': 'int32'}, {'dtype': 'int64'}, {'dtype': 'float16'}, {'dtype': 'float32'}, {'dtype': 'float64'}]
        self.attrs = [{'x_low': -10, 'x_high': 10, 'y_low': -10, 'y_high': 10}]

class TestRemainderOpBroadcast(TestCaseHelper):

    def init_attrs(self):
        if False:
            return 10
        self.class_name = 'TestRemainderOpCase'
        self.cls = TestRemainderOp
        self.inputs = [{'x_shape': [1], 'y_shape': [1]}, {'x_shape': [1024], 'y_shape': [1]}, {'x_shape': [512, 256], 'y_shape': [1, 256]}, {'x_shape': [128, 64, 32], 'y_shape': [128, 1, 32]}, {'x_shape': [16, 8, 4, 2], 'y_shape': [16, 1, 1, 2]}, {'x_shape': [16, 8, 4, 2, 1], 'y_shape': [16, 1, 1, 1, 1]}]
        self.dtypes = [{'dtype': 'int32'}]
        self.attrs = [{'x_low': -10, 'x_high': 10, 'y_low': -10, 'y_high': 10}]
if __name__ == '__main__':
    TestRemainderOpShape().run()
    TestRemainderOpDtype().run()
    TestRemainderOpBroadcast().run()