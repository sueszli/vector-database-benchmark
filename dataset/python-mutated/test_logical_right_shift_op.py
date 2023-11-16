import numpy as np
from cinn.common import is_compiled_with_cuda
from cinn.frontend import NetBuilder
from op_test import OpTest, OpTestTool
from op_test_helper import TestCaseHelper
import paddle

@OpTestTool.skip_if(not is_compiled_with_cuda(), 'x86 test will be skipped due to timeout.')
class TestLogicalRightShift(OpTest):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.prepare_inputs()

    def prepare_inputs(self):
        if False:
            i = 10
            return i + 15
        iinfo = np.iinfo(self.case['dtype'])
        self.x_np = self.random(shape=self.case['shape'], dtype=self.case['dtype'], low=0, high=iinfo.max)
        self.y_np = self.random(shape=self.case['shape'], dtype=self.case['dtype'], low=0, high=iinfo.bits)

    def build_paddle_program(self, target):
        if False:
            print('Hello World!')
        out_np = np.right_shift(self.x_np, self.y_np)
        out = paddle.to_tensor(out_np, stop_gradient=True)
        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        if False:
            return 10
        builder = NetBuilder('logical_right_shift')
        x = builder.create_input(self.nptype2cinntype(self.x_np.dtype), self.x_np.shape, 'x')
        y = builder.create_input(self.nptype2cinntype(self.y_np.dtype), self.y_np.shape, 'y')
        out = builder.logical_right_shift(x, y)
        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x, y], [self.x_np, self.y_np], [out])
        self.cinn_outputs = res

    def test_check_results(self):
        if False:
            i = 10
            return i + 15
        self.check_outputs_and_grads(all_equal=True)

class TestLogicalRightShiftShape(TestCaseHelper):

    def init_attrs(self):
        if False:
            i = 10
            return i + 15
        self.class_name = 'TestLogicalRightShiftCase'
        self.cls = TestLogicalRightShift
        self.inputs = [{'shape': [1]}, {'shape': [1024]}, {'shape': [512, 256]}, {'shape': [128, 64, 32]}, {'shape': [16, 8, 4, 2]}, {'shape': [16, 8, 4, 2, 1]}]
        self.dtypes = [{'dtype': 'int32'}]
        self.attrs = []

class TestLogicalRightShiftDtype(TestCaseHelper):

    def init_attrs(self):
        if False:
            while True:
                i = 10
        self.class_name = 'TestLogicalRightShiftCase'
        self.cls = TestLogicalRightShift
        self.inputs = [{'shape': [1024]}]
        self.dtypes = [{'dtype': 'uint8'}, {'dtype': 'int8'}, {'dtype': 'int16'}, {'dtype': 'int32'}, {'dtype': 'int64'}]
        self.attrs = []
if __name__ == '__main__':
    TestLogicalRightShiftShape().run()
    TestLogicalRightShiftDtype().run()