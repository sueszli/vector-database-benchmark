import numpy as np
from cinn.common import is_compiled_with_cuda
from cinn.frontend import NetBuilder
from op_test import OpTest, OpTestTool
from op_test_helper import TestCaseHelper
import paddle

@OpTestTool.skip_if(not is_compiled_with_cuda(), 'x86 test will be skipped due to timeout.')
class TestCbrtOp(OpTest):

    def setUp(self):
        if False:
            return 10
        print(f'\nRunning {self.__class__.__name__}: {self.case}')
        self.inputs = {}
        self.prepare_inputs()

    def prepare_inputs(self):
        if False:
            for i in range(10):
                print('nop')
        self.inputs = {'x': self.random(self.case['shape'], self.case['dtype'], -100.0, 100.0)}

    def build_paddle_program(self, target):
        if False:
            for i in range(10):
                print('nop')
        numpy_out = np.cbrt(self.inputs['x'])
        out = paddle.to_tensor(numpy_out, stop_gradient=False)
        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        if False:
            return 10
        builder = NetBuilder('cbrt')
        x = builder.create_input(self.nptype2cinntype(self.inputs['x'].dtype), self.inputs['x'].shape, 'x')
        out = builder.cbrt(x)
        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x], [self.inputs['x']], [out])
        self.cinn_outputs = res

    def test_check_results(self):
        if False:
            while True:
                i = 10
        self.check_outputs_and_grads(max_relative_error=0.001, max_absolute_error=0.001)

class TestCbrtOpShape(TestCaseHelper):

    def init_attrs(self):
        if False:
            print('Hello World!')
        self.class_name = 'TestCbrtOpShape'
        self.cls = TestCbrtOp
        self.inputs = [{'shape': [10]}, {'shape': [8, 5]}, {'shape': [10, 3, 5]}, {'shape': [80, 40, 5, 7]}, {'shape': [80, 1, 5, 7]}, {'shape': [80, 3, 1024, 7]}, {'shape': [10, 5, 1024, 2048]}, {'shape': [1]}, {'shape': [512]}, {'shape': [1024]}, {'shape': [2048]}, {'shape': [1, 1, 1, 1]}]
        self.dtypes = [{'dtype': 'float32'}]
        self.attrs = []

class TestCbrtOpDtype(TestCaseHelper):

    def init_attrs(self):
        if False:
            for i in range(10):
                print('nop')
        self.class_name = 'TestCbrtOpDtype'
        self.cls = TestCbrtOp
        self.inputs = [{'shape': [1]}, {'shape': [5]}, {'shape': [80, 40, 5, 7]}]
        self.dtypes = [{'dtype': 'float16'}, {'dtype': 'float32'}, {'dtype': 'float64'}]
        self.attrs = []
if __name__ == '__main__':
    TestCbrtOpShape().run()
    TestCbrtOpDtype().run()