from cinn.common import is_compiled_with_cuda
from cinn.frontend import NetBuilder
from op_test import OpTest, OpTestTool
from op_test_helper import TestCaseHelper
import paddle
import paddle.nn.functional as F

@OpTestTool.skip_if(not is_compiled_with_cuda(), 'x86 test will be skipped due to timeout.')
class TestReluOp(OpTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        print(f'\nRunning {self.__class__.__name__}: {self.case}')
        self.inputs = {}
        self.prepare_inputs()

    def prepare_inputs(self):
        if False:
            return 10
        self.inputs = {'x': self.random(self.case['shape'], self.case['dtype'], -1.0, 1.0), 'dout': self.random(self.case['shape'], self.case['dtype'], -1.0, 1.0)}

    def build_paddle_program(self, target):
        if False:
            print('Hello World!')
        x = paddle.to_tensor(self.inputs['x'], stop_gradient=False)
        out = F.relu(x)
        self.paddle_outputs = [out]
        self.paddle_grads = self.get_paddle_grads([out], [x], [self.inputs['dout']])

    def build_cinn_program(self, target):
        if False:
            return 10
        builder = NetBuilder('relu')
        x = builder.create_input(self.nptype2cinntype(self.inputs['x'].dtype), self.inputs['x'].shape, 'x')
        out = builder.relu(x)
        dout = builder.create_input(self.nptype2cinntype(self.inputs['dout'].dtype), self.inputs['dout'].shape, 'dout')
        x_grad = builder.relu_grad(dout, out)
        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x, dout], [self.inputs['x'], self.inputs['dout']], [out, x_grad], passes=[])
        self.cinn_outputs = [res[0]]
        self.cinn_grads = [res[1]]

    def test_check_results(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_outputs_and_grads()

class TestReluOpShape(TestCaseHelper):

    def init_attrs(self):
        if False:
            i = 10
            return i + 15
        self.class_name = 'TestReluOpShape'
        self.cls = TestReluOp
        self.inputs = [{'shape': [10]}, {'shape': [8, 5]}, {'shape': [10, 3, 5]}, {'shape': [80, 40, 5, 7]}, {'shape': [80, 1, 5, 7]}, {'shape': [80, 3, 1024, 7]}, {'shape': [10, 5, 1024, 2048]}, {'shape': [1]}, {'shape': [512]}, {'shape': [1024]}, {'shape': [2048]}, {'shape': [1, 1, 1, 1]}]
        self.dtypes = [{'dtype': 'float32'}]
        self.attrs = []

class TestReluOpDtype(TestCaseHelper):

    def init_attrs(self):
        if False:
            i = 10
            return i + 15
        self.class_name = 'TestReluOpDtype'
        self.cls = TestReluOp
        self.inputs = [{'shape': [1]}, {'shape': [5]}, {'shape': [80, 40, 5, 7]}]
        self.dtypes = [{'dtype': 'float16'}, {'dtype': 'float32'}, {'dtype': 'float64'}]
        self.attrs = []
if __name__ == '__main__':
    TestReluOpShape().run()
    TestReluOpDtype().run()