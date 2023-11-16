from cinn.common import is_compiled_with_cuda
from cinn.frontend import NetBuilder
from op_test import OpTest, OpTestTool
from op_test_helper import TestCaseHelper
import paddle

@OpTestTool.skip_if(not is_compiled_with_cuda(), 'x86 test will be skipped due to timeout.')
class TestAsinOp(OpTest):

    def setUp(self):
        if False:
            print('Hello World!')
        print(f'\nRunning {self.__class__.__name__}: {self.case}')
        self.prepare_inputs()

    def prepare_inputs(self):
        if False:
            while True:
                i = 10
        self.x_np = self.random(shape=self.case['x_shape'], dtype=self.case['x_dtype'], low=-1.0, high=1.0)

    def build_paddle_program(self, target):
        if False:
            while True:
                i = 10
        x = paddle.to_tensor(self.x_np, stop_gradient=True)
        out = paddle.asin(x)
        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        if False:
            print('Hello World!')
        builder = NetBuilder('unary_elementwise_test')
        x = builder.create_input(self.nptype2cinntype(self.case['x_dtype']), self.case['x_shape'], 'x')
        out = builder.asin(x)
        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x], [self.x_np], [out])
        self.cinn_outputs = [res[0]]

    def test_check_results(self):
        if False:
            return 10
        self.check_outputs_and_grads()

class TestAsinOpShape(TestCaseHelper):

    def init_attrs(self):
        if False:
            i = 10
            return i + 15
        self.class_name = 'TestAsinOpShape'
        self.cls = TestAsinOp
        self.inputs = [{'x_shape': [1]}, {'x_shape': [1024]}, {'x_shape': [1, 2048]}, {'x_shape': [1, 1, 1]}, {'x_shape': [32, 64]}, {'x_shape': [16, 8, 4, 2]}, {'x_shape': [16, 8, 4, 2, 1]}]
        self.dtypes = [{'x_dtype': 'float32'}]
        self.attrs = []

class TestAsinOpDtype(TestCaseHelper):

    def init_attrs(self):
        if False:
            print('Hello World!')
        self.class_name = 'TestAsinOpDtype'
        self.cls = TestAsinOp
        self.inputs = [{'x_shape': [32, 64]}]
        self.dtypes = [{'x_dtype': 'float16', 'max_relative_error': 0.001}, {'x_dtype': 'float32'}, {'x_dtype': 'float64'}]
        self.attrs = []
if __name__ == '__main__':
    TestAsinOpShape().run()
    TestAsinOpDtype().run()