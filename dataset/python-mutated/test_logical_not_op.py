from cinn.common import is_compiled_with_cuda
from cinn.frontend import NetBuilder
from op_test import OpTest, OpTestTool
from op_test_helper import TestCaseHelper
import paddle

@OpTestTool.skip_if(not is_compiled_with_cuda(), 'x86 test will be skipped due to timeout.')
class TestLogicalNotOp(OpTest):

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
        self.x_np = self.random(shape=self.case['x_shape'], dtype=self.case['x_dtype'], low=-10, high=100)

    def build_paddle_program(self, target):
        if False:
            return 10
        x = paddle.to_tensor(self.x_np, stop_gradient=False)
        out = paddle.logical_not(x)
        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        if False:
            while True:
                i = 10
        builder = NetBuilder('logical_not')
        x = builder.create_input(self.nptype2cinntype(self.case['x_dtype']), self.case['x_shape'], 'x')
        out = builder.logical_not(x)
        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x], [self.x_np], [out])
        self.cinn_outputs = res

    def test_check_results(self):
        if False:
            print('Hello World!')
        self.check_outputs_and_grads(all_equal=True)

class TestLogicalNotCase1(TestCaseHelper):

    def init_attrs(self):
        if False:
            print('Hello World!')
        self.class_name = 'TestLogicalNotCase1'
        self.cls = TestLogicalNotOp
        self.inputs = [{'x_shape': [512, 256]}]
        self.dtypes = [{'x_dtype': 'bool'}, {'x_dtype': 'int8'}, {'x_dtype': 'int16'}, {'x_dtype': 'int32'}, {'x_dtype': 'int64'}, {'x_dtype': 'float32'}, {'x_dtype': 'float64'}]
        self.attrs = []

class TestLogicalNotCase2(TestCaseHelper):

    def init_attrs(self):
        if False:
            return 10
        self.class_name = 'TestLogicalNotCase2'
        self.cls = TestLogicalNotOp
        self.inputs = [{'x_shape': [1]}, {'x_shape': [1024]}, {'x_shape': [512, 256]}, {'x_shape': [128, 64, 32]}, {'x_shape': [128, 2048, 32]}, {'x_shape': [16, 8, 4, 2]}, {'x_shape': [1, 1, 1, 1]}, {'x_shape': [16, 8, 4, 2, 1]}]
        self.dtypes = [{'x_dtype': 'bool'}]
        self.attrs = []

class TestLogicalNotCaseWithBroadcast1(TestCaseHelper):

    def init_attrs(self):
        if False:
            print('Hello World!')
        self.class_name = 'TestLogicalNotCaseWithBroadcast1'
        self.cls = TestLogicalNotOp
        self.inputs = [{'x_shape': [56]}]
        self.dtypes = [{'x_dtype': 'bool'}, {'x_dtype': 'int8'}, {'x_dtype': 'int16'}, {'x_dtype': 'int32'}, {'x_dtype': 'int64'}, {'x_dtype': 'float32'}, {'x_dtype': 'float64'}]
        self.attrs = []

class TestLogicalNotCaseWithBroadcast2(TestCaseHelper):

    def init_attrs(self):
        if False:
            for i in range(10):
                print('nop')
        self.class_name = 'TestLogicalNotCaseWithBroadcast2'
        self.cls = TestLogicalNotOp
        self.inputs = [{'x_shape': [56]}, {'x_shape': [1024]}, {'x_shape': [512, 256]}, {'x_shape': [128, 64, 32]}, {'x_shape': [16, 1, 1, 2]}, {'x_shape': [16, 1, 1, 2, 1]}]
        self.dtypes = [{'x_dtype': 'bool'}]
        self.attrs = []
if __name__ == '__main__':
    TestLogicalNotCase1().run()
    TestLogicalNotCase2().run()
    TestLogicalNotCaseWithBroadcast1().run()
    TestLogicalNotCaseWithBroadcast2().run()