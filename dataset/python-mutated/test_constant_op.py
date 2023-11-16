from cinn.common import is_compiled_with_cuda
from cinn.frontend import NetBuilder
from op_test import OpTest, OpTestTool
from op_test_helper import TestCaseHelper
import paddle

@OpTestTool.skip_if(not is_compiled_with_cuda(), 'x86 test will be skipped due to timeout.')
class TestConstantOp(OpTest):

    def setUp(self):
        if False:
            return 10
        print(f'\nRunning {self.__class__.__name__}: {self.case}')
        self.inputs = {}
        self.prepare_inputs()

    def prepare_inputs(self):
        if False:
            i = 10
            return i + 15
        self.name = 'x'
        dtype = self.case['dtype']
        if 'constant_value' in self.case:
            if 'bool' in dtype:
                self.value = bool(self.case['constant_value'])
            elif 'int' in dtype:
                self.value = int(self.case['constant_value'])
            elif 'float' in dtype:
                self.value = float(self.case['constant_value'])
        else:
            self.value = self.random(self.case['shape'], dtype).tolist()
        self.dtype = dtype

    def build_paddle_program(self, target):
        if False:
            for i in range(10):
                print('nop')
        x = paddle.to_tensor(self.value, dtype=self.dtype)
        self.paddle_outputs = [x]

    def build_cinn_program(self, target):
        if False:
            return 10
        builder = NetBuilder('constant')
        x = builder.constant(self.value, self.name, self.dtype)
        prog = builder.build()
        res = self.get_cinn_output(prog, target, [], [], [x])
        self.cinn_outputs = res

    def test_check_results(self):
        if False:
            return 10
        self.check_outputs_and_grads(all_equal=True)

class TestConstantOpShape(TestCaseHelper):

    def init_attrs(self):
        if False:
            i = 10
            return i + 15
        self.class_name = 'TestConstantOpShape'
        self.cls = TestConstantOp
        self.inputs = [{'constant_value': 10}, {'constant_value': -5}, {'shape': [10]}, {'shape': [8, 5]}, {'shape': [10, 3, 5]}, {'shape': [1, 2, 4, 8]}, {'shape': [1]}, {'shape': [512]}, {'shape': [1024]}, {'shape': [1, 1, 1, 1]}]
        self.dtypes = [{'dtype': 'float32'}]
        self.attrs = []

class TestConstantOpDtype(TestCaseHelper):

    def init_attrs(self):
        if False:
            print('Hello World!')
        self.class_name = 'TestConstantOpDtype'
        self.cls = TestConstantOp
        self.inputs = [{'constant_value': 1}, {'shape': [10]}, {'shape': [8, 5]}, {'shape': [10, 3, 5]}]
        self.dtypes = [{'dtype': 'float16'}, {'dtype': 'float32'}, {'dtype': 'float64'}, {'dtype': 'bool'}, {'dtype': 'uint8'}, {'dtype': 'int8'}, {'dtype': 'int32'}, {'dtype': 'int64'}]
        self.attrs = []
if __name__ == '__main__':
    TestConstantOpShape().run()
    TestConstantOpDtype().run()