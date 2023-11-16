import unittest
import paddle
from paddle import base

class TestProgramToReadableCode(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.program = base.Program()
        self.block = self.program.current_block()
        self.var = self.block.create_var(name='X', shape=[-1, 23, 48], dtype='float32')
        self.param = self.block.create_parameter(name='W', shape=[23, 48], dtype='float32', trainable=True)
        self.op = self.block.append_op(type='abs', inputs={'X': [self.var]}, outputs={'Out': [self.var]})
        self.append_cond_op(self.program)

    def append_cond_op(self, program):
        if False:
            print('Hello World!')

        def true_func():
            if False:
                print('Hello World!')
            return paddle.tensor.fill_constant(shape=[2, 3], dtype='int32', value=2)

        def false_func():
            if False:
                return 10
            return paddle.tensor.fill_constant(shape=[3, 2], dtype='int32', value=-1)
        with base.program_guard(program):
            x = paddle.tensor.fill_constant(shape=[1], dtype='float32', value=0.1)
            y = paddle.tensor.fill_constant(shape=[1], dtype='float32', value=0.23)
            pred = paddle.less_than(y, x)
            out = paddle.static.nn.cond(pred, true_func, false_func)

    def test_program_code(self):
        if False:
            while True:
                i = 10
        self.var._to_readable_code()
        self.param._to_readable_code()
        self.op._to_readable_code()
        self.block._to_readable_code()
        self.program._to_readable_code()

    def test_program_print(self):
        if False:
            return 10
        print(self.var)
        print(self.param)
        print(self.op)
        print(self.block)
        print(self.program)
if __name__ == '__main__':
    unittest.main()