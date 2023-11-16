import unittest
import paddle
from paddle import base
from paddle.static.nn.control_flow import Assert

class TestAssertOp(unittest.TestCase):

    def run_network(self, net_func):
        if False:
            for i in range(10):
                print('nop')
        main_program = base.Program()
        startup_program = base.Program()
        with base.program_guard(main_program, startup_program):
            net_func()
        exe = base.Executor()
        exe.run(main_program)

    def test_assert_true(self):
        if False:
            for i in range(10):
                print('nop')

        def net_func():
            if False:
                return 10
            condition = paddle.tensor.fill_constant(shape=[1], dtype='bool', value=True)
            Assert(condition, [])
        self.run_network(net_func)

    def test_assert_false(self):
        if False:
            print('Hello World!')

        def net_func():
            if False:
                for i in range(10):
                    print('nop')
            condition = paddle.tensor.fill_constant(shape=[1], dtype='bool', value=False)
            Assert(condition)
        with self.assertRaises(ValueError):
            self.run_network(net_func)

    def test_assert_cond_numel_error(self):
        if False:
            i = 10
            return i + 15

        def net_func():
            if False:
                i = 10
                return i + 15
            condition = paddle.tensor.fill_constant(shape=[1, 2], dtype='bool', value=True)
            Assert(condition, [])
        with self.assertRaises(ValueError):
            self.run_network(net_func)

    def test_assert_print_data(self):
        if False:
            i = 10
            return i + 15

        def net_func():
            if False:
                while True:
                    i = 10
            zero = paddle.tensor.fill_constant(shape=[1], dtype='int64', value=0)
            one = paddle.tensor.fill_constant(shape=[1], dtype='int64', value=1)
            condition = paddle.less_than(one, zero)
            Assert(condition, [zero, one])
        print('test_assert_print_data')
        with self.assertRaises(ValueError):
            self.run_network(net_func)

    def test_assert_summary(self):
        if False:
            print('Hello World!')

        def net_func():
            if False:
                i = 10
                return i + 15
            x = paddle.tensor.fill_constant(shape=[10], dtype='float32', value=2.0)
            condition = paddle.max(x) < 1.0
            Assert(condition, (x,), 5)
        print('test_assert_summary')
        with self.assertRaises(ValueError):
            self.run_network(net_func)

    def test_assert_summary_greater_than_size(self):
        if False:
            for i in range(10):
                print('nop')

        def net_func():
            if False:
                return 10
            x = paddle.tensor.fill_constant(shape=[2, 3], dtype='float32', value=2.0)
            condition = paddle.max(x) < 1.0
            Assert(condition, [x], 10, name='test')
        print('test_assert_summary_greater_than_size')
        with self.assertRaises(ValueError):
            self.run_network(net_func)
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()