import unittest
import numpy as np
import paddle
from paddle.static import Program, program_guard

class TestTensorScalarTypePromotionStatic(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        paddle.enable_static()

    def check_operation(self, a, b, c, op):
        if False:
            return 10
        exe = paddle.static.Executor()
        if op == '+':
            c_rlt = a + b
        elif op == '-':
            c_rlt = a - b
        elif op == '*':
            c_rlt = a * b
        elif op == '/':
            c_rlt = a / b
        elif op == '**':
            c_rlt = a ** b
        elif op == '//':
            c_rlt = a // b
        elif op == '%':
            c_rlt = a % b
        else:
            raise ValueError('Unsupported operation.')
        rlt = exe.run(fetch_list=[c_rlt.name, c.name])
        self.assertEqual(rlt[0].dtype, rlt[1].dtype)
        np.testing.assert_array_equal(rlt[0], rlt[1])

    def test_tensor_add_scalar(self):
        if False:
            while True:
                i = 10
        with program_guard(Program()):
            a = paddle.ones([2, 2, 2], dtype='int64')
            b = 1
            c = paddle.full([2, 2, 2], 2, dtype='int64')
            self.check_operation(a, b, c, '+')
        with program_guard(Program()):
            a = paddle.ones([2, 2, 2], dtype='float32')
            b = 1
            c = paddle.full([2, 2, 2], 2, dtype='float32')
            self.check_operation(a, b, c, '+')
        with program_guard(Program()):
            a = paddle.ones([2, 2, 2], dtype='int64')
            b = 1.0
            c = paddle.full([2, 2, 2], 2, dtype='float32')
            self.check_operation(a, b, c, '+')
        with program_guard(Program()):
            a = paddle.ones([2, 2, 2], dtype='int64')
            b = 1.5
            c = paddle.full([2, 2, 2], 2.5, dtype='float32')
            self.check_operation(a, b, c, '+')
        with program_guard(Program()):
            a = paddle.ones([2, 2, 2], dtype='float32')
            b = 1.5
            c = paddle.full([2, 2, 2], 2.5, dtype='float32')
            self.check_operation(a, b, c, '+')

    def test_tensor_sub_scalar(self):
        if False:
            while True:
                i = 10
        with program_guard(Program()):
            a = paddle.ones([2, 2, 2], dtype='int64')
            b = 1
            c = paddle.zeros([2, 2, 2], dtype='int64')
            self.check_operation(a, b, c, '-')
        with program_guard(Program()):
            a = paddle.ones([2, 2, 2], dtype='float32')
            b = 1
            c = paddle.zeros([2, 2, 2], dtype='float32')
            self.check_operation(a, b, c, '-')
        with program_guard(Program()):
            a = paddle.ones([2, 2, 2], dtype='int64')
            b = 1.0
            c = paddle.zeros([2, 2, 2], dtype='float32')
            self.check_operation(a, b, c, '-')
        with program_guard(Program()):
            a = paddle.full([2, 2, 2], 2, dtype='int64')
            b = 1.5
            c = paddle.full([2, 2, 2], 0.5, dtype='float32')
            self.check_operation(a, b, c, '-')
        with program_guard(Program()):
            a = paddle.full([2, 2, 2], 2, dtype='float32')
            b = 1.5
            c = paddle.full([2, 2, 2], 0.5, dtype='float32')
            self.check_operation(a, b, c, '-')

    def test_scalar_sub_tensor(self):
        if False:
            return 10
        with program_guard(Program()):
            a = 1
            b = paddle.ones([2, 2, 2], dtype='int64')
            c = paddle.zeros([2, 2, 2], dtype='int64')
            self.check_operation(a, b, c, '-')
        with program_guard(Program()):
            a = 1
            b = paddle.ones([2, 2, 2], dtype='float32')
            c = paddle.zeros([2, 2, 2], dtype='float32')
            self.check_operation(a, b, c, '-')
        with program_guard(Program()):
            a = 1.0
            b = paddle.ones([2, 2, 2], dtype='int64')
            c = paddle.zeros([2, 2, 2], dtype='float32')
            self.check_operation(a, b, c, '-')
        with program_guard(Program()):
            a = 1.5
            b = paddle.full([2, 2, 2], 2, dtype='int64')
            c = paddle.full([2, 2, 2], -0.5, dtype='float32')
            self.check_operation(a, b, c, '-')
        with program_guard(Program()):
            a = 1.5
            b = paddle.full([2, 2, 2], 2, dtype='float32')
            c = paddle.full([2, 2, 2], -0.5, dtype='float32')
            self.check_operation(a, b, c, '-')

    def test_tensor_mul_tensor(self):
        if False:
            print('Hello World!')
        with program_guard(Program()):
            a = paddle.ones([2, 2, 2], dtype='int64')
            b = 1
            c = paddle.ones([2, 2, 2], dtype='int64')
            self.check_operation(a, b, c, '*')
        with program_guard(Program()):
            a = paddle.ones([2, 2, 2], dtype='float32')
            b = 1
            c = paddle.ones([2, 2, 2], dtype='float32')
            self.check_operation(a, b, c, '*')
        with program_guard(Program()):
            a = paddle.ones([2, 2, 2], dtype='int64')
            b = 1.0
            c = paddle.ones([2, 2, 2], dtype='float32')
            self.check_operation(a, b, c, '*')
        with program_guard(Program()):
            a = paddle.ones([2, 2, 2], dtype='int64')
            b = 1.5
            c = paddle.full([2, 2, 2], 1.5, dtype='float32')
            self.check_operation(a, b, c, '*')
        with program_guard(Program()):
            a = paddle.ones([2, 2, 2], dtype='float32')
            b = 1.5
            c = paddle.full([2, 2, 2], 1.5, dtype='float32')
            self.check_operation(a, b, c, '*')

    def test_tensor_div_scalar(self):
        if False:
            return 10
        with program_guard(Program()):
            a = paddle.ones([2, 2, 2], dtype='int64')
            b = 2
            c = paddle.full([2, 2, 2], 0.5, dtype='float32')
            self.check_operation(a, b, c, '/')
        with program_guard(Program()):
            a = paddle.ones([2, 2, 2], dtype='float32')
            b = 2
            c = paddle.full([2, 2, 2], 0.5, dtype='float32')
            self.check_operation(a, b, c, '/')
        with program_guard(Program()):
            a = paddle.ones([2, 2, 2], dtype='int64')
            b = 2.0
            c = paddle.full([2, 2, 2], 0.5, dtype='float32')
            self.check_operation(a, b, c, '/')
        with program_guard(Program()):
            a = paddle.ones([2, 2, 2], dtype='int64')
            b = 0.5
            c = paddle.full([2, 2, 2], 2, dtype='float32')
            self.check_operation(a, b, c, '/')
        with program_guard(Program()):
            a = paddle.ones([2, 2, 2], dtype='float32')
            b = 0.5
            c = paddle.full([2, 2, 2], 2, dtype='float32')
            self.check_operation(a, b, c, '/')

    def test_scalar_div_tensor(self):
        if False:
            while True:
                i = 10
        with program_guard(Program()):
            a = 1
            b = paddle.full([2, 2, 2], 2, dtype='int64')
            c = paddle.full([2, 2, 2], 0.5, dtype='float32')
            self.check_operation(a, b, c, '/')
        with program_guard(Program()):
            a = 1
            b = paddle.full([2, 2, 2], 0.5, dtype='float32')
            c = paddle.full([2, 2, 2], 2, dtype='float32')
            self.check_operation(a, b, c, '/')
        with program_guard(Program()):
            a = 1.0
            b = paddle.full([2, 2, 2], 2, dtype='int64')
            c = paddle.full([2, 2, 2], 0.5, dtype='float32')
            self.check_operation(a, b, c, '/')
        with program_guard(Program()):
            a = 1.0
            b = paddle.full([2, 2, 2], 0.5, dtype='float32')
            c = paddle.full([2, 2, 2], 2, dtype='float32')
            self.check_operation(a, b, c, '/')

    def test_tensor_pow_scalar(self):
        if False:
            while True:
                i = 10
        with program_guard(Program()):
            a = paddle.full([2, 2, 2], 2, dtype='int64')
            b = 3
            c = paddle.full([2, 2, 2], 8, dtype='int64')
            self.check_operation(a, b, c, '**')
        with program_guard(Program()):
            a = paddle.full([2, 2, 2], 2, dtype='int64')
            b = 3.0
            c = paddle.full([2, 2, 2], 8, dtype='float32')
            self.check_operation(a, b, c, '**')
        with program_guard(Program()):
            a = paddle.full([2, 2, 2], 2, dtype='float32')
            b = 3
            c = paddle.full([2, 2, 2], 8, dtype='float32')
            self.check_operation(a, b, c, '**')
        with program_guard(Program()):
            a = paddle.full([2, 2, 2], 2, dtype='float32')
            b = 3.0
            c = paddle.full([2, 2, 2], 8, dtype='float32')
            self.check_operation(a, b, c, '**')

    def test_scalar_pow_tensor(self):
        if False:
            while True:
                i = 10
        with program_guard(Program()):
            a = 3
            b = paddle.full([2, 2, 2], 2, dtype='int64')
            c = paddle.full([2, 2, 2], 9, dtype='int64')
            self.check_operation(a, b, c, '**')
        with program_guard(Program()):
            a = 3.0
            b = paddle.full([2, 2, 2], 2, dtype='int64')
            c = paddle.full([2, 2, 2], 9, dtype='float32')
            self.check_operation(a, b, c, '**')
        with program_guard(Program()):
            a = 3
            b = paddle.full([2, 2, 2], 2, dtype='float32')
            c = paddle.full([2, 2, 2], 9, dtype='float32')
            self.check_operation(a, b, c, '**')
        with program_guard(Program()):
            a = 3.0
            b = paddle.full([2, 2, 2], 2, dtype='float32')
            c = paddle.full([2, 2, 2], 9, dtype='float32')
            self.check_operation(a, b, c, '**')

    def test_tensor_floordiv_scalar(self):
        if False:
            while True:
                i = 10
        with program_guard(Program()):
            a = paddle.full([2, 2, 2], 3, dtype='int64')
            b = 2
            c = paddle.full([2, 2, 2], 1, dtype='int64')
            self.check_operation(a, b, c, '//')

    def test_tensor_mod_scalar(self):
        if False:
            for i in range(10):
                print('nop')
        with program_guard(Program()):
            a = paddle.full([2, 2, 2], 3, dtype='int64')
            b = 2
            c = paddle.full([2, 2, 2], 1, dtype='int64')
            self.check_operation(a, b, c, '%')
        with program_guard(Program()):
            a = paddle.full([2, 2, 2], 3, dtype='int64')
            b = 2.0
            c = paddle.full([2, 2, 2], 1, dtype='float32')
            self.check_operation(a, b, c, '%')
        with program_guard(Program()):
            a = paddle.full([2, 2, 2], 3, dtype='float32')
            b = 2
            c = paddle.full([2, 2, 2], 1, dtype='float32')
            self.check_operation(a, b, c, '%')
        with program_guard(Program()):
            a = paddle.full([2, 2, 2], 3, dtype='float32')
            b = 2.0
            c = paddle.full([2, 2, 2], 1, dtype='float32')
            self.check_operation(a, b, c, '%')
if __name__ == '__main__':
    unittest.main()