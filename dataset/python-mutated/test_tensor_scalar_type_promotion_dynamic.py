import unittest
import numpy as np
import paddle

class TestTensorScalarTypePromotionDynamic(unittest.TestCase):

    def check_operation(self, a, b, c, op):
        if False:
            while True:
                i = 10
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
        self.assertEqual(c_rlt.dtype, c.dtype)
        np.testing.assert_array_equal(c_rlt.numpy(), c.numpy())

    def test_tensor_add_scalar(self):
        if False:
            return 10
        a = paddle.ones([2, 2, 2], dtype='int64')
        b = 1
        c = paddle.full([2, 2, 2], 2, dtype='int64')
        self.check_operation(a, b, c, '+')
        a = paddle.ones([2, 2, 2], dtype='float32')
        b = 1
        c = paddle.full([2, 2, 2], 2, dtype='float32')
        self.check_operation(a, b, c, '+')
        a = paddle.ones([2, 2, 2], dtype='int64')
        b = 1.0
        c = paddle.full([2, 2, 2], 2, dtype='float32')
        self.check_operation(a, b, c, '+')
        a = paddle.ones([2, 2, 2], dtype='int64')
        b = 1.5
        c = paddle.full([2, 2, 2], 2.5, dtype='float32')
        self.check_operation(a, b, c, '+')
        a = paddle.ones([2, 2, 2], dtype='float32')
        b = 1.5
        c = paddle.full([2, 2, 2], 2.5, dtype='float32')
        self.check_operation(a, b, c, '+')

    def test_tensor_sub_scalar(self):
        if False:
            for i in range(10):
                print('nop')
        a = paddle.ones([2, 2, 2], dtype='int64')
        b = 1
        c = paddle.zeros([2, 2, 2], dtype='int64')
        self.check_operation(a, b, c, '-')
        a = paddle.ones([2, 2, 2], dtype='float32')
        b = 1
        c = paddle.zeros([2, 2, 2], dtype='float32')
        self.check_operation(a, b, c, '-')
        a = paddle.ones([2, 2, 2], dtype='int64')
        b = 1.0
        c = paddle.zeros([2, 2, 2], dtype='float32')
        self.check_operation(a, b, c, '-')
        a = paddle.full([2, 2, 2], 2, dtype='int64')
        b = 1.5
        c = paddle.full([2, 2, 2], 0.5, dtype='float32')
        self.check_operation(a, b, c, '-')
        a = paddle.full([2, 2, 2], 2, dtype='float32')
        b = 1.5
        c = paddle.full([2, 2, 2], 0.5, dtype='float32')
        self.check_operation(a, b, c, '-')

    def test_scalar_sub_tensor(self):
        if False:
            print('Hello World!')
        a = 1
        b = paddle.ones([2, 2, 2], dtype='int64')
        c = paddle.zeros([2, 2, 2], dtype='int64')
        self.check_operation(a, b, c, '-')
        a = 1
        b = paddle.ones([2, 2, 2], dtype='float32')
        c = paddle.zeros([2, 2, 2], dtype='float32')
        self.check_operation(a, b, c, '-')
        a = 1.0
        b = paddle.ones([2, 2, 2], dtype='int64')
        c = paddle.zeros([2, 2, 2], dtype='float32')
        self.check_operation(a, b, c, '-')
        a = 1.5
        b = paddle.full([2, 2, 2], 2, dtype='int64')
        c = paddle.full([2, 2, 2], -0.5, dtype='float32')
        self.check_operation(a, b, c, '-')
        a = 1.5
        b = paddle.full([2, 2, 2], 2, dtype='float32')
        c = paddle.full([2, 2, 2], -0.5, dtype='float32')
        self.check_operation(a, b, c, '-')

    def test_tensor_mul_tensor(self):
        if False:
            i = 10
            return i + 15
        a = paddle.ones([2, 2, 2], dtype='int64')
        b = 1
        c = paddle.ones([2, 2, 2], dtype='int64')
        self.check_operation(a, b, c, '*')
        a = paddle.ones([2, 2, 2], dtype='float32')
        b = 1
        c = paddle.ones([2, 2, 2], dtype='float32')
        self.check_operation(a, b, c, '*')
        a = paddle.ones([2, 2, 2], dtype='int64')
        b = 1.0
        c = paddle.ones([2, 2, 2], dtype='float32')
        self.check_operation(a, b, c, '*')
        a = paddle.ones([2, 2, 2], dtype='int64')
        b = 1.5
        c = paddle.full([2, 2, 2], 1.5, dtype='float32')
        self.check_operation(a, b, c, '*')
        a = paddle.ones([2, 2, 2], dtype='float32')
        b = 1.5
        c = paddle.full([2, 2, 2], 1.5, dtype='float32')
        self.check_operation(a, b, c, '*')

    def test_tensor_div_scalar(self):
        if False:
            for i in range(10):
                print('nop')
        a = paddle.ones([2, 2, 2], dtype='int64')
        b = 2
        c = paddle.full([2, 2, 2], 0.5, dtype='float32')
        self.check_operation(a, b, c, '/')
        a = paddle.ones([2, 2, 2], dtype='float32')
        b = 2
        c = paddle.full([2, 2, 2], 0.5, dtype='float32')
        self.check_operation(a, b, c, '/')
        a = paddle.ones([2, 2, 2], dtype='int64')
        b = 2.0
        c = paddle.full([2, 2, 2], 0.5, dtype='float32')
        self.check_operation(a, b, c, '/')
        a = paddle.ones([2, 2, 2], dtype='int64')
        b = 0.5
        c = paddle.full([2, 2, 2], 2, dtype='float32')
        self.check_operation(a, b, c, '/')
        a = paddle.ones([2, 2, 2], dtype='float32')
        b = 0.5
        c = paddle.full([2, 2, 2], 2, dtype='float32')
        self.check_operation(a, b, c, '/')

    def test_scalar_div_tensor(self):
        if False:
            return 10
        a = 1
        b = paddle.full([2, 2, 2], 2, dtype='int64')
        c = paddle.full([2, 2, 2], 0.5, dtype='float32')
        self.check_operation(a, b, c, '/')
        a = 1
        b = paddle.full([2, 2, 2], 0.5, dtype='float32')
        c = paddle.full([2, 2, 2], 2, dtype='float32')
        self.check_operation(a, b, c, '/')
        a = 1.0
        b = paddle.full([2, 2, 2], 2, dtype='int64')
        c = paddle.full([2, 2, 2], 0.5, dtype='float32')
        self.check_operation(a, b, c, '/')
        a = 1.0
        b = paddle.full([2, 2, 2], 0.5, dtype='float32')
        c = paddle.full([2, 2, 2], 2, dtype='float32')
        self.check_operation(a, b, c, '/')

    def test_tensor_pow_scalar(self):
        if False:
            return 10
        a = paddle.full([2, 2, 2], 2, dtype='int64')
        b = 3
        c = paddle.full([2, 2, 2], 8, dtype='int64')
        self.check_operation(a, b, c, '**')
        a = paddle.full([2, 2, 2], 2, dtype='int64')
        b = 3.0
        c = paddle.full([2, 2, 2], 8, dtype='float32')
        self.check_operation(a, b, c, '**')
        a = paddle.full([2, 2, 2], 2, dtype='float32')
        b = 3
        c = paddle.full([2, 2, 2], 8, dtype='float32')
        self.check_operation(a, b, c, '**')
        a = paddle.full([2, 2, 2], 2, dtype='float32')
        b = 3.0
        c = paddle.full([2, 2, 2], 8, dtype='float32')
        self.check_operation(a, b, c, '**')

    def test_scalar_pow_tensor(self):
        if False:
            print('Hello World!')
        a = 3
        b = paddle.full([2, 2, 2], 2, dtype='int64')
        c = paddle.full([2, 2, 2], 9, dtype='int64')
        self.check_operation(a, b, c, '**')
        a = 3.0
        b = paddle.full([2, 2, 2], 2, dtype='int64')
        c = paddle.full([2, 2, 2], 9, dtype='float32')
        self.check_operation(a, b, c, '**')
        a = 3
        b = paddle.full([2, 2, 2], 2, dtype='float32')
        c = paddle.full([2, 2, 2], 9, dtype='float32')
        self.check_operation(a, b, c, '**')
        a = 3.0
        b = paddle.full([2, 2, 2], 2, dtype='float32')
        c = paddle.full([2, 2, 2], 9, dtype='float32')
        self.check_operation(a, b, c, '**')

    def test_tensor_floordiv_scalar(self):
        if False:
            for i in range(10):
                print('nop')
        a = paddle.full([2, 2, 2], 3, dtype='int64')
        b = 2
        c = paddle.full([2, 2, 2], 1, dtype='int64')
        self.check_operation(a, b, c, '//')

    def test_tensor_mod_scalar(self):
        if False:
            print('Hello World!')
        a = paddle.full([2, 2, 2], 3, dtype='int64')
        b = 2
        c = paddle.full([2, 2, 2], 1, dtype='int64')
        self.check_operation(a, b, c, '%')
        a = paddle.full([2, 2, 2], 3, dtype='int64')
        b = 2.0
        c = paddle.full([2, 2, 2], 1, dtype='float32')
        self.check_operation(a, b, c, '%')
        a = paddle.full([2, 2, 2], 3, dtype='float32')
        b = 2
        c = paddle.full([2, 2, 2], 1, dtype='float32')
        self.check_operation(a, b, c, '%')
        a = paddle.full([2, 2, 2], 3, dtype='float32')
        b = 2.0
        c = paddle.full([2, 2, 2], 1, dtype='float32')
        self.check_operation(a, b, c, '%')
if __name__ == '__main__':
    unittest.main()