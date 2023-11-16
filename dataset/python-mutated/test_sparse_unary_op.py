import unittest
import numpy as np
import paddle
from paddle.base.framework import convert_np_dtype_to_dtype_

class TestSparseUnary(unittest.TestCase):

    def to_sparse(self, x, format):
        if False:
            while True:
                i = 10
        if format == 'coo':
            return x.detach().to_sparse_coo(sparse_dim=x.ndim)
        elif format == 'csr':
            return x.detach().to_sparse_csr()

    def check_result(self, dense_func, sparse_func, format, *args):
        if False:
            return 10
        origin_x = paddle.rand([8, 16, 32], dtype='float32')
        mask = paddle.randint(0, 2, [8, 16, 32]).astype('float32')
        while paddle.sum(mask) == 0:
            mask = paddle.randint(0, 2, [8, 16, 32]).astype('float32')
        dense_x = origin_x * mask
        sp_x = self.to_sparse(dense_x, format)
        sp_x.stop_gradient = False
        if len(args) == 0:
            sp_out = sparse_func(sp_x)
        elif len(args) == 1:
            sp_out = sparse_func(sp_x, args[0])
        elif len(args) == 2:
            sp_out = sparse_func(sp_x, args[0], args[1])
        sp_out.backward()
        dense_x.stop_gradient = False
        if len(args) == 0:
            dense_out = dense_func(dense_x)
        elif len(args) == 1:
            dense_out = dense_func(dense_x, args[0])
        elif len(args) == 2:
            if dense_func == paddle.cast:
                dense_out = dense_func(dense_x, args[1])
                int_dtype = convert_np_dtype_to_dtype_(args[0])
                if sp_out.is_sparse_csr():
                    self.assertEqual(sp_out.crows().dtype, int_dtype)
                    self.assertEqual(sp_out.cols().dtype, int_dtype)
                elif sp_out.is_sparse_coo():
                    self.assertEqual(sp_out.indices().dtype, int_dtype)
            else:
                dense_out = dense_func(dense_x, args[0], args[1])
        dense_out.backward()
        np.testing.assert_allclose(sp_out.to_dense().numpy(), dense_out.numpy(), rtol=1e-05)
        if dense_func == paddle.sqrt:
            expect_grad = np.nan_to_num(dense_x.grad.numpy(), 0.0, 0.0, 0.0)
        else:
            expect_grad = (dense_x.grad * mask).numpy()
        np.testing.assert_allclose(sp_x.grad.to_dense().numpy(), expect_grad, rtol=1e-05)

    def compare_with_dense(self, dense_func, sparse_func):
        if False:
            i = 10
            return i + 15
        self.check_result(dense_func, sparse_func, 'coo')
        self.check_result(dense_func, sparse_func, 'csr')

    def compare_with_dense_one_attr(self, dense_func, sparse_func, attr1):
        if False:
            return 10
        self.check_result(dense_func, sparse_func, 'coo', attr1)
        self.check_result(dense_func, sparse_func, 'csr', attr1)

    def compare_with_dense_two_attr(self, dense_func, sparse_func, attr1, attr2):
        if False:
            while True:
                i = 10
        self.check_result(dense_func, sparse_func, 'coo', attr1, attr2)
        self.check_result(dense_func, sparse_func, 'csr', attr1, attr2)

    def test_sparse_sin(self):
        if False:
            return 10
        self.compare_with_dense(paddle.sin, paddle.sparse.sin)

    def test_sparse_tan(self):
        if False:
            i = 10
            return i + 15
        self.compare_with_dense(paddle.tan, paddle.sparse.tan)

    def test_sparse_asin(self):
        if False:
            while True:
                i = 10
        self.compare_with_dense(paddle.asin, paddle.sparse.asin)

    def test_sparse_atan(self):
        if False:
            for i in range(10):
                print('nop')
        self.compare_with_dense(paddle.atan, paddle.sparse.atan)

    def test_sparse_sinh(self):
        if False:
            print('Hello World!')
        self.compare_with_dense(paddle.sinh, paddle.sparse.sinh)

    def test_sparse_tanh(self):
        if False:
            print('Hello World!')
        self.compare_with_dense(paddle.tanh, paddle.sparse.tanh)

    def test_sparse_asinh(self):
        if False:
            for i in range(10):
                print('nop')
        self.compare_with_dense(paddle.asinh, paddle.sparse.asinh)

    def test_sparse_atanh(self):
        if False:
            return 10
        self.compare_with_dense(paddle.atanh, paddle.sparse.atanh)

    def test_sparse_sqrt(self):
        if False:
            return 10
        self.compare_with_dense(paddle.sqrt, paddle.sparse.sqrt)

    def test_sparse_square(self):
        if False:
            return 10
        self.compare_with_dense(paddle.square, paddle.sparse.square)

    def test_sparse_log1p(self):
        if False:
            for i in range(10):
                print('nop')
        self.compare_with_dense(paddle.log1p, paddle.sparse.log1p)

    def test_sparse_relu(self):
        if False:
            print('Hello World!')
        self.compare_with_dense(paddle.nn.ReLU(), paddle.sparse.nn.ReLU())

    def test_sparse_relu6(self):
        if False:
            for i in range(10):
                print('nop')
        self.compare_with_dense(paddle.nn.ReLU6(), paddle.sparse.nn.ReLU6())

    def test_sparse_leaky_relu(self):
        if False:
            print('Hello World!')
        self.compare_with_dense(paddle.nn.LeakyReLU(0.1), paddle.sparse.nn.LeakyReLU(0.1))

    def test_sparse_abs(self):
        if False:
            return 10
        self.compare_with_dense(paddle.abs, paddle.sparse.abs)

    def test_sparse_expm1(self):
        if False:
            for i in range(10):
                print('nop')
        self.compare_with_dense(paddle.expm1, paddle.sparse.expm1)

    def test_sparse_deg2rad(self):
        if False:
            while True:
                i = 10
        self.compare_with_dense(paddle.deg2rad, paddle.sparse.deg2rad)

    def test_sparse_rad2deg(self):
        if False:
            i = 10
            return i + 15
        self.compare_with_dense(paddle.rad2deg, paddle.sparse.rad2deg)

    def test_sparse_neg(self):
        if False:
            i = 10
            return i + 15
        self.compare_with_dense(paddle.neg, paddle.sparse.neg)

    def test_sparse_pow(self):
        if False:
            return 10
        self.compare_with_dense_one_attr(paddle.pow, paddle.sparse.pow, 3)

    def test_sparse_mul_scalar(self):
        if False:
            print('Hello World!')
        self.compare_with_dense_one_attr(paddle.Tensor.__mul__, paddle.sparse.multiply, 3)

    def test_sparse_div_scalar(self):
        if False:
            for i in range(10):
                print('nop')
        self.compare_with_dense_one_attr(paddle.Tensor.__div__, paddle.sparse.divide, 2)

    def test_sparse_cast(self):
        if False:
            print('Hello World!')
        self.compare_with_dense_two_attr(paddle.cast, paddle.sparse.cast, 'int32', 'float32')
        self.compare_with_dense_two_attr(paddle.cast, paddle.sparse.cast, 'int32', 'float64')
if __name__ == '__main__':
    unittest.main()