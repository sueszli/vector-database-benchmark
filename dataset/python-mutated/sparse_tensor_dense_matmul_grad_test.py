"""Tests for the gradient of `tf.sparse.sparse_dense_matmul()`."""
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import sparse_ops
import tensorflow.python.ops.sparse_grad
from tensorflow.python.platform import test

class SparseTensorDenseMatMulGradientTest(test.TestCase):

    def _sparsify(self, x, indices_dtype=np.int64):
        if False:
            i = 10
            return i + 15
        x[x < 0.5] = 0
        non_zero = np.where(x)
        x_indices = np.vstack(non_zero).astype(indices_dtype).T
        x_values = x[non_zero]
        x_shape = x.shape
        return (sparse_tensor.SparseTensor(indices=x_indices, values=x_values, dense_shape=x_shape), len(x_values))

    def _randomTensor(self, size, values_dtype, adjoint=False, sparse=False, indices_dtype=np.int64):
        if False:
            print('Hello World!')
        (n, m) = size
        x = np.random.randn(n, m).astype(values_dtype)
        if values_dtype in (np.complex64, np.complex128):
            x.imag = np.random.randn(n, m)
        if adjoint:
            x = x.transpose().conj()
        if sparse:
            return self._sparsify(x, indices_dtype=indices_dtype)
        else:
            return constant_op.constant(x, dtype=values_dtype)

    def _testGradients(self, adjoint_a, adjoint_b, name, values_dtype, indices_dtype):
        if False:
            for i in range(10):
                print('nop')
        (n, k, m) = np.random.randint(1, 10, size=3)
        (sp_t, nnz) = self._randomTensor([n, k], values_dtype, adjoint=adjoint_a, sparse=True, indices_dtype=indices_dtype)
        dense_t = self._randomTensor([k, m], values_dtype, adjoint=adjoint_b)
        matmul = sparse_ops.sparse_tensor_dense_matmul(sp_t, dense_t, adjoint_a=adjoint_a, adjoint_b=adjoint_b, name=name)
        with self.cached_session():
            dense_t_shape = [m, k] if adjoint_b else [k, m]
            sp_t_val_shape = [nnz]
            delta = 1 / 16.0 if values_dtype == np.float16 else 0.001
            tolerance = delta / 2.0 if values_dtype == np.float16 else 0.001
            err = gradient_checker.compute_gradient_error([dense_t, sp_t.values], [dense_t_shape, sp_t_val_shape], matmul, [n, m], delta=delta)
            print('%s gradient err = %s' % (name, err))
            self.assertLess(err, tolerance)

    def _testGradientsType(self, values_dtype, indices_dtype):
        if False:
            i = 10
            return i + 15
        for adjoint_a in [True, False]:
            for adjoint_b in [True, False]:
                name = 'sparse_tensor_dense_matmul_%s_%s_%s_%s' % (adjoint_a, adjoint_b, values_dtype.__name__, indices_dtype.__name__)
                self._testGradients(adjoint_a, adjoint_b, name, values_dtype, indices_dtype)

    @test_util.run_deprecated_v1
    def testGradients(self):
        if False:
            while True:
                i = 10
        np.random.seed(5)
        self._testGradientsType(np.float16, np.int64)
        self._testGradientsType(np.float32, np.int64)
        self._testGradientsType(np.float64, np.int64)
        self._testGradientsType(np.complex64, np.int64)
        self._testGradientsType(np.complex128, np.int64)
        self._testGradientsType(np.float32, np.int32)
        self._testGradientsType(np.complex64, np.int32)
if __name__ == '__main__':
    test.main()