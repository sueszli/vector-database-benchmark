"""CSR sparse matrix tests."""
import itertools
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg.sparse import sparse_csr_matrix_grad
from tensorflow.python.ops.linalg.sparse import sparse_csr_matrix_ops
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging

def dense_and_sparse_from_vals(vals, datatype):
    if False:
        while True:
            i = 10
    locs = array_ops.where(math_ops.abs(vals) > 0)
    dense_t = ops.convert_to_tensor(vals, dtype=datatype)
    return (dense_t, sparse_csr_matrix_ops.dense_to_csr_sparse_matrix(dense_t, locs))

def _add_test(test, op_name, testcase_name, fn):
    if False:
        i = 10
        return i + 15
    if fn is None:
        return
    test_name = '_'.join(['test', op_name, testcase_name])
    if hasattr(test, test_name):
        raise RuntimeError('Test %s defined more than once' % test_name)
    setattr(test, test_name, fn)

class CSRSparseMatrixGradTest(test.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        super(CSRSparseMatrixGradTest, cls).setUpClass()
        cls._gpu_available = test_util.is_gpu_available()

    @test_util.run_deprecated_v1
    def _testLargeBatchSparseMatrixSparseMatMulGrad(self, datatype, transpose_a, transpose_b, adjoint_a, adjoint_b):
        if False:
            print('Hello World!')
        if not self._gpu_available:
            return
        sparsify = lambda m: m * (m > 0)
        a_mats_val = sparsify(np.random.randn(3, 5, 11) + 1j * np.random.randn(3, 5, 11)).astype(datatype)
        if transpose_a or adjoint_a:
            a_mats_val = np.transpose(a_mats_val, (0, 2, 1))
        if adjoint_a:
            a_mats_val = np.conj(a_mats_val)
        b_mats_val = sparsify(np.random.randn(3, 11, 13) + 1j * np.random.randn(3, 11, 13)).astype(datatype)
        if transpose_b or adjoint_b:
            b_mats_val = np.transpose(b_mats_val, (0, 2, 1))
        if adjoint_b:
            b_mats_val = np.conj(b_mats_val)
        with self.test_session():
            (a_mats, a_sm) = dense_and_sparse_from_vals(a_mats_val, datatype)
            (b_mats, b_sm) = dense_and_sparse_from_vals(b_mats_val, datatype)
            c_sm = sparse_csr_matrix_ops.sparse_matrix_sparse_mat_mul(a_sm, b_sm, transpose_a=transpose_a, transpose_b=transpose_b, adjoint_a=adjoint_a, adjoint_b=adjoint_b, type=datatype)
            c_dense = sparse_csr_matrix_ops.csr_sparse_matrix_to_dense(c_sm, type=datatype)
            for (ten, val, nn) in [[a_mats, a_mats_val, 'a'], [b_mats, b_mats_val, 'b']]:
                tf_logging.info('Testing gradients for %s' % nn)
                (theoretical, numerical) = gradient_checker.compute_gradient(ten, ten.get_shape().as_list(), c_dense, c_dense.get_shape().as_list(), x_init_value=val, delta=0.001)
                self.assertAllClose(theoretical, numerical, atol=0.001, rtol=0.001)
for dtype in (np.float32, np.complex64):
    for (t_a, t_b, adj_a, adj_b) in itertools.product(*([False, True],) * 4):

        def create_sparse_mat_mul_test_fn(dtype_, t_a_, t_b_, adj_a_, adj_b_):
            if False:
                while True:
                    i = 10
            if t_a_ and adj_a_ or (t_b_ and adj_b_):
                return
            if dtype_ == np.float32 and (adj_a_ or adj_b_):
                return

            def test_fn(self):
                if False:
                    while True:
                        i = 10
                self._testLargeBatchSparseMatrixSparseMatMulGrad(dtype_, t_a_, t_b_, adj_a_, adj_b_)
            return test_fn
        name = '_testLargeBatchSparseMatrixSparseMatMulGrad_dtype_%s_t_a_%s_t_b_%s_adj_a_%s_adj_b_%s' % (dtype.__name__, t_a, t_b, adj_a, adj_b)
        _add_test(CSRSparseMatrixGradTest, 'CSRSparseMatrixSparseGradTest', name, create_sparse_mat_mul_test_fn(dtype, t_a, t_b, adj_a, adj_b))
if __name__ == '__main__':
    test.main()