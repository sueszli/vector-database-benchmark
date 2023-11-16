from tensorflow.python.framework import config
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.ops.linalg import linalg as linalg_lib
from tensorflow.python.ops.linalg import linear_operator_test_util
from tensorflow.python.platform import test
linalg = linalg_lib

@test_util.run_all_in_graph_and_eager_modes
class LinearOperatorLowerTriangularTest(linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
    """Most tests done in the base class LinearOperatorDerivedClassTest."""

    @staticmethod
    def skip_these_tests():
        if False:
            i = 10
            return i + 15
        return ['cholesky']

    def operator_and_matrix(self, build_info, dtype, use_placeholder, ensure_self_adjoint_and_pd=False):
        if False:
            for i in range(10):
                print('nop')
        shape = list(build_info.shape)
        tril = linear_operator_test_util.random_tril_matrix(shape, dtype=dtype, force_well_conditioned=True, remove_upper=False)
        if ensure_self_adjoint_and_pd:
            tril = array_ops.matrix_diag_part(tril)
            tril = math_ops.abs(tril) + 0.1
            tril = array_ops.matrix_diag(tril)
        lin_op_tril = tril
        if use_placeholder:
            lin_op_tril = array_ops.placeholder_with_default(lin_op_tril, shape=None)
        operator = linalg.LinearOperatorLowerTriangular(lin_op_tril, is_self_adjoint=True if ensure_self_adjoint_and_pd else None, is_positive_definite=True if ensure_self_adjoint_and_pd else None)
        matrix = array_ops.matrix_band_part(tril, -1, 0)
        return (operator, matrix)

    def test_assert_non_singular(self):
        if False:
            print('Hello World!')
        with self.cached_session():
            tril = [[1.0, 0.0], [1.0, 0.0]]
            operator = linalg.LinearOperatorLowerTriangular(tril)
            with self.assertRaisesOpError('Singular operator'):
                operator.assert_non_singular().run()

    def test_is_x_flags(self):
        if False:
            for i in range(10):
                print('nop')
        tril = [[1.0, 0.0], [1.0, 1.0]]
        operator = linalg.LinearOperatorLowerTriangular(tril, is_positive_definite=True, is_non_singular=True, is_self_adjoint=False)
        self.assertTrue(operator.is_positive_definite)
        self.assertTrue(operator.is_non_singular)
        self.assertFalse(operator.is_self_adjoint)

    def test_tril_must_have_at_least_two_dims_or_raises(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaisesRegex(ValueError, 'at least 2 dimensions'):
            linalg.LinearOperatorLowerTriangular([1.0])

    def test_triangular_diag_matmul(self):
        if False:
            print('Hello World!')
        operator1 = linalg_lib.LinearOperatorLowerTriangular([[1.0, 0.0, 0.0], [2.0, 1.0, 0.0], [2.0, 3.0, 3.0]])
        operator2 = linalg_lib.LinearOperatorDiag([2.0, 2.0, 3.0])
        operator_matmul = operator1.matmul(operator2)
        self.assertTrue(isinstance(operator_matmul, linalg_lib.LinearOperatorLowerTriangular))
        self.assertAllClose(math_ops.matmul(operator1.to_dense(), operator2.to_dense()), self.evaluate(operator_matmul.to_dense()))
        operator_matmul = operator2.matmul(operator1)
        self.assertTrue(isinstance(operator_matmul, linalg_lib.LinearOperatorLowerTriangular))
        self.assertAllClose(math_ops.matmul(operator2.to_dense(), operator1.to_dense()), self.evaluate(operator_matmul.to_dense()))

    def test_tape_safe(self):
        if False:
            while True:
                i = 10
        tril = variables_module.Variable([[1.0, 0.0], [0.0, 1.0]])
        operator = linalg_lib.LinearOperatorLowerTriangular(tril, is_non_singular=True)
        self.check_tape_safe(operator)

    def test_convert_variables_to_tensors(self):
        if False:
            i = 10
            return i + 15
        tril = variables_module.Variable([[1.0, 0.0], [0.0, 1.0]])
        operator = linalg_lib.LinearOperatorLowerTriangular(tril, is_non_singular=True)
        with self.cached_session() as sess:
            sess.run([tril.initializer])
            self.check_convert_variables_to_tensors(operator)

    def test_llt_composition_with_pd_l(self):
        if False:
            print('Hello World!')
        l = linalg_lib.LinearOperatorLowerTriangular([[1.0, 0.0], [0.5, 0.2]], is_non_singular=True, is_positive_definite=True)
        self.assertIs(l, (l @ l.H).cholesky())

    def test_llt_composition_with_non_pd_l(self):
        if False:
            for i in range(10):
                print('nop')
        l = linalg_lib.LinearOperatorLowerTriangular([[-1.0, 0.0, 0.0], [0.5, 0.2, 0.0], [0.1, 0.1, 1.0]], is_non_singular=True)
        llt = l @ l.H
        chol = llt.cholesky()
        self.assertIsInstance(chol, linalg_lib.LinearOperatorLowerTriangular)
        self.assertGreater(self.evaluate(chol.diag_part()).min(), 0)
        self.assertAllClose(self.evaluate(llt.to_dense()), self.evaluate((chol @ chol.H).to_dense()))

    def test_llt_composition_with_non_pd_complex_l(self):
        if False:
            while True:
                i = 10
        i = math_ops.complex(0.0, 1.0)
        l = linalg_lib.LinearOperatorLowerTriangular([[-1.0 + i, 0.0, 0.0], [0.5, 0.2 - 2 * i, 0.0], [0.1, 0.1, 1.0]], is_non_singular=True)
        llt = l @ l.H
        chol = llt.cholesky()
        self.assertIsInstance(chol, linalg_lib.LinearOperatorLowerTriangular)
        self.assertGreater(self.evaluate(math_ops.real(chol.diag_part())).min(), 0)
        self.assertAllClose(self.evaluate(math_ops.imag(chol.diag_part())).min(), 0)
        self.assertAllClose(self.evaluate(llt.to_dense()), self.evaluate((chol @ chol.H).to_dense()))
if __name__ == '__main__':
    config.enable_tensor_float_32_execution(False)
    linear_operator_test_util.add_tests(LinearOperatorLowerTriangularTest)
    test.main()