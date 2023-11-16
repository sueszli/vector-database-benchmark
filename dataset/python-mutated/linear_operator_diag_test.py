from tensorflow.python.framework import config
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.ops.linalg import linalg as linalg_lib
from tensorflow.python.ops.linalg import linear_operator_test_util
from tensorflow.python.platform import test
linalg = linalg_lib

@test_util.run_all_in_graph_and_eager_modes
class LinearOperatorDiagTest(linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
    """Most tests done in the base class LinearOperatorDerivedClassTest."""

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        config.enable_tensor_float_32_execution(self.tf32_keep_)

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.tf32_keep_ = config.tensor_float_32_execution_enabled()
        config.enable_tensor_float_32_execution(False)

    @staticmethod
    def optional_tests():
        if False:
            print('Hello World!')
        'List of optional test names to run.'
        return ['operator_matmul_with_same_type', 'operator_solve_with_same_type']

    def operator_and_matrix(self, build_info, dtype, use_placeholder, ensure_self_adjoint_and_pd=False):
        if False:
            return 10
        shape = list(build_info.shape)
        diag = linear_operator_test_util.random_sign_uniform(shape[:-1], minval=1.0, maxval=2.0, dtype=dtype)
        if ensure_self_adjoint_and_pd:
            diag = math_ops.cast(math_ops.abs(diag), dtype=dtype)
        lin_op_diag = diag
        if use_placeholder:
            lin_op_diag = array_ops.placeholder_with_default(diag, shape=None)
        operator = linalg.LinearOperatorDiag(lin_op_diag, is_self_adjoint=True if ensure_self_adjoint_and_pd else None, is_positive_definite=True if ensure_self_adjoint_and_pd else None)
        matrix = array_ops.matrix_diag(diag)
        return (operator, matrix)

    def test_assert_positive_definite_raises_for_zero_eigenvalue(self):
        if False:
            while True:
                i = 10
        with self.cached_session():
            diag = [1.0, 0.0]
            operator = linalg.LinearOperatorDiag(diag)
            self.assertTrue(operator.is_self_adjoint)
            with self.assertRaisesOpError('non-positive.*not positive definite'):
                operator.assert_positive_definite().run()

    def test_assert_positive_definite_raises_for_negative_real_eigvalues(self):
        if False:
            i = 10
            return i + 15
        with self.cached_session():
            diag_x = [1.0, -2.0]
            diag_y = [0.0, 0.0]
            diag = math_ops.complex(diag_x, diag_y)
            operator = linalg.LinearOperatorDiag(diag)
            self.assertTrue(operator.is_self_adjoint is None)
            with self.assertRaisesOpError('non-positive real.*not positive definite'):
                operator.assert_positive_definite().run()

    def test_assert_positive_definite_does_not_raise_if_pd_and_complex(self):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session():
            x = [1.0, 2.0]
            y = [1.0, 0.0]
            diag = math_ops.complex(x, y)
            self.evaluate(linalg.LinearOperatorDiag(diag).assert_positive_definite())

    def test_assert_non_singular_raises_if_zero_eigenvalue(self):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session():
            diag = [1.0, 0.0]
            operator = linalg.LinearOperatorDiag(diag, is_self_adjoint=True)
            with self.assertRaisesOpError('Singular operator'):
                operator.assert_non_singular().run()

    def test_assert_non_singular_does_not_raise_for_complex_nonsingular(self):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session():
            x = [1.0, 0.0]
            y = [0.0, 1.0]
            diag = math_ops.complex(x, y)
            self.evaluate(linalg.LinearOperatorDiag(diag).assert_non_singular())

    def test_assert_self_adjoint_raises_if_diag_has_complex_part(self):
        if False:
            print('Hello World!')
        with self.cached_session():
            x = [1.0, 0.0]
            y = [0.0, 1.0]
            diag = math_ops.complex(x, y)
            operator = linalg.LinearOperatorDiag(diag)
            with self.assertRaisesOpError('imaginary.*not self-adjoint'):
                operator.assert_self_adjoint().run()

    def test_assert_self_adjoint_does_not_raise_for_diag_with_zero_imag(self):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session():
            x = [1.0, 0.0]
            y = [0.0, 0.0]
            diag = math_ops.complex(x, y)
            operator = linalg.LinearOperatorDiag(diag)
            self.evaluate(operator.assert_self_adjoint())

    def test_scalar_diag_raises(self):
        if False:
            while True:
                i = 10
        with self.assertRaisesRegex(ValueError, 'must have at least 1 dimension'):
            linalg.LinearOperatorDiag(1.0)

    def test_broadcast_matmul_and_solve(self):
        if False:
            return 10
        with self.cached_session() as sess:
            x = random_ops.random_normal(shape=(2, 2, 3, 4))
            diag = random_ops.random_uniform(shape=(2, 1, 3))
            operator = linalg.LinearOperatorDiag(diag, is_self_adjoint=True)
            self.assertAllEqual((2, 1, 3, 3), operator.shape)
            diag_broadcast = array_ops.concat((diag, diag), 1)
            mat = array_ops.matrix_diag(diag_broadcast)
            self.assertAllEqual((2, 2, 3, 3), mat.shape)
            operator_matmul = operator.matmul(x)
            mat_matmul = math_ops.matmul(mat, x)
            self.assertAllEqual(operator_matmul.shape, mat_matmul.shape)
            self.assertAllClose(*self.evaluate([operator_matmul, mat_matmul]))
            operator_solve = operator.solve(x)
            mat_solve = linalg_ops.matrix_solve(mat, x)
            self.assertAllEqual(operator_solve.shape, mat_solve.shape)
            self.assertAllClose(*self.evaluate([operator_solve, mat_solve]))

    def test_diag_matmul(self):
        if False:
            return 10
        operator1 = linalg_lib.LinearOperatorDiag([2.0, 3.0])
        operator2 = linalg_lib.LinearOperatorDiag([1.0, 2.0])
        operator3 = linalg_lib.LinearOperatorScaledIdentity(num_rows=2, multiplier=3.0)
        operator_matmul = operator1.matmul(operator2)
        self.assertTrue(isinstance(operator_matmul, linalg_lib.LinearOperatorDiag))
        self.assertAllClose([2.0, 6.0], self.evaluate(operator_matmul.diag))
        operator_matmul = operator2.matmul(operator1)
        self.assertTrue(isinstance(operator_matmul, linalg_lib.LinearOperatorDiag))
        self.assertAllClose([2.0, 6.0], self.evaluate(operator_matmul.diag))
        operator_matmul = operator1.matmul(operator3)
        self.assertTrue(isinstance(operator_matmul, linalg_lib.LinearOperatorDiag))
        self.assertAllClose([6.0, 9.0], self.evaluate(operator_matmul.diag))
        operator_matmul = operator3.matmul(operator1)
        self.assertTrue(isinstance(operator_matmul, linalg_lib.LinearOperatorDiag))
        self.assertAllClose([6.0, 9.0], self.evaluate(operator_matmul.diag))

    def test_diag_solve(self):
        if False:
            return 10
        operator1 = linalg_lib.LinearOperatorDiag([2.0, 3.0], is_non_singular=True)
        operator2 = linalg_lib.LinearOperatorDiag([1.0, 2.0], is_non_singular=True)
        operator3 = linalg_lib.LinearOperatorScaledIdentity(num_rows=2, multiplier=3.0, is_non_singular=True)
        operator_solve = operator1.solve(operator2)
        self.assertTrue(isinstance(operator_solve, linalg_lib.LinearOperatorDiag))
        self.assertAllClose([0.5, 2 / 3.0], self.evaluate(operator_solve.diag))
        operator_solve = operator2.solve(operator1)
        self.assertTrue(isinstance(operator_solve, linalg_lib.LinearOperatorDiag))
        self.assertAllClose([2.0, 3 / 2.0], self.evaluate(operator_solve.diag))
        operator_solve = operator1.solve(operator3)
        self.assertTrue(isinstance(operator_solve, linalg_lib.LinearOperatorDiag))
        self.assertAllClose([3 / 2.0, 1.0], self.evaluate(operator_solve.diag))
        operator_solve = operator3.solve(operator1)
        self.assertTrue(isinstance(operator_solve, linalg_lib.LinearOperatorDiag))
        self.assertAllClose([2 / 3.0, 1.0], self.evaluate(operator_solve.diag))

    def test_diag_adjoint_type(self):
        if False:
            return 10
        diag = [1.0, 3.0, 5.0, 8.0]
        operator = linalg.LinearOperatorDiag(diag, is_non_singular=True)
        self.assertIsInstance(operator.adjoint(), linalg.LinearOperatorDiag)

    def test_diag_cholesky_type(self):
        if False:
            for i in range(10):
                print('nop')
        diag = [1.0, 3.0, 5.0, 8.0]
        operator = linalg.LinearOperatorDiag(diag, is_positive_definite=True, is_self_adjoint=True)
        self.assertIsInstance(operator.cholesky(), linalg.LinearOperatorDiag)

    def test_diag_inverse_type(self):
        if False:
            print('Hello World!')
        diag = [1.0, 3.0, 5.0, 8.0]
        operator = linalg.LinearOperatorDiag(diag, is_non_singular=True)
        self.assertIsInstance(operator.inverse(), linalg.LinearOperatorDiag)

    def test_tape_safe(self):
        if False:
            for i in range(10):
                print('nop')
        diag = variables_module.Variable([[2.0]])
        operator = linalg.LinearOperatorDiag(diag)
        self.check_tape_safe(operator)

    def test_convert_variables_to_tensors(self):
        if False:
            i = 10
            return i + 15
        diag = variables_module.Variable([[2.0]])
        operator = linalg.LinearOperatorDiag(diag)
        with self.cached_session() as sess:
            sess.run([diag.initializer])
            self.check_convert_variables_to_tensors(operator)
if __name__ == '__main__':
    linear_operator_test_util.add_tests(LinearOperatorDiagTest)
    test.main()