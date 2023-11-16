import numpy as np
from tensorflow.python.framework import config
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.ops.linalg import linalg as linalg_lib
from tensorflow.python.ops.linalg import linear_operator_adjoint
from tensorflow.python.ops.linalg import linear_operator_test_util
from tensorflow.python.platform import test
linalg = linalg_lib
LinearOperatorAdjoint = linear_operator_adjoint.LinearOperatorAdjoint

@test_util.run_all_in_graph_and_eager_modes
class LinearOperatorAdjointTest(linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
    """Most tests done in the base class LinearOperatorDerivedClassTest."""

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        config.enable_tensor_float_32_execution(self.tf32_keep_)

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.tf32_keep_ = config.tensor_float_32_execution_enabled()
        config.enable_tensor_float_32_execution(False)
        self._atol[dtypes.complex64] = 1e-05
        self._rtol[dtypes.complex64] = 1e-05

    def operator_and_matrix(self, build_info, dtype, use_placeholder, ensure_self_adjoint_and_pd=False):
        if False:
            for i in range(10):
                print('nop')
        shape = list(build_info.shape)
        if ensure_self_adjoint_and_pd:
            matrix = linear_operator_test_util.random_positive_definite_matrix(shape, dtype, force_well_conditioned=True)
        else:
            matrix = linear_operator_test_util.random_tril_matrix(shape, dtype, force_well_conditioned=True, remove_upper=True)
        lin_op_matrix = matrix
        if use_placeholder:
            lin_op_matrix = array_ops.placeholder_with_default(matrix, shape=None)
        if ensure_self_adjoint_and_pd:
            operator = LinearOperatorAdjoint(linalg.LinearOperatorFullMatrix(lin_op_matrix, is_positive_definite=True, is_self_adjoint=True))
        else:
            operator = LinearOperatorAdjoint(linalg.LinearOperatorLowerTriangular(lin_op_matrix))
        return (operator, linalg.adjoint(matrix))

    def test_base_operator_hint_used(self):
        if False:
            for i in range(10):
                print('nop')
        matrix = [[1.0, 0.0], [1.0, 1.0]]
        operator = linalg.LinearOperatorFullMatrix(matrix, is_positive_definite=True, is_non_singular=True, is_self_adjoint=False)
        operator_adjoint = operator.adjoint()
        self.assertIsInstance(operator_adjoint, LinearOperatorAdjoint)
        self.assertTrue(operator_adjoint.is_positive_definite)
        self.assertTrue(operator_adjoint.is_non_singular)
        self.assertFalse(operator_adjoint.is_self_adjoint)

    def test_adjoint_of_adjoint_is_operator(self):
        if False:
            i = 10
            return i + 15
        matrix = [[1.0, 0.0], [1.0, 1.0]]
        operator = linalg.LinearOperatorFullMatrix(matrix)
        operator_adjoint = operator.adjoint()
        self.assertIsInstance(operator_adjoint, LinearOperatorAdjoint)
        adjoint_of_op_adjoint = operator_adjoint.adjoint()
        self.assertIsInstance(adjoint_of_op_adjoint, linalg.LinearOperatorFullMatrix)

    def test_supplied_hint_used(self):
        if False:
            print('Hello World!')
        matrix = [[1.0, 0.0], [1.0, 1.0]]
        operator = linalg.LinearOperatorFullMatrix(matrix)
        operator_adjoint = LinearOperatorAdjoint(operator, is_positive_definite=True, is_non_singular=True, is_self_adjoint=False)
        self.assertTrue(operator_adjoint.is_positive_definite)
        self.assertTrue(operator_adjoint.is_non_singular)
        self.assertFalse(operator_adjoint.is_self_adjoint)

    def test_contradicting_hints_raise(self):
        if False:
            while True:
                i = 10
        matrix = [[1.0, 0.0], [1.0, 1.0]]
        operator = linalg.LinearOperatorFullMatrix(matrix, is_positive_definite=False)
        with self.assertRaisesRegex(ValueError, 'positive-definite'):
            LinearOperatorAdjoint(operator, is_positive_definite=True)
        operator = linalg.LinearOperatorFullMatrix(matrix, is_self_adjoint=False)
        with self.assertRaisesRegex(ValueError, 'self-adjoint'):
            LinearOperatorAdjoint(operator, is_self_adjoint=True)

    def test_name(self):
        if False:
            while True:
                i = 10
        matrix = [[11.0, 0.0], [1.0, 8.0]]
        operator = linalg.LinearOperatorFullMatrix(matrix, name='my_operator', is_non_singular=True)
        operator = LinearOperatorAdjoint(operator)
        self.assertEqual('my_operator_adjoint', operator.name)

    def test_matmul_adjoint_operator(self):
        if False:
            return 10
        matrix1 = np.random.randn(4, 4)
        matrix2 = np.random.randn(4, 4)
        full_matrix1 = linalg.LinearOperatorFullMatrix(matrix1)
        full_matrix2 = linalg.LinearOperatorFullMatrix(matrix2)
        self.assertAllClose(np.matmul(matrix1, matrix2.T), self.evaluate(full_matrix1.matmul(full_matrix2, adjoint_arg=True).to_dense()))
        self.assertAllClose(np.matmul(matrix1.T, matrix2), self.evaluate(full_matrix1.matmul(full_matrix2, adjoint=True).to_dense()))
        self.assertAllClose(np.matmul(matrix1.T, matrix2.T), self.evaluate(full_matrix1.matmul(full_matrix2, adjoint=True, adjoint_arg=True).to_dense()))

    def test_matmul_adjoint_complex_operator(self):
        if False:
            for i in range(10):
                print('nop')
        matrix1 = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
        matrix2 = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
        full_matrix1 = linalg.LinearOperatorFullMatrix(matrix1)
        full_matrix2 = linalg.LinearOperatorFullMatrix(matrix2)
        self.assertAllClose(np.matmul(matrix1, matrix2.conj().T), self.evaluate(full_matrix1.matmul(full_matrix2, adjoint_arg=True).to_dense()))
        self.assertAllClose(np.matmul(matrix1.conj().T, matrix2), self.evaluate(full_matrix1.matmul(full_matrix2, adjoint=True).to_dense()))
        self.assertAllClose(np.matmul(matrix1.conj().T, matrix2.conj().T), self.evaluate(full_matrix1.matmul(full_matrix2, adjoint=True, adjoint_arg=True).to_dense()))

    def test_matvec(self):
        if False:
            i = 10
            return i + 15
        matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
        x = np.array([1.0, 2.0])
        operator = linalg.LinearOperatorFullMatrix(matrix)
        self.assertAllClose(matrix.dot(x), self.evaluate(operator.matvec(x)))
        self.assertAllClose(matrix.T.dot(x), self.evaluate(operator.H.matvec(x)))

    def test_solve_adjoint_operator(self):
        if False:
            while True:
                i = 10
        matrix1 = self.evaluate(linear_operator_test_util.random_tril_matrix([4, 4], dtype=dtypes.float64, force_well_conditioned=True))
        matrix2 = np.random.randn(4, 4)
        full_matrix1 = linalg.LinearOperatorLowerTriangular(matrix1, is_non_singular=True)
        full_matrix2 = linalg.LinearOperatorFullMatrix(matrix2)
        self.assertAllClose(self.evaluate(linalg.triangular_solve(matrix1, matrix2.T)), self.evaluate(full_matrix1.solve(full_matrix2, adjoint_arg=True).to_dense()))
        self.assertAllClose(self.evaluate(linalg.triangular_solve(matrix1.T, matrix2, lower=False)), self.evaluate(full_matrix1.solve(full_matrix2, adjoint=True).to_dense()))
        self.assertAllClose(self.evaluate(linalg.triangular_solve(matrix1.T, matrix2.T, lower=False)), self.evaluate(full_matrix1.solve(full_matrix2, adjoint=True, adjoint_arg=True).to_dense()))

    def test_solve_adjoint_complex_operator(self):
        if False:
            while True:
                i = 10
        matrix1 = self.evaluate(linear_operator_test_util.random_tril_matrix([4, 4], dtype=dtypes.complex128, force_well_conditioned=True) + 1j * linear_operator_test_util.random_tril_matrix([4, 4], dtype=dtypes.complex128, force_well_conditioned=True))
        matrix2 = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
        full_matrix1 = linalg.LinearOperatorLowerTriangular(matrix1, is_non_singular=True)
        full_matrix2 = linalg.LinearOperatorFullMatrix(matrix2)
        self.assertAllClose(self.evaluate(linalg.triangular_solve(matrix1, matrix2.conj().T)), self.evaluate(full_matrix1.solve(full_matrix2, adjoint_arg=True).to_dense()))
        self.assertAllClose(self.evaluate(linalg.triangular_solve(matrix1.conj().T, matrix2, lower=False)), self.evaluate(full_matrix1.solve(full_matrix2, adjoint=True).to_dense()))
        self.assertAllClose(self.evaluate(linalg.triangular_solve(matrix1.conj().T, matrix2.conj().T, lower=False)), self.evaluate(full_matrix1.solve(full_matrix2, adjoint=True, adjoint_arg=True).to_dense()))

    def test_solvevec(self):
        if False:
            return 10
        matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
        inv_matrix = np.linalg.inv(matrix)
        x = np.array([1.0, 2.0])
        operator = linalg.LinearOperatorFullMatrix(matrix)
        self.assertAllClose(inv_matrix.dot(x), self.evaluate(operator.solvevec(x)))
        self.assertAllClose(inv_matrix.T.dot(x), self.evaluate(operator.H.solvevec(x)))

    def test_tape_safe(self):
        if False:
            return 10
        matrix = variables_module.Variable([[1.0, 2.0], [3.0, 4.0]])
        operator = LinearOperatorAdjoint(linalg.LinearOperatorFullMatrix(matrix))
        self.check_tape_safe(operator)

@test_util.run_all_in_graph_and_eager_modes
class LinearOperatorAdjointNonSquareTest(linear_operator_test_util.NonSquareLinearOperatorDerivedClassTest):
    """Tests done in the base class NonSquareLinearOperatorDerivedClassTest."""

    def operator_and_matrix(self, build_info, dtype, use_placeholder):
        if False:
            for i in range(10):
                print('nop')
        shape_before_adjoint = list(build_info.shape)
        (shape_before_adjoint[-1], shape_before_adjoint[-2]) = (shape_before_adjoint[-2], shape_before_adjoint[-1])
        matrix = linear_operator_test_util.random_normal(shape_before_adjoint, dtype=dtype)
        lin_op_matrix = matrix
        if use_placeholder:
            lin_op_matrix = array_ops.placeholder_with_default(matrix, shape=None)
        operator = LinearOperatorAdjoint(linalg.LinearOperatorFullMatrix(lin_op_matrix))
        return (operator, linalg.adjoint(matrix))
if __name__ == '__main__':
    linear_operator_test_util.add_tests(LinearOperatorAdjointTest)
    test.main()