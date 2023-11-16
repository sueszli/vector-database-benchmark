import numpy as np
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.ops.linalg import linalg as linalg_lib
from tensorflow.python.ops.linalg import linear_operator_block_lower_triangular as block_lower_triangular
from tensorflow.python.ops.linalg import linear_operator_test_util
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.platform import test
linalg = linalg_lib
rng = np.random.RandomState(0)

def _block_lower_triangular_dense(expected_shape, blocks):
    if False:
        while True:
            i = 10
    'Convert a list of blocks into a dense blockwise lower-triangular matrix.'
    rows = []
    num_cols = 0
    for row_blocks in blocks:
        batch_row_shape = array_ops.shape(row_blocks[0])[:-1]
        num_cols += array_ops.shape(row_blocks[-1])[-1]
        zeros_to_pad_after_shape = array_ops.concat([batch_row_shape, [expected_shape[-2] - num_cols]], axis=-1)
        zeros_to_pad_after = array_ops.zeros(zeros_to_pad_after_shape, dtype=row_blocks[-1].dtype)
        row_blocks.append(zeros_to_pad_after)
        rows.append(array_ops.concat(row_blocks, axis=-1))
    return array_ops.concat(rows, axis=-2)

@test_util.run_all_in_graph_and_eager_modes
class SquareLinearOperatorBlockLowerTriangularTest(linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
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
        self._atol[dtypes.float32] = 1e-05
        self._atol[dtypes.complex64] = 1e-05
        self._rtol[dtypes.float32] = 1e-05
        self._rtol[dtypes.complex64] = 1e-05
        super(SquareLinearOperatorBlockLowerTriangularTest, self).setUp()

    @staticmethod
    def use_blockwise_arg():
        if False:
            while True:
                i = 10
        return True

    @staticmethod
    def skip_these_tests():
        if False:
            print('Hello World!')
        return ['cholesky', 'eigvalsh']

    @staticmethod
    def operator_shapes_infos():
        if False:
            while True:
                i = 10
        shape_info = linear_operator_test_util.OperatorShapesInfo
        return [shape_info((0, 0)), shape_info((1, 1)), shape_info((1, 3, 3)), shape_info((5, 5), blocks=[[(2, 2)], [(3, 2), (3, 3)]]), shape_info((3, 7, 7), blocks=[[(1, 2, 2)], [(1, 3, 2), (3, 3, 3)], [(1, 2, 2), (1, 2, 3), (1, 2, 2)]]), shape_info((2, 4, 6, 6), blocks=[[(2, 1, 2, 2)], [(1, 4, 2), (4, 4, 4)]])]

    def operator_and_matrix(self, shape_info, dtype, use_placeholder, ensure_self_adjoint_and_pd=False):
        if False:
            print('Hello World!')
        expected_blocks = shape_info.__dict__['blocks'] if 'blocks' in shape_info.__dict__ else [[list(shape_info.shape)]]
        matrices = []
        for (i, row_shapes) in enumerate(expected_blocks):
            row = []
            for (j, block_shape) in enumerate(row_shapes):
                if i == j:
                    row.append(linear_operator_test_util.random_positive_definite_matrix(block_shape, dtype, force_well_conditioned=True))
                else:
                    row.append(linear_operator_test_util.random_normal(block_shape, dtype=dtype))
            matrices.append(row)
        lin_op_matrices = matrices
        if use_placeholder:
            lin_op_matrices = [[array_ops.placeholder_with_default(matrix, shape=None) for matrix in row] for row in matrices]
        operator = block_lower_triangular.LinearOperatorBlockLowerTriangular([[linalg.LinearOperatorFullMatrix(l, is_square=True, is_self_adjoint=True if ensure_self_adjoint_and_pd else None, is_positive_definite=True if ensure_self_adjoint_and_pd else None) for l in row] for row in lin_op_matrices])
        self.assertTrue(operator.is_square)
        expected_shape = list(shape_info.shape)
        broadcasted_matrices = linear_operator_util.broadcast_matrix_batch_dims([op for row in matrices for op in row])
        matrices = [broadcasted_matrices[i * (i + 1) // 2:(i + 1) * (i + 2) // 2] for i in range(len(matrices))]
        block_lower_triangular_dense = _block_lower_triangular_dense(expected_shape, matrices)
        if not use_placeholder:
            block_lower_triangular_dense.set_shape(expected_shape)
        return (operator, block_lower_triangular_dense)

    def test_is_x_flags(self):
        if False:
            return 10
        matrix = [[1.0, 0.0], [1.0, 1.0]]
        operator = block_lower_triangular.LinearOperatorBlockLowerTriangular([[linalg.LinearOperatorFullMatrix(matrix)]], is_positive_definite=True, is_non_singular=True, is_self_adjoint=False)
        self.assertTrue(operator.is_positive_definite)
        self.assertTrue(operator.is_non_singular)
        self.assertFalse(operator.is_self_adjoint)

    def test_block_lower_triangular_inverse_type(self):
        if False:
            return 10
        matrix = [[1.0, 0.0], [0.0, 1.0]]
        operator = block_lower_triangular.LinearOperatorBlockLowerTriangular([[linalg.LinearOperatorFullMatrix(matrix, is_non_singular=True)], [linalg.LinearOperatorFullMatrix(matrix, is_non_singular=True), linalg.LinearOperatorFullMatrix(matrix, is_non_singular=True)]], is_non_singular=True)
        inverse = operator.inverse()
        self.assertIsInstance(inverse, block_lower_triangular.LinearOperatorBlockLowerTriangular)
        self.assertEqual(2, len(inverse.operators))
        self.assertEqual(1, len(inverse.operators[0]))
        self.assertEqual(2, len(inverse.operators[1]))

    def test_tape_safe(self):
        if False:
            return 10
        operator_1 = linalg.LinearOperatorFullMatrix(variables_module.Variable([[1.0, 0.0], [0.0, 1.0]]), is_self_adjoint=True, is_positive_definite=True)
        operator_2 = linalg.LinearOperatorFullMatrix(variables_module.Variable([[2.0, 0.0], [1.0, 0.0]]))
        operator_3 = linalg.LinearOperatorFullMatrix(variables_module.Variable([[3.0, 1.0], [1.0, 3.0]]), is_self_adjoint=True, is_positive_definite=True)
        operator = block_lower_triangular.LinearOperatorBlockLowerTriangular([[operator_1], [operator_2, operator_3]], is_self_adjoint=False, is_positive_definite=True)
        diagonal_grads_only = ['diag_part', 'trace', 'determinant', 'log_abs_determinant']
        self.check_tape_safe(operator, skip_options=diagonal_grads_only)
        for y in diagonal_grads_only:
            for diag_block in [operator_1, operator_3]:
                with backprop.GradientTape() as tape:
                    grads = tape.gradient(getattr(operator, y)(), diag_block.variables)
                    for item in grads:
                        self.assertIsNotNone(item)

    def test_convert_variables_to_tensors(self):
        if False:
            return 10
        operator_1 = linalg.LinearOperatorFullMatrix(variables_module.Variable([[1.0, 0.0], [0.0, 1.0]]), is_self_adjoint=True, is_positive_definite=True)
        operator_2 = linalg.LinearOperatorFullMatrix(variables_module.Variable([[2.0, 0.0], [1.0, 0.0]]))
        operator_3 = linalg.LinearOperatorFullMatrix(variables_module.Variable([[3.0, 1.0], [1.0, 3.0]]), is_self_adjoint=True, is_positive_definite=True)
        operator = block_lower_triangular.LinearOperatorBlockLowerTriangular([[operator_1], [operator_2, operator_3]], is_self_adjoint=False, is_positive_definite=True)
        with self.cached_session() as sess:
            sess.run([x.initializer for x in operator.variables])
            self.check_convert_variables_to_tensors(operator)

    def test_is_non_singular_auto_set(self):
        if False:
            while True:
                i = 10
        matrix = [[11.0, 0.0], [1.0, 8.0]]
        operator_1 = linalg.LinearOperatorFullMatrix(matrix, is_non_singular=True)
        operator_2 = linalg.LinearOperatorFullMatrix(matrix, is_non_singular=True)
        operator_3 = linalg.LinearOperatorFullMatrix(matrix, is_non_singular=True)
        operator = block_lower_triangular.LinearOperatorBlockLowerTriangular([[operator_1], [operator_2, operator_3]], is_positive_definite=False, is_non_singular=None)
        self.assertFalse(operator.is_positive_definite)
        self.assertTrue(operator.is_non_singular)
        with self.assertRaisesRegex(ValueError, 'always non-singular'):
            block_lower_triangular.LinearOperatorBlockLowerTriangular([[operator_1], [operator_2, operator_3]], is_non_singular=False)
        operator_4 = linalg.LinearOperatorFullMatrix([[1.0, 0.0], [2.0, 0.0]], is_non_singular=False)
        block_lower_triangular.LinearOperatorBlockLowerTriangular([[operator_1], [operator_4, operator_2]], is_non_singular=True)
        with self.assertRaisesRegex(ValueError, 'always singular'):
            block_lower_triangular.LinearOperatorBlockLowerTriangular([[operator_1], [operator_2, operator_4]], is_non_singular=True)

    def test_different_dtypes_raises(self):
        if False:
            while True:
                i = 10
        operators = [[linalg.LinearOperatorFullMatrix(rng.rand(2, 3, 3))], [linalg.LinearOperatorFullMatrix(rng.rand(2, 3, 3)), linalg.LinearOperatorFullMatrix(rng.rand(2, 3, 3).astype(np.float32))]]
        with self.assertRaisesRegex(TypeError, 'same dtype'):
            block_lower_triangular.LinearOperatorBlockLowerTriangular(operators)

    def test_non_square_operator_raises(self):
        if False:
            return 10
        operators = [[linalg.LinearOperatorFullMatrix(rng.rand(3, 4), is_square=False)], [linalg.LinearOperatorFullMatrix(rng.rand(4, 4)), linalg.LinearOperatorFullMatrix(rng.rand(4, 4))]]
        with self.assertRaisesRegex(ValueError, 'must be square'):
            block_lower_triangular.LinearOperatorBlockLowerTriangular(operators)

    def test_empty_operators_raises(self):
        if False:
            return 10
        with self.assertRaisesRegex(ValueError, 'must be a list of >=1'):
            block_lower_triangular.LinearOperatorBlockLowerTriangular([])

    def test_operators_wrong_length_raises(self):
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(ValueError, 'must contain `2` blocks'):
            block_lower_triangular.LinearOperatorBlockLowerTriangular([[linalg.LinearOperatorFullMatrix(rng.rand(2, 2))], [linalg.LinearOperatorFullMatrix(rng.rand(2, 2)) for _ in range(3)]])

    def test_operators_mismatched_dimension_raises(self):
        if False:
            for i in range(10):
                print('nop')
        operators = [[linalg.LinearOperatorFullMatrix(rng.rand(3, 3))], [linalg.LinearOperatorFullMatrix(rng.rand(3, 4)), linalg.LinearOperatorFullMatrix(rng.rand(3, 3))]]
        with self.assertRaisesRegex(ValueError, 'must be the same as'):
            block_lower_triangular.LinearOperatorBlockLowerTriangular(operators)

    def test_incompatible_input_blocks_raises(self):
        if False:
            while True:
                i = 10
        matrix_1 = array_ops.placeholder_with_default(rng.rand(4, 4), shape=None)
        matrix_2 = array_ops.placeholder_with_default(rng.rand(3, 4), shape=None)
        matrix_3 = array_ops.placeholder_with_default(rng.rand(3, 3), shape=None)
        operators = [[linalg.LinearOperatorFullMatrix(matrix_1, is_square=True)], [linalg.LinearOperatorFullMatrix(matrix_2), linalg.LinearOperatorFullMatrix(matrix_3, is_square=True)]]
        operator = block_lower_triangular.LinearOperatorBlockLowerTriangular(operators)
        x = np.random.rand(2, 4, 5).tolist()
        msg = 'dimension does not match' if context.executing_eagerly() else 'input structure is ambiguous'
        with self.assertRaisesRegex(ValueError, msg):
            operator.matmul(x)

    def test_composite_gradients(self):
        if False:
            print('Hello World!')
        with backprop.GradientTape() as tape:
            op1 = linalg.LinearOperatorFullMatrix(rng.rand(4, 4), is_square=True)
            op2 = linalg.LinearOperatorFullMatrix(rng.rand(3, 4))
            op3 = linalg.LinearOperatorFullMatrix(rng.rand(3, 3), is_square=True)
            tape.watch([op1, op2, op3])
            operator = block_lower_triangular.LinearOperatorBlockLowerTriangular([[op1], [op2, op3]])
            x = self.make_x(op1, adjoint=False)
            y = op1.matmul(x)
            (connected_grad, disconnected_grad, composite_grad) = tape.gradient(y, [op1, op3, operator])
        disconnected_component_grad = composite_grad.operators[1][1].to_dense()
        self.assertAllClose(connected_grad.to_dense(), composite_grad.operators[0][0].to_dense())
        self.assertAllClose(disconnected_component_grad, array_ops.zeros_like(disconnected_component_grad))
        self.assertIsNone(disconnected_grad)
if __name__ == '__main__':
    linear_operator_test_util.add_tests(SquareLinearOperatorBlockLowerTriangularTest)
    test.main()