from absl.testing import parameterized
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linalg as linalg_lib
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.platform import test
rng = np.random.RandomState(0)

class AssertZeroImagPartTest(test.TestCase):

    def test_real_tensor_doesnt_raise(self):
        if False:
            for i in range(10):
                print('nop')
        x = ops.convert_to_tensor([0.0, 2, 3])
        self.evaluate(linear_operator_util.assert_zero_imag_part(x, message='ABC123'))

    def test_complex_tensor_with_imag_zero_doesnt_raise(self):
        if False:
            return 10
        x = ops.convert_to_tensor([1.0, 0, 3])
        y = ops.convert_to_tensor([0.0, 0, 0])
        z = math_ops.complex(x, y)
        self.evaluate(linear_operator_util.assert_zero_imag_part(z, message='ABC123'))

    def test_complex_tensor_with_nonzero_imag_raises(self):
        if False:
            return 10
        x = ops.convert_to_tensor([1.0, 2, 0])
        y = ops.convert_to_tensor([1.0, 2, 0])
        z = math_ops.complex(x, y)
        with self.assertRaisesOpError('ABC123'):
            self.evaluate(linear_operator_util.assert_zero_imag_part(z, message='ABC123'))

class AssertNoEntriesWithModulusZeroTest(test.TestCase):

    def test_nonzero_real_tensor_doesnt_raise(self):
        if False:
            while True:
                i = 10
        x = ops.convert_to_tensor([1.0, 2, 3])
        self.evaluate(linear_operator_util.assert_no_entries_with_modulus_zero(x, message='ABC123'))

    def test_nonzero_complex_tensor_doesnt_raise(self):
        if False:
            return 10
        x = ops.convert_to_tensor([1.0, 0, 3])
        y = ops.convert_to_tensor([1.0, 2, 0])
        z = math_ops.complex(x, y)
        self.evaluate(linear_operator_util.assert_no_entries_with_modulus_zero(z, message='ABC123'))

    def test_zero_real_tensor_raises(self):
        if False:
            print('Hello World!')
        x = ops.convert_to_tensor([1.0, 0, 3])
        with self.assertRaisesOpError('ABC123'):
            self.evaluate(linear_operator_util.assert_no_entries_with_modulus_zero(x, message='ABC123'))

    def test_zero_complex_tensor_raises(self):
        if False:
            return 10
        x = ops.convert_to_tensor([1.0, 2, 0])
        y = ops.convert_to_tensor([1.0, 2, 0])
        z = math_ops.complex(x, y)
        with self.assertRaisesOpError('ABC123'):
            self.evaluate(linear_operator_util.assert_no_entries_with_modulus_zero(z, message='ABC123'))

class BroadcastMatrixBatchDimsTest(test.TestCase):

    def test_zero_batch_matrices_returned_as_empty_list(self):
        if False:
            while True:
                i = 10
        self.assertAllEqual([], linear_operator_util.broadcast_matrix_batch_dims([]))

    def test_one_batch_matrix_returned_after_tensor_conversion(self):
        if False:
            i = 10
            return i + 15
        arr = rng.rand(2, 3, 4)
        (tensor,) = linear_operator_util.broadcast_matrix_batch_dims([arr])
        self.assertTrue(isinstance(tensor, tensor_lib.Tensor))
        self.assertAllClose(arr, self.evaluate(tensor))

    def test_static_dims_broadcast(self):
        if False:
            print('Hello World!')
        x = rng.rand(3, 1, 2, 1, 5)
        y = rng.rand(4, 1, 3, 7)
        batch_of_zeros = np.zeros((3, 4, 2, 1, 1))
        x_bc_expected = x + batch_of_zeros
        y_bc_expected = y + batch_of_zeros
        (x_bc, y_bc) = linear_operator_util.broadcast_matrix_batch_dims([x, y])
        self.assertAllEqual(x_bc_expected.shape, x_bc.shape)
        self.assertAllEqual(y_bc_expected.shape, y_bc.shape)
        (x_bc_, y_bc_) = self.evaluate([x_bc, y_bc])
        self.assertAllClose(x_bc_expected, x_bc_)
        self.assertAllClose(y_bc_expected, y_bc_)

    def test_static_dims_broadcast_second_arg_higher_rank(self):
        if False:
            return 10
        x = rng.rand(1, 2, 1, 5)
        y = rng.rand(1, 3, 2, 3, 7)
        batch_of_zeros = np.zeros((1, 3, 2, 1, 1))
        x_bc_expected = x + batch_of_zeros
        y_bc_expected = y + batch_of_zeros
        (x_bc, y_bc) = linear_operator_util.broadcast_matrix_batch_dims([x, y])
        self.assertAllEqual(x_bc_expected.shape, x_bc.shape)
        self.assertAllEqual(y_bc_expected.shape, y_bc.shape)
        (x_bc_, y_bc_) = self.evaluate([x_bc, y_bc])
        self.assertAllClose(x_bc_expected, x_bc_)
        self.assertAllClose(y_bc_expected, y_bc_)

    def test_dynamic_dims_broadcast_32bit(self):
        if False:
            return 10
        x = rng.rand(3, 1, 2, 1, 5).astype(np.float32)
        y = rng.rand(4, 1, 3, 7).astype(np.float32)
        batch_of_zeros = np.zeros((3, 4, 2, 1, 1)).astype(np.float32)
        x_bc_expected = x + batch_of_zeros
        y_bc_expected = y + batch_of_zeros
        x_ph = array_ops.placeholder_with_default(x, shape=None)
        y_ph = array_ops.placeholder_with_default(y, shape=None)
        (x_bc, y_bc) = linear_operator_util.broadcast_matrix_batch_dims([x_ph, y_ph])
        (x_bc_, y_bc_) = self.evaluate([x_bc, y_bc])
        self.assertAllClose(x_bc_expected, x_bc_)
        self.assertAllClose(y_bc_expected, y_bc_)

    def test_dynamic_dims_broadcast_32bit_second_arg_higher_rank(self):
        if False:
            print('Hello World!')
        x = rng.rand(1, 2, 1, 5).astype(np.float32)
        y = rng.rand(3, 4, 1, 3, 7).astype(np.float32)
        batch_of_zeros = np.zeros((3, 4, 2, 1, 1)).astype(np.float32)
        x_bc_expected = x + batch_of_zeros
        y_bc_expected = y + batch_of_zeros
        x_ph = array_ops.placeholder_with_default(x, shape=None)
        y_ph = array_ops.placeholder_with_default(y, shape=None)
        (x_bc, y_bc) = linear_operator_util.broadcast_matrix_batch_dims([x_ph, y_ph])
        (x_bc_, y_bc_) = self.evaluate([x_bc, y_bc])
        self.assertAllClose(x_bc_expected, x_bc_)
        self.assertAllClose(y_bc_expected, y_bc_)

    def test_less_than_two_dims_raises_static(self):
        if False:
            for i in range(10):
                print('nop')
        x = rng.rand(3)
        y = rng.rand(1, 1)
        with self.assertRaisesRegex(ValueError, 'at least two dimensions'):
            linear_operator_util.broadcast_matrix_batch_dims([x, y])
        with self.assertRaisesRegex(ValueError, 'at least two dimensions'):
            linear_operator_util.broadcast_matrix_batch_dims([y, x])

class MatrixSolveWithBroadcastTest(test.TestCase):

    def test_static_dims_broadcast_matrix_has_extra_dims(self):
        if False:
            return 10
        matrix = rng.rand(2, 3, 3)
        rhs = rng.rand(3, 7)
        rhs_broadcast = rhs + np.zeros((2, 1, 1))
        result = linear_operator_util.matrix_solve_with_broadcast(matrix, rhs)
        self.assertAllEqual((2, 3, 7), result.shape)
        expected = linalg_ops.matrix_solve(matrix, rhs_broadcast)
        self.assertAllClose(*self.evaluate([expected, result]))

    def test_static_dims_broadcast_rhs_has_extra_dims(self):
        if False:
            print('Hello World!')
        matrix = rng.rand(3, 3)
        rhs = rng.rand(2, 3, 2)
        matrix_broadcast = matrix + np.zeros((2, 1, 1))
        result = linear_operator_util.matrix_solve_with_broadcast(matrix, rhs)
        self.assertAllEqual((2, 3, 2), result.shape)
        expected = linalg_ops.matrix_solve(matrix_broadcast, rhs)
        self.assertAllClose(*self.evaluate([expected, result]))

    def test_static_dims_broadcast_rhs_has_extra_dims_dynamic(self):
        if False:
            while True:
                i = 10
        matrix = rng.rand(3, 3)
        rhs = rng.rand(2, 3, 2)
        matrix_broadcast = matrix + np.zeros((2, 1, 1))
        matrix_ph = array_ops.placeholder_with_default(matrix, shape=[None, None])
        rhs_ph = array_ops.placeholder_with_default(rhs, shape=[None, None, None])
        result = linear_operator_util.matrix_solve_with_broadcast(matrix_ph, rhs_ph)
        self.assertAllEqual(3, result.shape.ndims)
        expected = linalg_ops.matrix_solve(matrix_broadcast, rhs)
        self.assertAllClose(*self.evaluate([expected, result]))

    def test_static_dims_broadcast_rhs_has_extra_dims_and_adjoint(self):
        if False:
            for i in range(10):
                print('nop')
        matrix = rng.rand(3, 3)
        rhs = rng.rand(2, 3, 2)
        matrix_broadcast = matrix + np.zeros((2, 1, 1))
        result = linear_operator_util.matrix_solve_with_broadcast(matrix, rhs, adjoint=True)
        self.assertAllEqual((2, 3, 2), result.shape)
        expected = linalg_ops.matrix_solve(matrix_broadcast, rhs, adjoint=True)
        self.assertAllClose(*self.evaluate([expected, result]))

    def test_dynamic_dims_broadcast_64bit(self):
        if False:
            print('Hello World!')
        matrix = rng.rand(2, 3, 3)
        rhs = rng.rand(2, 1, 3, 7)
        matrix_broadcast = matrix + np.zeros((2, 2, 1, 1))
        rhs_broadcast = rhs + np.zeros((2, 2, 1, 1))
        matrix_ph = array_ops.placeholder_with_default(matrix, shape=None)
        rhs_ph = array_ops.placeholder_with_default(rhs, shape=None)
        (result, expected) = self.evaluate([linear_operator_util.matrix_solve_with_broadcast(matrix_ph, rhs_ph), linalg_ops.matrix_solve(matrix_broadcast, rhs_broadcast)])
        self.assertAllClose(expected, result)

class DomainDimensionStubOperator(object):

    def __init__(self, domain_dimension):
        if False:
            i = 10
            return i + 15
        self._domain_dimension = ops.convert_to_tensor(domain_dimension)

    def domain_dimension_tensor(self):
        if False:
            while True:
                i = 10
        return self._domain_dimension

class AssertCompatibleMatrixDimensionsTest(test.TestCase):

    def test_compatible_dimensions_do_not_raise(self):
        if False:
            while True:
                i = 10
        x = ops.convert_to_tensor(rng.rand(2, 3, 4))
        operator = DomainDimensionStubOperator(3)
        self.evaluate(linear_operator_util.assert_compatible_matrix_dimensions(operator, x))

    def test_incompatible_dimensions_raise(self):
        if False:
            for i in range(10):
                print('nop')
        x = ops.convert_to_tensor(rng.rand(2, 4, 4))
        operator = DomainDimensionStubOperator(3)
        with self.assertRaisesOpError('Dimensions are not compatible'):
            self.evaluate(linear_operator_util.assert_compatible_matrix_dimensions(operator, x))

class IsAdjointPairTest(test.TestCase):

    def test_one_is_explicitly_adjoint_of_other_returns_true(self):
        if False:
            i = 10
            return i + 15
        x = linalg_lib.LinearOperatorFullMatrix([[1.0, 2.0], [3.0, 4.0]], is_self_adjoint=False)
        self.assertTrue(linear_operator_util.is_adjoint_pair(x, x.H))
        self.assertTrue(linear_operator_util.is_adjoint_pair(x.H, x))

    def test_repeated_non_self_adjoint_operator_returns_false(self):
        if False:
            return 10
        x = linalg_lib.LinearOperatorFullMatrix([[1.0, 2.0], [3.0, 4.0]], is_self_adjoint=False)
        self.assertFalse(linear_operator_util.is_adjoint_pair(x, x))

    def test_repeated_self_adjoint_operator_returns_true(self):
        if False:
            print('Hello World!')
        x = linalg_lib.LinearOperatorFullMatrix([[1.0, 2.0], [2.0, 1.0]], is_self_adjoint=True)
        self.assertTrue(linear_operator_util.is_adjoint_pair(x, x))

    def test_pair_of_non_self_adjoint_operator_returns_false(self):
        if False:
            return 10
        x = linalg_lib.LinearOperatorFullMatrix([[1.0, 2.0], [3.0, 4.0]], is_self_adjoint=False)
        y = linalg_lib.LinearOperatorFullMatrix([[10.0, 20.0], [3.0, 4.0]], is_self_adjoint=False)
        self.assertFalse(linear_operator_util.is_adjoint_pair(x, y))

class IsAATFormTest(test.TestCase):

    def test_empty_operators_raises(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaisesRegex(ValueError, 'empty operators'):
            linear_operator_util.is_aat_form(operators=[])

    def test_odd_length_returns_false(self):
        if False:
            print('Hello World!')
        x = linalg_lib.LinearOperatorFullMatrix([[1.0, 2.0], [2.0, 1]], is_self_adjoint=True)
        self.assertFalse(linear_operator_util.is_aat_form([x]))
        self.assertFalse(linear_operator_util.is_aat_form([x, x, x.H]))

    def test_length_2_aat_form_with_sa_x(self):
        if False:
            return 10
        x = linalg_lib.LinearOperatorFullMatrix([[1.0, 2.0], [2.0, 1]], is_self_adjoint=True)
        self.assertTrue(linear_operator_util.is_aat_form([x, x.H]))

    def test_length_2_aat_form_with_non_sa_x(self):
        if False:
            while True:
                i = 10
        x = linalg_lib.LinearOperatorFullMatrix([[1.0, 5.0], [2.0, 1]], is_self_adjoint=False)
        self.assertTrue(linear_operator_util.is_aat_form([x, x.H]))

    def test_length_4_aat_form(self):
        if False:
            while True:
                i = 10
        x = linalg_lib.LinearOperatorFullMatrix([[1.0, 2.0], [5.0, 1]], is_self_adjoint=False)
        y = linalg_lib.LinearOperatorFullMatrix([[10.0, 2.0], [3.0, 10]], is_self_adjoint=False)
        self.assertTrue(linear_operator_util.is_aat_form([x, y, y.H, x.H]))

class DummyOperatorWithHint(object):

    def __init__(self, **kwargs):
        if False:
            i = 10
            return i + 15
        self.__dict__.update(kwargs)

class UseOperatorOrProvidedHintUnlessContradictingTest(test.TestCase, parameterized.TestCase):

    @parameterized.named_parameters(('none_none', None, None, None), ('none_true', None, True, True), ('true_none', True, None, True), ('true_true', True, True, True), ('none_false', None, False, False), ('false_none', False, None, False), ('false_false', False, False, False))
    def test_computes_an_or_if_non_contradicting(self, operator_hint_value, provided_hint_value, expected_result):
        if False:
            while True:
                i = 10
        self.assertEqual(expected_result, linear_operator_util.use_operator_or_provided_hint_unless_contradicting(operator=DummyOperatorWithHint(my_hint=operator_hint_value), hint_attr_name='my_hint', provided_hint_value=provided_hint_value, message='should not be needed here'))

    @parameterized.named_parameters(('true_false', True, False), ('false_true', False, True))
    def test_raises_if_contradicting(self, operator_hint_value, provided_hint_value):
        if False:
            while True:
                i = 10
        with self.assertRaisesRegex(ValueError, 'my error message'):
            linear_operator_util.use_operator_or_provided_hint_unless_contradicting(operator=DummyOperatorWithHint(my_hint=operator_hint_value), hint_attr_name='my_hint', provided_hint_value=provided_hint_value, message='my error message')

class BlockwiseTest(test.TestCase, parameterized.TestCase):

    @parameterized.named_parameters(('split_dim_1', [3, 3, 4], -1), ('split_dim_2', [2, 5], -2))
    def test_blockwise_input(self, op_dimension_values, split_dim):
        if False:
            while True:
                i = 10
        op_dimensions = [tensor_shape.Dimension(v) for v in op_dimension_values]
        unknown_op_dimensions = [tensor_shape.Dimension(None) for _ in op_dimension_values]
        batch_shape = [2, 1]
        arg_dim = 5
        if split_dim == -1:
            blockwise_arrays = [np.zeros(batch_shape + [arg_dim, d]) for d in op_dimension_values]
        else:
            blockwise_arrays = [np.zeros(batch_shape + [d, arg_dim]) for d in op_dimension_values]
        blockwise_list = [block.tolist() for block in blockwise_arrays]
        blockwise_tensors = [ops.convert_to_tensor(block) for block in blockwise_arrays]
        blockwise_placeholders = [array_ops.placeholder_with_default(block, shape=None) for block in blockwise_arrays]
        for op_dims in [op_dimensions, unknown_op_dimensions]:
            for blockwise_inputs in [blockwise_arrays, blockwise_list, blockwise_tensors, blockwise_placeholders]:
                self.assertTrue(linear_operator_util.arg_is_blockwise(op_dims, blockwise_inputs, split_dim))

    def test_non_blockwise_input(self):
        if False:
            print('Hello World!')
        x = np.zeros((2, 3, 4, 6))
        x_tensor = ops.convert_to_tensor(x)
        x_placeholder = array_ops.placeholder_with_default(x, shape=None)
        x_list = x.tolist()
        op_dimension_values = [2, 1, 3]
        op_dimensions = [tensor_shape.Dimension(d) for d in op_dimension_values]
        for inputs in [x, x_tensor, x_placeholder, x_list]:
            self.assertFalse(linear_operator_util.arg_is_blockwise(op_dimensions, inputs, -1))
        unknown_op_dimensions = [tensor_shape.Dimension(None) for _ in op_dimension_values]
        for inputs in [x, x_tensor, x_placeholder, x_list]:
            self.assertFalse(linear_operator_util.arg_is_blockwise(unknown_op_dimensions, inputs, -1))

    def test_ambiguous_input_raises(self):
        if False:
            print('Hello World!')
        x = np.zeros((3, 4, 2)).tolist()
        op_dimensions = [tensor_shape.Dimension(None) for _ in range(3)]
        with self.assertRaisesRegex(ValueError, 'structure is ambiguous'):
            linear_operator_util.arg_is_blockwise(op_dimensions, x, -2)

    def test_mismatched_input_raises(self):
        if False:
            return 10
        x = np.zeros((2, 3, 4, 6)).tolist()
        op_dimension_values = [4, 3]
        op_dimensions = [tensor_shape.Dimension(v) for v in op_dimension_values]
        with self.assertRaisesRegex(ValueError, 'dimension does not match'):
            linear_operator_util.arg_is_blockwise(op_dimensions, x, -1)
if __name__ == '__main__':
    test.main()