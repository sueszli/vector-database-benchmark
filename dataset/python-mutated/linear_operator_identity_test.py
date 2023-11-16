import numpy as np
from tensorflow.python.framework import config
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.ops.linalg import linalg as linalg_lib
from tensorflow.python.ops.linalg import linear_operator_test_util
from tensorflow.python.platform import test
rng = np.random.RandomState(2016)

@test_util.run_all_in_graph_and_eager_modes
class LinearOperatorIdentityTest(linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
    """Most tests done in the base class LinearOperatorDerivedClassTest."""

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        config.enable_tensor_float_32_execution(self.tf32_keep_)

    def setUp(self):
        if False:
            print('Hello World!')
        self.tf32_keep_ = config.tensor_float_32_execution_enabled()
        config.enable_tensor_float_32_execution(False)

    @staticmethod
    def dtypes_to_test():
        if False:
            for i in range(10):
                print('nop')
        return [dtypes.float32, dtypes.float64, dtypes.complex64, dtypes.complex128]

    @staticmethod
    def optional_tests():
        if False:
            print('Hello World!')
        'List of optional test names to run.'
        return ['operator_matmul_with_same_type', 'operator_solve_with_same_type']

    def operator_and_matrix(self, build_info, dtype, use_placeholder, ensure_self_adjoint_and_pd=False):
        if False:
            print('Hello World!')
        del ensure_self_adjoint_and_pd
        shape = list(build_info.shape)
        assert shape[-1] == shape[-2]
        batch_shape = shape[:-2]
        num_rows = shape[-1]
        operator = linalg_lib.LinearOperatorIdentity(num_rows, batch_shape=batch_shape, dtype=dtype)
        mat = linalg_ops.eye(num_rows, batch_shape=batch_shape, dtype=dtype)
        return (operator, mat)

    def test_assert_positive_definite(self):
        if False:
            i = 10
            return i + 15
        with self.cached_session():
            operator = linalg_lib.LinearOperatorIdentity(num_rows=2)
            self.evaluate(operator.assert_positive_definite())

    def test_assert_non_singular(self):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session():
            operator = linalg_lib.LinearOperatorIdentity(num_rows=2)
            self.evaluate(operator.assert_non_singular())

    def test_assert_self_adjoint(self):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session():
            operator = linalg_lib.LinearOperatorIdentity(num_rows=2)
            self.evaluate(operator.assert_self_adjoint())

    def test_float16_matmul(self):
        if False:
            i = 10
            return i + 15
        with self.cached_session():
            operator = linalg_lib.LinearOperatorIdentity(num_rows=2, dtype=dtypes.float16)
            x = rng.randn(2, 3).astype(np.float16)
            y = operator.matmul(x)
            self.assertAllClose(x, self.evaluate(y))

    def test_non_scalar_num_rows_raises_static(self):
        if False:
            while True:
                i = 10
        with self.assertRaisesRegex(ValueError, 'must be a 0-D Tensor'):
            linalg_lib.LinearOperatorIdentity(num_rows=[2])

    def test_non_integer_num_rows_raises_static(self):
        if False:
            return 10
        with self.assertRaisesRegex(TypeError, 'must be integer'):
            linalg_lib.LinearOperatorIdentity(num_rows=2.0)

    def test_negative_num_rows_raises_static(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(ValueError, 'must be non-negative'):
            linalg_lib.LinearOperatorIdentity(num_rows=-2)

    def test_non_1d_batch_shape_raises_static(self):
        if False:
            return 10
        with self.assertRaisesRegex(ValueError, 'must be a 1-D'):
            linalg_lib.LinearOperatorIdentity(num_rows=2, batch_shape=2)

    def test_non_integer_batch_shape_raises_static(self):
        if False:
            return 10
        with self.assertRaisesRegex(TypeError, 'must be integer'):
            linalg_lib.LinearOperatorIdentity(num_rows=2, batch_shape=[2.0])

    def test_negative_batch_shape_raises_static(self):
        if False:
            return 10
        with self.assertRaisesRegex(ValueError, 'must be non-negative'):
            linalg_lib.LinearOperatorIdentity(num_rows=2, batch_shape=[-2])

    def test_non_scalar_num_rows_raises_dynamic(self):
        if False:
            while True:
                i = 10
        with self.cached_session():
            num_rows = array_ops.placeholder_with_default([2], shape=None)
            with self.assertRaisesError('must be a 0-D Tensor'):
                operator = linalg_lib.LinearOperatorIdentity(num_rows, assert_proper_shapes=True)
                self.evaluate(operator.to_dense())

    def test_negative_num_rows_raises_dynamic(self):
        if False:
            return 10
        with self.cached_session():
            num_rows = array_ops.placeholder_with_default(-2, shape=None)
            with self.assertRaisesError('must be non-negative'):
                operator = linalg_lib.LinearOperatorIdentity(num_rows, assert_proper_shapes=True)
                self.evaluate(operator.to_dense())

    def test_non_1d_batch_shape_raises_dynamic(self):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session():
            batch_shape = array_ops.placeholder_with_default(2, shape=None)
            with self.assertRaisesError('must be a 1-D'):
                operator = linalg_lib.LinearOperatorIdentity(num_rows=2, batch_shape=batch_shape, assert_proper_shapes=True)
                self.evaluate(operator.to_dense())

    def test_negative_batch_shape_raises_dynamic(self):
        if False:
            while True:
                i = 10
        with self.cached_session():
            batch_shape = array_ops.placeholder_with_default([-2], shape=None)
            with self.assertRaisesError('must be non-negative'):
                operator = linalg_lib.LinearOperatorIdentity(num_rows=2, batch_shape=batch_shape, assert_proper_shapes=True)
                self.evaluate(operator.to_dense())

    def test_wrong_matrix_dimensions_raises_static(self):
        if False:
            for i in range(10):
                print('nop')
        operator = linalg_lib.LinearOperatorIdentity(num_rows=2)
        x = rng.randn(3, 3).astype(np.float32)
        with self.assertRaisesRegex(ValueError, 'Dimensions.*not compatible'):
            operator.matmul(x)

    def test_wrong_matrix_dimensions_raises_dynamic(self):
        if False:
            print('Hello World!')
        num_rows = array_ops.placeholder_with_default(2, shape=None)
        x = array_ops.placeholder_with_default(rng.rand(3, 3).astype(np.float32), shape=None)
        with self.cached_session():
            with self.assertRaisesError('Dimensions.*not.compatible'):
                operator = linalg_lib.LinearOperatorIdentity(num_rows, assert_proper_shapes=True)
                self.evaluate(operator.matmul(x))

    def test_default_batch_shape_broadcasts_with_everything_static(self):
        if False:
            return 10
        with self.cached_session() as sess:
            x = random_ops.random_normal(shape=(1, 2, 3, 4))
            operator = linalg_lib.LinearOperatorIdentity(num_rows=3, dtype=x.dtype)
            operator_matmul = operator.matmul(x)
            expected = x
            self.assertAllEqual(operator_matmul.shape, expected.shape)
            self.assertAllClose(*self.evaluate([operator_matmul, expected]))

    def test_default_batch_shape_broadcasts_with_everything_dynamic(self):
        if False:
            return 10
        with self.cached_session():
            x = array_ops.placeholder_with_default(rng.randn(1, 2, 3, 4), shape=None)
            operator = linalg_lib.LinearOperatorIdentity(num_rows=3, dtype=x.dtype)
            operator_matmul = operator.matmul(x)
            expected = x
            self.assertAllClose(*self.evaluate([operator_matmul, expected]))

    def test_broadcast_matmul_static_shapes(self):
        if False:
            print('Hello World!')
        with self.cached_session() as sess:
            x = random_ops.random_normal(shape=(1, 2, 3, 4))
            operator = linalg_lib.LinearOperatorIdentity(num_rows=3, batch_shape=(2, 1), dtype=x.dtype)
            zeros = array_ops.zeros(shape=(2, 2, 3, 4), dtype=x.dtype)
            expected = x + zeros
            operator_matmul = operator.matmul(x)
            self.assertAllEqual(operator_matmul.shape, expected.shape)
            self.assertAllClose(*self.evaluate([operator_matmul, expected]))

    def test_broadcast_matmul_dynamic_shapes(self):
        if False:
            return 10
        with self.cached_session():
            x = array_ops.placeholder_with_default(rng.rand(1, 2, 3, 4), shape=None)
            num_rows = array_ops.placeholder_with_default(3, shape=None)
            batch_shape = array_ops.placeholder_with_default((2, 1), shape=None)
            operator = linalg_lib.LinearOperatorIdentity(num_rows, batch_shape=batch_shape, dtype=dtypes.float64)
            zeros = array_ops.zeros(shape=(2, 2, 3, 4), dtype=x.dtype)
            expected = x + zeros
            operator_matmul = operator.matmul(x)
            self.assertAllClose(*self.evaluate([operator_matmul, expected]))

    def test_is_x_flags(self):
        if False:
            while True:
                i = 10
        operator = linalg_lib.LinearOperatorIdentity(num_rows=2)
        self.assertTrue(operator.is_positive_definite)
        self.assertTrue(operator.is_non_singular)
        self.assertTrue(operator.is_self_adjoint)
        with self.assertRaisesRegex(ValueError, 'is always non-singular'):
            operator = linalg_lib.LinearOperatorIdentity(num_rows=2, is_non_singular=None)

    def test_identity_adjoint_type(self):
        if False:
            print('Hello World!')
        operator = linalg_lib.LinearOperatorIdentity(num_rows=2, is_non_singular=True)
        self.assertIsInstance(operator.adjoint(), linalg_lib.LinearOperatorIdentity)

    def test_identity_cholesky_type(self):
        if False:
            for i in range(10):
                print('nop')
        operator = linalg_lib.LinearOperatorIdentity(num_rows=2, is_positive_definite=True, is_self_adjoint=True)
        self.assertIsInstance(operator.cholesky(), linalg_lib.LinearOperatorIdentity)

    def test_identity_inverse_type(self):
        if False:
            while True:
                i = 10
        operator = linalg_lib.LinearOperatorIdentity(num_rows=2, is_non_singular=True)
        self.assertIsInstance(operator.inverse(), linalg_lib.LinearOperatorIdentity)

    def test_ref_type_shape_args_raises(self):
        if False:
            while True:
                i = 10
        with self.assertRaisesRegex(TypeError, 'num_rows.*reference'):
            linalg_lib.LinearOperatorIdentity(num_rows=variables_module.Variable(2))
        with self.assertRaisesRegex(TypeError, 'batch_shape.*reference'):
            linalg_lib.LinearOperatorIdentity(num_rows=2, batch_shape=variables_module.Variable([3]))

@test_util.run_all_in_graph_and_eager_modes
class LinearOperatorScaledIdentityTest(linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
    """Most tests done in the base class LinearOperatorDerivedClassTest."""

    def tearDown(self):
        if False:
            return 10
        config.enable_tensor_float_32_execution(self.tf32_keep_)

    def setUp(self):
        if False:
            return 10
        self.tf32_keep_ = config.tensor_float_32_execution_enabled()
        config.enable_tensor_float_32_execution(False)

    @staticmethod
    def dtypes_to_test():
        if False:
            return 10
        return [dtypes.float32, dtypes.float64, dtypes.complex64, dtypes.complex128]

    @staticmethod
    def optional_tests():
        if False:
            for i in range(10):
                print('nop')
        'List of optional test names to run.'
        return ['operator_matmul_with_same_type', 'operator_solve_with_same_type']

    def operator_and_matrix(self, build_info, dtype, use_placeholder, ensure_self_adjoint_and_pd=False):
        if False:
            while True:
                i = 10
        shape = list(build_info.shape)
        assert shape[-1] == shape[-2]
        batch_shape = shape[:-2]
        num_rows = shape[-1]
        multiplier = linear_operator_test_util.random_sign_uniform(shape=batch_shape, minval=1.0, maxval=2.0, dtype=dtype)
        if ensure_self_adjoint_and_pd:
            multiplier = math_ops.cast(math_ops.abs(multiplier), dtype=dtype)
        lin_op_multiplier = multiplier
        if use_placeholder:
            lin_op_multiplier = array_ops.placeholder_with_default(multiplier, shape=None)
        operator = linalg_lib.LinearOperatorScaledIdentity(num_rows, lin_op_multiplier, is_self_adjoint=True if ensure_self_adjoint_and_pd else None, is_positive_definite=True if ensure_self_adjoint_and_pd else None)
        multiplier_matrix = array_ops.expand_dims(array_ops.expand_dims(multiplier, -1), -1)
        matrix = multiplier_matrix * linalg_ops.eye(num_rows, batch_shape=batch_shape, dtype=dtype)
        return (operator, matrix)

    def test_assert_positive_definite_does_not_raise_when_positive(self):
        if False:
            print('Hello World!')
        with self.cached_session():
            operator = linalg_lib.LinearOperatorScaledIdentity(num_rows=2, multiplier=1.0)
            self.evaluate(operator.assert_positive_definite())

    def test_assert_positive_definite_raises_when_negative(self):
        if False:
            return 10
        with self.cached_session():
            operator = linalg_lib.LinearOperatorScaledIdentity(num_rows=2, multiplier=-1.0)
            with self.assertRaisesOpError('not positive definite'):
                self.evaluate(operator.assert_positive_definite())

    def test_assert_non_singular_does_not_raise_when_non_singular(self):
        if False:
            return 10
        with self.cached_session():
            operator = linalg_lib.LinearOperatorScaledIdentity(num_rows=2, multiplier=[1.0, 2.0, 3.0])
            self.evaluate(operator.assert_non_singular())

    def test_assert_non_singular_raises_when_singular(self):
        if False:
            return 10
        with self.cached_session():
            operator = linalg_lib.LinearOperatorScaledIdentity(num_rows=2, multiplier=[1.0, 2.0, 0.0])
            with self.assertRaisesOpError('was singular'):
                self.evaluate(operator.assert_non_singular())

    def test_assert_self_adjoint_does_not_raise_when_self_adjoint(self):
        if False:
            return 10
        with self.cached_session():
            operator = linalg_lib.LinearOperatorScaledIdentity(num_rows=2, multiplier=[1.0 + 0j])
            self.evaluate(operator.assert_self_adjoint())

    def test_assert_self_adjoint_raises_when_not_self_adjoint(self):
        if False:
            while True:
                i = 10
        with self.cached_session():
            operator = linalg_lib.LinearOperatorScaledIdentity(num_rows=2, multiplier=[1.0 + 1j])
            with self.assertRaisesOpError('not self-adjoint'):
                self.evaluate(operator.assert_self_adjoint())

    def test_float16_matmul(self):
        if False:
            print('Hello World!')
        with self.cached_session():
            multiplier = rng.rand(3).astype(np.float16)
            operator = linalg_lib.LinearOperatorScaledIdentity(num_rows=2, multiplier=multiplier)
            x = rng.randn(2, 3).astype(np.float16)
            y = operator.matmul(x)
            self.assertAllClose(multiplier[..., None, None] * x, self.evaluate(y))

    def test_non_scalar_num_rows_raises_static(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(ValueError, 'must be a 0-D Tensor'):
            linalg_lib.LinearOperatorScaledIdentity(num_rows=[2], multiplier=123.0)

    def test_wrong_matrix_dimensions_raises_static(self):
        if False:
            while True:
                i = 10
        operator = linalg_lib.LinearOperatorScaledIdentity(num_rows=2, multiplier=2.2)
        x = rng.randn(3, 3).astype(np.float32)
        with self.assertRaisesRegex(ValueError, 'Dimensions.*not compatible'):
            operator.matmul(x)

    def test_wrong_matrix_dimensions_raises_dynamic(self):
        if False:
            i = 10
            return i + 15
        num_rows = array_ops.placeholder_with_default(2, shape=None)
        x = array_ops.placeholder_with_default(rng.rand(3, 3).astype(np.float32), shape=None)
        with self.cached_session():
            with self.assertRaisesError('Dimensions.*not.compatible'):
                operator = linalg_lib.LinearOperatorScaledIdentity(num_rows, multiplier=[1.0, 2], assert_proper_shapes=True)
                self.evaluate(operator.matmul(x))

    def test_broadcast_matmul_and_solve(self):
        if False:
            print('Hello World!')
        with self.cached_session() as sess:
            x = random_ops.random_normal(shape=(1, 2, 3, 4))
            operator = linalg_lib.LinearOperatorScaledIdentity(num_rows=3, multiplier=2.2 * array_ops.ones((2, 1)))
            zeros = array_ops.zeros(shape=(2, 2, 3, 4), dtype=x.dtype)
            expected = x * 2.2 + zeros
            operator_matmul = operator.matmul(x)
            self.assertAllEqual(operator_matmul.shape, expected.shape)
            self.assertAllClose(*self.evaluate([operator_matmul, expected]))
            expected = x / 2.2 + zeros
            operator_solve = operator.solve(x)
            self.assertAllEqual(operator_solve.shape, expected.shape)
            self.assertAllClose(*self.evaluate([operator_solve, expected]))

    def test_broadcast_matmul_and_solve_scalar_scale_multiplier(self):
        if False:
            return 10
        with self.cached_session() as sess:
            x = random_ops.random_normal(shape=(1, 2, 3, 4))
            operator = linalg_lib.LinearOperatorScaledIdentity(num_rows=3, multiplier=2.2)
            expected = x * 2.2
            operator_matmul = operator.matmul(x)
            self.assertAllEqual(operator_matmul.shape, expected.shape)
            self.assertAllClose(*self.evaluate([operator_matmul, expected]))
            expected = x / 2.2
            operator_solve = operator.solve(x)
            self.assertAllEqual(operator_solve.shape, expected.shape)
            self.assertAllClose(*self.evaluate([operator_solve, expected]))

    def test_is_x_flags(self):
        if False:
            for i in range(10):
                print('nop')
        operator = linalg_lib.LinearOperatorScaledIdentity(num_rows=2, multiplier=1.0, is_positive_definite=False, is_non_singular=True)
        self.assertFalse(operator.is_positive_definite)
        self.assertTrue(operator.is_non_singular)
        self.assertTrue(operator.is_self_adjoint)

    def test_identity_adjoint_type(self):
        if False:
            while True:
                i = 10
        operator = linalg_lib.LinearOperatorScaledIdentity(num_rows=2, multiplier=1.0, is_non_singular=True)
        self.assertIsInstance(operator.adjoint(), linalg_lib.LinearOperatorScaledIdentity)

    def test_identity_matmul(self):
        if False:
            return 10
        operator1 = linalg_lib.LinearOperatorIdentity(num_rows=2)
        operator2 = linalg_lib.LinearOperatorScaledIdentity(num_rows=2, multiplier=3.0)
        self.assertIsInstance(operator1.matmul(operator1), linalg_lib.LinearOperatorIdentity)
        self.assertIsInstance(operator1.matmul(operator1), linalg_lib.LinearOperatorIdentity)
        self.assertIsInstance(operator2.matmul(operator2), linalg_lib.LinearOperatorScaledIdentity)
        operator_matmul = operator1.matmul(operator2)
        self.assertIsInstance(operator_matmul, linalg_lib.LinearOperatorScaledIdentity)
        self.assertAllClose(3.0, self.evaluate(operator_matmul.multiplier))
        operator_matmul = operator2.matmul(operator1)
        self.assertIsInstance(operator_matmul, linalg_lib.LinearOperatorScaledIdentity)
        self.assertAllClose(3.0, self.evaluate(operator_matmul.multiplier))

    def test_identity_solve(self):
        if False:
            return 10
        operator1 = linalg_lib.LinearOperatorIdentity(num_rows=2)
        operator2 = linalg_lib.LinearOperatorScaledIdentity(num_rows=2, multiplier=3.0)
        self.assertIsInstance(operator1.solve(operator1), linalg_lib.LinearOperatorIdentity)
        self.assertIsInstance(operator2.solve(operator2), linalg_lib.LinearOperatorScaledIdentity)
        operator_solve = operator1.solve(operator2)
        self.assertIsInstance(operator_solve, linalg_lib.LinearOperatorScaledIdentity)
        self.assertAllClose(3.0, self.evaluate(operator_solve.multiplier))
        operator_solve = operator2.solve(operator1)
        self.assertIsInstance(operator_solve, linalg_lib.LinearOperatorScaledIdentity)
        self.assertAllClose(1.0 / 3.0, self.evaluate(operator_solve.multiplier))

    def test_scaled_identity_cholesky_type(self):
        if False:
            print('Hello World!')
        operator = linalg_lib.LinearOperatorScaledIdentity(num_rows=2, multiplier=3.0, is_positive_definite=True, is_self_adjoint=True)
        self.assertIsInstance(operator.cholesky(), linalg_lib.LinearOperatorScaledIdentity)

    def test_scaled_identity_inverse_type(self):
        if False:
            i = 10
            return i + 15
        operator = linalg_lib.LinearOperatorScaledIdentity(num_rows=2, multiplier=3.0, is_non_singular=True)
        self.assertIsInstance(operator.inverse(), linalg_lib.LinearOperatorScaledIdentity)

    def test_ref_type_shape_args_raises(self):
        if False:
            return 10
        with self.assertRaisesRegex(TypeError, 'num_rows.*reference'):
            linalg_lib.LinearOperatorScaledIdentity(num_rows=variables_module.Variable(2), multiplier=1.23)

    def test_tape_safe(self):
        if False:
            while True:
                i = 10
        multiplier = variables_module.Variable(1.23)
        operator = linalg_lib.LinearOperatorScaledIdentity(num_rows=2, multiplier=multiplier)
        self.check_tape_safe(operator)

    def test_convert_variables_to_tensors(self):
        if False:
            while True:
                i = 10
        multiplier = variables_module.Variable(1.23)
        operator = linalg_lib.LinearOperatorScaledIdentity(num_rows=2, multiplier=multiplier)
        with self.cached_session() as sess:
            sess.run([multiplier.initializer])
            self.check_convert_variables_to_tensors(operator)
if __name__ == '__main__':
    linear_operator_test_util.add_tests(LinearOperatorIdentityTest)
    linear_operator_test_util.add_tests(LinearOperatorScaledIdentityTest)
    test.main()