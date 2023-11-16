import numpy as np
from tensorflow.python.framework import config
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.ops.linalg import linalg as linalg_lib
from tensorflow.python.ops.linalg import linear_operator_test_util
from tensorflow.python.platform import test
linalg = linalg_lib
rng = np.random.RandomState(0)

class BaseLinearOperatorLowRankUpdatetest(object):
    """Base test for this type of operator."""
    _use_diag_update = None
    _is_diag_update_positive = None
    _use_v = None

    @staticmethod
    def operator_shapes_infos():
        if False:
            while True:
                i = 10
        shape_info = linear_operator_test_util.OperatorShapesInfo
        return [shape_info((0, 0)), shape_info((1, 1)), shape_info((1, 3, 3)), shape_info((3, 4, 4)), shape_info((2, 1, 4, 4))]

    def _gen_positive_diag(self, dtype, diag_shape):
        if False:
            for i in range(10):
                print('nop')
        if dtype.is_complex:
            diag = linear_operator_test_util.random_uniform(diag_shape, minval=0.0001, maxval=1.0, dtype=dtypes.float32)
            return math_ops.cast(diag, dtype=dtype)
        return linear_operator_test_util.random_uniform(diag_shape, minval=0.0001, maxval=1.0, dtype=dtype)

    def operator_and_matrix(self, shape_info, dtype, use_placeholder, ensure_self_adjoint_and_pd=False):
        if False:
            for i in range(10):
                print('nop')
        shape = list(shape_info.shape)
        diag_shape = shape[:-1]
        k = shape[-2] // 2 + 1
        u_perturbation_shape = shape[:-1] + [k]
        diag_update_shape = shape[:-2] + [k]
        base_diag = self._gen_positive_diag(dtype, diag_shape)
        lin_op_base_diag = base_diag
        u = linear_operator_test_util.random_normal_correlated_columns(u_perturbation_shape, dtype=dtype)
        lin_op_u = u
        v = linear_operator_test_util.random_normal_correlated_columns(u_perturbation_shape, dtype=dtype)
        lin_op_v = v
        if self._is_diag_update_positive or ensure_self_adjoint_and_pd:
            diag_update = self._gen_positive_diag(dtype, diag_update_shape)
        else:
            diag_update = linear_operator_test_util.random_normal(diag_update_shape, stddev=0.0001, dtype=dtype)
        lin_op_diag_update = diag_update
        if use_placeholder:
            lin_op_base_diag = array_ops.placeholder_with_default(base_diag, shape=None)
            lin_op_u = array_ops.placeholder_with_default(u, shape=None)
            lin_op_v = array_ops.placeholder_with_default(v, shape=None)
            lin_op_diag_update = array_ops.placeholder_with_default(diag_update, shape=None)
        base_operator = linalg.LinearOperatorDiag(lin_op_base_diag, is_positive_definite=True, is_self_adjoint=True)
        operator = linalg.LinearOperatorLowRankUpdate(base_operator, lin_op_u, v=lin_op_v if self._use_v else None, diag_update=lin_op_diag_update if self._use_diag_update else None, is_diag_update_positive=self._is_diag_update_positive)
        base_diag_mat = array_ops.matrix_diag(base_diag)
        diag_update_mat = array_ops.matrix_diag(diag_update)
        if self._use_v and self._use_diag_update:
            expect_use_cholesky = False
            matrix = base_diag_mat + math_ops.matmul(u, math_ops.matmul(diag_update_mat, v, adjoint_b=True))
        elif self._use_v:
            expect_use_cholesky = False
            matrix = base_diag_mat + math_ops.matmul(u, v, adjoint_b=True)
        elif self._use_diag_update:
            expect_use_cholesky = self._is_diag_update_positive
            matrix = base_diag_mat + math_ops.matmul(u, math_ops.matmul(diag_update_mat, u, adjoint_b=True))
        else:
            expect_use_cholesky = True
            matrix = base_diag_mat + math_ops.matmul(u, u, adjoint_b=True)
        if expect_use_cholesky:
            self.assertTrue(operator._use_cholesky)
        else:
            self.assertFalse(operator._use_cholesky)
        return (operator, matrix)

    def test_tape_safe(self):
        if False:
            i = 10
            return i + 15
        base_operator = linalg.LinearOperatorDiag(variables_module.Variable([1.0], name='diag'), is_positive_definite=True, is_self_adjoint=True)
        operator = linalg.LinearOperatorLowRankUpdate(base_operator, u=variables_module.Variable([[2.0]], name='u'), v=variables_module.Variable([[1.25]], name='v') if self._use_v else None, diag_update=variables_module.Variable([1.25], name='diag_update') if self._use_diag_update else None, is_diag_update_positive=self._is_diag_update_positive)
        self.check_tape_safe(operator)

    def test_convert_variables_to_tensors(self):
        if False:
            for i in range(10):
                print('nop')
        base_operator = linalg.LinearOperatorDiag(variables_module.Variable([1.0], name='diag'), is_positive_definite=True, is_self_adjoint=True)
        operator = linalg.LinearOperatorLowRankUpdate(base_operator, u=variables_module.Variable([[2.0]], name='u'), v=variables_module.Variable([[1.25]], name='v') if self._use_v else None, diag_update=variables_module.Variable([1.25], name='diag_update') if self._use_diag_update else None, is_diag_update_positive=self._is_diag_update_positive)
        with self.cached_session() as sess:
            sess.run([x.initializer for x in operator.variables])
            self.check_convert_variables_to_tensors(operator)

class LinearOperatorLowRankUpdatetestWithDiagUseCholesky(BaseLinearOperatorLowRankUpdatetest, linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
    """A = L + UDU^H, D > 0, L > 0 ==> A > 0 and we can use a Cholesky."""
    _use_diag_update = True
    _is_diag_update_positive = True
    _use_v = False

    def tearDown(self):
        if False:
            print('Hello World!')
        config.enable_tensor_float_32_execution(self.tf32_keep_)

    def setUp(self):
        if False:
            return 10
        self.tf32_keep_ = config.tensor_float_32_execution_enabled()
        config.enable_tensor_float_32_execution(False)
        self._atol[dtypes.float32] = 1e-05
        self._rtol[dtypes.float32] = 1e-05
        self._atol[dtypes.float64] = 1e-10
        self._rtol[dtypes.float64] = 1e-10
        self._rtol[dtypes.complex64] = 0.0001

class LinearOperatorLowRankUpdatetestWithDiagCannotUseCholesky(BaseLinearOperatorLowRankUpdatetest, linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
    """A = L + UDU^H, D !> 0, L > 0 ==> A !> 0 and we cannot use a Cholesky."""

    @staticmethod
    def skip_these_tests():
        if False:
            return 10
        return ['cholesky', 'eigvalsh']
    _use_diag_update = True
    _is_diag_update_positive = False
    _use_v = False

    def tearDown(self):
        if False:
            return 10
        config.enable_tensor_float_32_execution(self.tf32_keep_)

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.tf32_keep_ = config.tensor_float_32_execution_enabled()
        config.enable_tensor_float_32_execution(False)
        self._atol[dtypes.float32] = 0.0001
        self._rtol[dtypes.float32] = 0.0001
        self._atol[dtypes.float64] = 1e-09
        self._rtol[dtypes.float64] = 1e-09
        self._rtol[dtypes.complex64] = 0.0002

class LinearOperatorLowRankUpdatetestNoDiagUseCholesky(BaseLinearOperatorLowRankUpdatetest, linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
    """A = L + UU^H, L > 0 ==> A > 0 and we can use a Cholesky."""
    _use_diag_update = False
    _is_diag_update_positive = None
    _use_v = False

    def tearDown(self):
        if False:
            while True:
                i = 10
        config.enable_tensor_float_32_execution(self.tf32_keep_)

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.tf32_keep_ = config.tensor_float_32_execution_enabled()
        config.enable_tensor_float_32_execution(False)
        self._atol[dtypes.float32] = 1e-05
        self._rtol[dtypes.float32] = 1e-05
        self._atol[dtypes.float64] = 1e-10
        self._rtol[dtypes.float64] = 1e-10
        self._rtol[dtypes.complex64] = 0.0001

class LinearOperatorLowRankUpdatetestNoDiagCannotUseCholesky(BaseLinearOperatorLowRankUpdatetest, linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
    """A = L + UV^H, L > 0 ==> A is not symmetric and we cannot use a Cholesky."""

    @staticmethod
    def skip_these_tests():
        if False:
            return 10
        return ['cholesky', 'eigvalsh']
    _use_diag_update = False
    _is_diag_update_positive = None
    _use_v = True

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        config.enable_tensor_float_32_execution(self.tf32_keep_)

    def setUp(self):
        if False:
            return 10
        self.tf32_keep_ = config.tensor_float_32_execution_enabled()
        config.enable_tensor_float_32_execution(False)
        self._atol[dtypes.float32] = 0.0001
        self._rtol[dtypes.float32] = 0.0001
        self._atol[dtypes.float64] = 1e-09
        self._rtol[dtypes.float64] = 1e-09
        self._atol[dtypes.complex64] = 1e-05
        self._rtol[dtypes.complex64] = 0.0002

class LinearOperatorLowRankUpdatetestWithDiagNotSquare(BaseLinearOperatorLowRankUpdatetest, linear_operator_test_util.NonSquareLinearOperatorDerivedClassTest):
    """A = L + UDU^H, D > 0, L > 0 ==> A > 0 and we can use a Cholesky."""
    _use_diag_update = True
    _is_diag_update_positive = True
    _use_v = True

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        config.enable_tensor_float_32_execution(self.tf32_keep_)

    def setUp(self):
        if False:
            return 10
        self.tf32_keep_ = config.tensor_float_32_execution_enabled()
        config.enable_tensor_float_32_execution(False)

@test_util.run_all_without_tensor_float_32('Linear op calls matmul which uses TensorFloat-32.')
class LinearOperatorLowRankUpdateBroadcastsShape(test.TestCase):
    """Test that the operator's shape is the broadcast of arguments."""

    def test_static_shape_broadcasts_up_from_operator_to_other_args(self):
        if False:
            print('Hello World!')
        base_operator = linalg.LinearOperatorIdentity(num_rows=3)
        u = array_ops.ones(shape=[2, 3, 2])
        diag = array_ops.ones(shape=[2, 2])
        operator = linalg.LinearOperatorLowRankUpdate(base_operator, u, diag)
        self.assertAllEqual([2, 3, 3], operator.shape)
        self.assertAllEqual([2, 3, 3], self.evaluate(operator.to_dense()).shape)

    @test_util.run_deprecated_v1
    def test_dynamic_shape_broadcasts_up_from_operator_to_other_args(self):
        if False:
            print('Hello World!')
        num_rows_ph = array_ops.placeholder(dtypes.int32)
        base_operator = linalg.LinearOperatorIdentity(num_rows=num_rows_ph)
        u_shape_ph = array_ops.placeholder(dtypes.int32)
        u = array_ops.ones(shape=u_shape_ph)
        v_shape_ph = array_ops.placeholder(dtypes.int32)
        v = array_ops.ones(shape=v_shape_ph)
        diag_shape_ph = array_ops.placeholder(dtypes.int32)
        diag_update = array_ops.ones(shape=diag_shape_ph)
        operator = linalg.LinearOperatorLowRankUpdate(base_operator, u=u, diag_update=diag_update, v=v)
        feed_dict = {num_rows_ph: 3, u_shape_ph: [1, 1, 2, 3, 2], v_shape_ph: [1, 2, 1, 3, 2], diag_shape_ph: [2, 1, 1, 2]}
        with self.cached_session():
            shape_tensor = operator.shape_tensor().eval(feed_dict=feed_dict)
            self.assertAllEqual([2, 2, 2, 3, 3], shape_tensor)
            dense = operator.to_dense().eval(feed_dict=feed_dict)
            self.assertAllEqual([2, 2, 2, 3, 3], dense.shape)

    def test_u_and_v_incompatible_batch_shape_raises(self):
        if False:
            i = 10
            return i + 15
        base_operator = linalg.LinearOperatorIdentity(num_rows=3, dtype=np.float64)
        u = rng.rand(5, 3, 2)
        v = rng.rand(4, 3, 2)
        with self.assertRaisesRegex(ValueError, 'Incompatible shapes'):
            linalg.LinearOperatorLowRankUpdate(base_operator, u=u, v=v)

    def test_u_and_base_operator_incompatible_batch_shape_raises(self):
        if False:
            while True:
                i = 10
        base_operator = linalg.LinearOperatorIdentity(num_rows=3, batch_shape=[4], dtype=np.float64)
        u = rng.rand(5, 3, 2)
        with self.assertRaisesRegex(ValueError, 'Incompatible shapes'):
            linalg.LinearOperatorLowRankUpdate(base_operator, u=u)

    def test_u_and_base_operator_incompatible_domain_dimension(self):
        if False:
            for i in range(10):
                print('nop')
        base_operator = linalg.LinearOperatorIdentity(num_rows=3, dtype=np.float64)
        u = rng.rand(5, 4, 2)
        with self.assertRaisesRegex(ValueError, 'not compatible'):
            linalg.LinearOperatorLowRankUpdate(base_operator, u=u)

    def test_u_and_diag_incompatible_low_rank_raises(self):
        if False:
            return 10
        base_operator = linalg.LinearOperatorIdentity(num_rows=3, dtype=np.float64)
        u = rng.rand(5, 3, 2)
        diag = rng.rand(5, 4)
        with self.assertRaisesRegex(ValueError, 'not compatible'):
            linalg.LinearOperatorLowRankUpdate(base_operator, u=u, diag_update=diag)

    def test_diag_incompatible_batch_shape_raises(self):
        if False:
            return 10
        base_operator = linalg.LinearOperatorIdentity(num_rows=3, dtype=np.float64)
        u = rng.rand(5, 3, 2)
        diag = rng.rand(4, 2)
        with self.assertRaisesRegex(ValueError, 'Incompatible shapes'):
            linalg.LinearOperatorLowRankUpdate(base_operator, u=u, diag_update=diag)
if __name__ == '__main__':
    linear_operator_test_util.add_tests(LinearOperatorLowRankUpdatetestWithDiagUseCholesky)
    linear_operator_test_util.add_tests(LinearOperatorLowRankUpdatetestWithDiagCannotUseCholesky)
    linear_operator_test_util.add_tests(LinearOperatorLowRankUpdatetestNoDiagUseCholesky)
    linear_operator_test_util.add_tests(LinearOperatorLowRankUpdatetestNoDiagCannotUseCholesky)
    linear_operator_test_util.add_tests(LinearOperatorLowRankUpdatetestWithDiagNotSquare)
    test.main()