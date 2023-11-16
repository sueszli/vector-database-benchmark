from tensorflow.python.framework import config
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.ops.linalg import linalg as linalg_lib
from tensorflow.python.ops.linalg import linear_operator_householder as householder
from tensorflow.python.ops.linalg import linear_operator_test_util
from tensorflow.python.platform import test
linalg = linalg_lib
CheckTapeSafeSkipOptions = linear_operator_test_util.CheckTapeSafeSkipOptions

@test_util.run_all_in_graph_and_eager_modes
class LinearOperatorHouseholderTest(linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
    """Most tests done in the base class LinearOperatorDerivedClassTest."""

    def tearDown(self):
        if False:
            print('Hello World!')
        config.enable_tensor_float_32_execution(self.tf32_keep_)

    def setUp(self):
        if False:
            while True:
                i = 10
        self.tf32_keep_ = config.tensor_float_32_execution_enabled()
        config.enable_tensor_float_32_execution(False)

    @staticmethod
    def operator_shapes_infos():
        if False:
            print('Hello World!')
        shape_info = linear_operator_test_util.OperatorShapesInfo
        return [shape_info((1, 1)), shape_info((1, 3, 3)), shape_info((3, 4, 4)), shape_info((2, 1, 4, 4))]

    @staticmethod
    def skip_these_tests():
        if False:
            print('Hello World!')
        return ['cholesky']

    def operator_and_matrix(self, build_info, dtype, use_placeholder, ensure_self_adjoint_and_pd=False):
        if False:
            for i in range(10):
                print('nop')
        shape = list(build_info.shape)
        reflection_axis = linear_operator_test_util.random_sign_uniform(shape[:-1], minval=1.0, maxval=2.0, dtype=dtype)
        reflection_axis = reflection_axis / linalg_ops.norm(reflection_axis, axis=-1, keepdims=True)
        lin_op_reflection_axis = reflection_axis
        if use_placeholder:
            lin_op_reflection_axis = array_ops.placeholder_with_default(reflection_axis, shape=None)
        operator = householder.LinearOperatorHouseholder(lin_op_reflection_axis)
        mat = reflection_axis[..., array_ops.newaxis]
        matrix = -2 * math_ops.matmul(mat, mat, adjoint_b=True)
        matrix = array_ops.matrix_set_diag(matrix, 1.0 + array_ops.matrix_diag_part(matrix))
        return (operator, matrix)

    def test_scalar_reflection_axis_raises(self):
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(ValueError, 'must have at least 1 dimension'):
            householder.LinearOperatorHouseholder(1.0)

    def test_householder_adjoint_type(self):
        if False:
            while True:
                i = 10
        reflection_axis = [1.0, 3.0, 5.0, 8.0]
        operator = householder.LinearOperatorHouseholder(reflection_axis)
        self.assertIsInstance(operator.adjoint(), householder.LinearOperatorHouseholder)

    def test_householder_inverse_type(self):
        if False:
            for i in range(10):
                print('nop')
        reflection_axis = [1.0, 3.0, 5.0, 8.0]
        operator = householder.LinearOperatorHouseholder(reflection_axis)
        self.assertIsInstance(operator.inverse(), householder.LinearOperatorHouseholder)

    def test_tape_safe(self):
        if False:
            for i in range(10):
                print('nop')
        reflection_axis = variables_module.Variable([1.0, 3.0, 5.0, 8.0])
        operator = householder.LinearOperatorHouseholder(reflection_axis)
        self.check_tape_safe(operator, skip_options=[CheckTapeSafeSkipOptions.DETERMINANT, CheckTapeSafeSkipOptions.LOG_ABS_DETERMINANT, CheckTapeSafeSkipOptions.TRACE])

    def test_convert_variables_to_tensors(self):
        if False:
            i = 10
            return i + 15
        reflection_axis = variables_module.Variable([1.0, 3.0, 5.0, 8.0])
        operator = householder.LinearOperatorHouseholder(reflection_axis)
        with self.cached_session() as sess:
            sess.run([reflection_axis.initializer])
            self.check_convert_variables_to_tensors(operator)
if __name__ == '__main__':
    linear_operator_test_util.add_tests(LinearOperatorHouseholderTest)
    test.main()