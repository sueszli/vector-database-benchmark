from tensorflow.python.framework import config
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import manip_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.ops.linalg import linalg as linalg_lib
from tensorflow.python.ops.linalg import linear_operator_test_util
from tensorflow.python.platform import test

class _LinearOperatorTriDiagBase(object):

    def build_operator_and_matrix(self, build_info, dtype, use_placeholder, ensure_self_adjoint_and_pd=False, diagonals_format='sequence'):
        if False:
            for i in range(10):
                print('nop')
        shape = list(build_info.shape)
        diag = linear_operator_test_util.random_sign_uniform(shape[:-1], minval=4.0, maxval=6.0, dtype=dtype)
        subdiag = linear_operator_test_util.random_sign_uniform(shape[:-1], minval=1.0, maxval=2.0, dtype=dtype)
        if ensure_self_adjoint_and_pd:
            diag = math_ops.cast(math_ops.abs(diag), dtype=dtype)
            superdiag = math_ops.conj(subdiag)
            superdiag = manip_ops.roll(superdiag, shift=-1, axis=-1)
        else:
            superdiag = linear_operator_test_util.random_sign_uniform(shape[:-1], minval=1.0, maxval=2.0, dtype=dtype)
        matrix_diagonals = array_ops_stack.stack([superdiag, diag, subdiag], axis=-2)
        matrix = gen_array_ops.matrix_diag_v3(matrix_diagonals, k=(-1, 1), num_rows=-1, num_cols=-1, align='LEFT_RIGHT', padding_value=0.0)
        if diagonals_format == 'sequence':
            diagonals = [superdiag, diag, subdiag]
        elif diagonals_format == 'compact':
            diagonals = array_ops_stack.stack([superdiag, diag, subdiag], axis=-2)
        elif diagonals_format == 'matrix':
            diagonals = matrix
        lin_op_diagonals = diagonals
        if use_placeholder:
            if diagonals_format == 'sequence':
                lin_op_diagonals = [array_ops.placeholder_with_default(d, shape=None) for d in lin_op_diagonals]
            else:
                lin_op_diagonals = array_ops.placeholder_with_default(lin_op_diagonals, shape=None)
        operator = linalg_lib.LinearOperatorTridiag(diagonals=lin_op_diagonals, diagonals_format=diagonals_format, is_self_adjoint=True if ensure_self_adjoint_and_pd else None, is_positive_definite=True if ensure_self_adjoint_and_pd else None)
        return (operator, matrix)

    @staticmethod
    def operator_shapes_infos():
        if False:
            for i in range(10):
                print('nop')
        shape_info = linear_operator_test_util.OperatorShapesInfo
        return [shape_info((3, 3)), shape_info((1, 6, 6)), shape_info((3, 4, 4)), shape_info((2, 1, 3, 3))]

@test_util.with_eager_op_as_function
@test_util.run_all_in_graph_and_eager_modes
class LinearOperatorTriDiagCompactTest(_LinearOperatorTriDiagBase, linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
    """Most tests done in the base class LinearOperatorDerivedClassTest."""

    def tearDown(self):
        if False:
            while True:
                i = 10
        config.enable_tensor_float_32_execution(self.tf32_keep_)

    def setUp(self):
        if False:
            return 10
        self.tf32_keep_ = config.tensor_float_32_execution_enabled()
        config.enable_tensor_float_32_execution(False)

    def operator_and_matrix(self, build_info, dtype, use_placeholder, ensure_self_adjoint_and_pd=False):
        if False:
            print('Hello World!')
        return self.build_operator_and_matrix(build_info, dtype, use_placeholder, ensure_self_adjoint_and_pd=ensure_self_adjoint_and_pd, diagonals_format='compact')

    @test_util.disable_xla('Current implementation does not yet support pivoting')
    def test_tape_safe(self):
        if False:
            return 10
        diag = variables_module.Variable([[3.0, 6.0, 2.0], [2.0, 4.0, 2.0], [5.0, 1.0, 2.0]])
        operator = linalg_lib.LinearOperatorTridiag(diag, diagonals_format='compact')
        self.check_tape_safe(operator)

    def test_convert_variables_to_tensors(self):
        if False:
            while True:
                i = 10
        diag = variables_module.Variable([[3.0, 6.0, 2.0], [2.0, 4.0, 2.0], [5.0, 1.0, 2.0]])
        operator = linalg_lib.LinearOperatorTridiag(diag, diagonals_format='compact')
        with self.cached_session() as sess:
            sess.run([diag.initializer])
            self.check_convert_variables_to_tensors(operator)

@test_util.with_eager_op_as_function
@test_util.run_all_in_graph_and_eager_modes
class LinearOperatorTriDiagSequenceTest(_LinearOperatorTriDiagBase, linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
    """Most tests done in the base class LinearOperatorDerivedClassTest."""

    def tearDown(self):
        if False:
            print('Hello World!')
        config.enable_tensor_float_32_execution(self.tf32_keep_)

    def setUp(self):
        if False:
            return 10
        self.tf32_keep_ = config.tensor_float_32_execution_enabled()
        config.enable_tensor_float_32_execution(False)

    def operator_and_matrix(self, build_info, dtype, use_placeholder, ensure_self_adjoint_and_pd=False):
        if False:
            print('Hello World!')
        return self.build_operator_and_matrix(build_info, dtype, use_placeholder, ensure_self_adjoint_and_pd=ensure_self_adjoint_and_pd, diagonals_format='sequence')

    @test_util.disable_xla('Current implementation does not yet support pivoting')
    def test_tape_safe(self):
        if False:
            return 10
        diagonals = [variables_module.Variable([3.0, 6.0, 2.0]), variables_module.Variable([2.0, 4.0, 2.0]), variables_module.Variable([5.0, 1.0, 2.0])]
        operator = linalg_lib.LinearOperatorTridiag(diagonals, diagonals_format='sequence')
        self.check_tape_safe(operator, skip_options=['diag_part', 'trace'])
        diagonals = [[3.0, 6.0, 2.0], variables_module.Variable([2.0, 4.0, 2.0]), [5.0, 1.0, 2.0]]
        operator = linalg_lib.LinearOperatorTridiag(diagonals, diagonals_format='sequence')

@test_util.with_eager_op_as_function
@test_util.run_all_in_graph_and_eager_modes
class LinearOperatorTriDiagMatrixTest(_LinearOperatorTriDiagBase, linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
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

    def operator_and_matrix(self, build_info, dtype, use_placeholder, ensure_self_adjoint_and_pd=False):
        if False:
            while True:
                i = 10
        return self.build_operator_and_matrix(build_info, dtype, use_placeholder, ensure_self_adjoint_and_pd=ensure_self_adjoint_and_pd, diagonals_format='matrix')

    @test_util.disable_xla('Current implementation does not yet support pivoting')
    def test_tape_safe(self):
        if False:
            return 10
        matrix = variables_module.Variable([[3.0, 2.0, 0.0], [1.0, 6.0, 4.0], [0.0, 2, 2]])
        operator = linalg_lib.LinearOperatorTridiag(matrix, diagonals_format='matrix')
        self.check_tape_safe(operator)
if __name__ == '__main__':
    if not test_util.is_xla_enabled():
        linear_operator_test_util.add_tests(LinearOperatorTriDiagCompactTest)
        linear_operator_test_util.add_tests(LinearOperatorTriDiagSequenceTest)
        linear_operator_test_util.add_tests(LinearOperatorTriDiagMatrixTest)
    test.main()