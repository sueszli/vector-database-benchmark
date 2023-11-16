import contextlib
import numpy as np
import scipy.linalg
from tensorflow.python.eager import backprop
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.ops.linalg import linalg as linalg_lib
from tensorflow.python.ops.linalg import linear_operator_test_util
from tensorflow.python.ops.linalg import linear_operator_toeplitz
from tensorflow.python.platform import test
linalg = linalg_lib
_to_complex = linear_operator_toeplitz._to_complex

@test_util.run_all_in_graph_and_eager_modes
class LinearOperatorToeplitzTest(linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
    """Most tests done in the base class LinearOperatorDerivedClassTest."""

    @contextlib.contextmanager
    def _constrain_devices_and_set_default(self, sess, use_gpu, force_gpu):
        if False:
            while True:
                i = 10
        'We overwrite the FFT operation mapping for testing.'
        with test.TestCase._constrain_devices_and_set_default(self, sess, use_gpu, force_gpu) as sess:
            yield sess

    def setUp(self):
        if False:
            print('Hello World!')
        self._atol[dtypes.float32] = 0.0001
        self._rtol[dtypes.float32] = 0.0001
        self._atol[dtypes.float64] = 1e-09
        self._rtol[dtypes.float64] = 1e-09
        self._atol[dtypes.complex64] = 0.0001
        self._rtol[dtypes.complex64] = 0.0001
        self._atol[dtypes.complex128] = 1e-09
        self._rtol[dtypes.complex128] = 1e-09

    @staticmethod
    def skip_these_tests():
        if False:
            for i in range(10):
                print('nop')
        return ['cholesky', 'cond', 'inverse', 'solve', 'solve_with_broadcast']

    @staticmethod
    def operator_shapes_infos():
        if False:
            for i in range(10):
                print('nop')
        shape_info = linear_operator_test_util.OperatorShapesInfo
        return [shape_info((1, 1)), shape_info((1, 6, 6)), shape_info((3, 4, 4)), shape_info((2, 1, 3, 3))]

    def operator_and_matrix(self, build_info, dtype, use_placeholder, ensure_self_adjoint_and_pd=False):
        if False:
            for i in range(10):
                print('nop')
        shape = list(build_info.shape)
        row = np.random.uniform(low=1.0, high=5.0, size=shape[:-1])
        col = np.random.uniform(low=1.0, high=5.0, size=shape[:-1])
        row[..., 0] = col[..., 0]
        if ensure_self_adjoint_and_pd:
            row = np.linspace(start=10.0, stop=1.0, num=shape[-1])
            row = col
        lin_op_row = math_ops.cast(row, dtype=dtype)
        lin_op_col = math_ops.cast(col, dtype=dtype)
        if use_placeholder:
            lin_op_row = array_ops.placeholder_with_default(lin_op_row, shape=None)
            lin_op_col = array_ops.placeholder_with_default(lin_op_col, shape=None)
        operator = linear_operator_toeplitz.LinearOperatorToeplitz(row=lin_op_row, col=lin_op_col, is_self_adjoint=True if ensure_self_adjoint_and_pd else None, is_positive_definite=True if ensure_self_adjoint_and_pd else None)
        flattened_row = np.reshape(row, (-1, shape[-1]))
        flattened_col = np.reshape(col, (-1, shape[-1]))
        flattened_toeplitz = np.zeros([flattened_row.shape[0], shape[-1], shape[-1]])
        for i in range(flattened_row.shape[0]):
            flattened_toeplitz[i] = scipy.linalg.toeplitz(flattened_col[i], flattened_row[i])
        matrix = np.reshape(flattened_toeplitz, shape)
        matrix = math_ops.cast(matrix, dtype=dtype)
        return (operator, matrix)

    def test_scalar_row_col_raises(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaisesRegex(ValueError, 'must have at least 1 dimension'):
            linear_operator_toeplitz.LinearOperatorToeplitz(1.0, 1.0)
        with self.assertRaisesRegex(ValueError, 'must have at least 1 dimension'):
            linear_operator_toeplitz.LinearOperatorToeplitz([1.0], 1.0)
        with self.assertRaisesRegex(ValueError, 'must have at least 1 dimension'):
            linear_operator_toeplitz.LinearOperatorToeplitz(1.0, [1.0])

    def test_tape_safe(self):
        if False:
            print('Hello World!')
        col = variables_module.Variable([1.0])
        row = variables_module.Variable([1.0])
        operator = linear_operator_toeplitz.LinearOperatorToeplitz(col, row, is_self_adjoint=True, is_positive_definite=True)
        self.check_tape_safe(operator, skip_options=[linear_operator_test_util.CheckTapeSafeSkipOptions.DIAG_PART, linear_operator_test_util.CheckTapeSafeSkipOptions.TRACE])
        with backprop.GradientTape() as tape:
            self.assertIsNotNone(tape.gradient(operator.diag_part(), col))
        with backprop.GradientTape() as tape:
            self.assertIsNotNone(tape.gradient(operator.trace(), col))

    def test_convert_variables_to_tensors(self):
        if False:
            print('Hello World!')
        col = variables_module.Variable([1.0])
        row = variables_module.Variable([1.0])
        operator = linear_operator_toeplitz.LinearOperatorToeplitz(col, row, is_self_adjoint=True, is_positive_definite=True)
        with self.cached_session() as sess:
            sess.run([x.initializer for x in operator.variables])
            self.check_convert_variables_to_tensors(operator)
if __name__ == '__main__':
    linear_operator_test_util.add_tests(LinearOperatorToeplitzTest)
    test.main()