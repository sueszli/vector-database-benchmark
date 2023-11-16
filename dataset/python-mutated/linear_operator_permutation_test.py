from tensorflow.python.framework import constant_op
from tensorflow.python.framework import config
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.linalg import linalg as linalg_lib
from tensorflow.python.ops.linalg import linear_operator_permutation as permutation
from tensorflow.python.ops.linalg import linear_operator_test_util
from tensorflow.python.platform import test
linalg = linalg_lib
CheckTapeSafeSkipOptions = linear_operator_test_util.CheckTapeSafeSkipOptions

@test_util.run_all_in_graph_and_eager_modes
class LinearOperatorPermutationTest(linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
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
    def operator_shapes_infos():
        if False:
            i = 10
            return i + 15
        shape_info = linear_operator_test_util.OperatorShapesInfo
        return [shape_info((1, 1)), shape_info((1, 3, 3)), shape_info((3, 4, 4)), shape_info((2, 1, 4, 4))]

    @staticmethod
    def skip_these_tests():
        if False:
            print('Hello World!')
        return ['cholesky', 'eigvalsh']

    def operator_and_matrix(self, build_info, dtype, use_placeholder, ensure_self_adjoint_and_pd=False):
        if False:
            i = 10
            return i + 15
        shape = list(build_info.shape)
        perm = math_ops.range(0, shape[-1])
        perm = array_ops.broadcast_to(perm, shape[:-1])
        perm = random_ops.random_shuffle(perm)
        if use_placeholder:
            perm = array_ops.placeholder_with_default(perm, shape=None)
        operator = permutation.LinearOperatorPermutation(perm, dtype=dtype)
        matrix = math_ops.cast(math_ops.equal(math_ops.range(0, shape[-1]), perm[..., array_ops.newaxis]), dtype)
        return (operator, matrix)

    def test_permutation_raises(self):
        if False:
            i = 10
            return i + 15
        perm = constant_op.constant(0, dtype=dtypes.int32)
        with self.assertRaisesRegex(ValueError, 'must have at least 1 dimension'):
            permutation.LinearOperatorPermutation(perm)
        perm = [0.0, 1.0, 2.0]
        with self.assertRaisesRegex(TypeError, 'must be integer dtype'):
            permutation.LinearOperatorPermutation(perm)
        perm = [-1, 2, 3]
        with self.assertRaisesRegex(ValueError, 'must be a vector of unique integers'):
            permutation.LinearOperatorPermutation(perm)

    def test_to_dense_4x4(self):
        if False:
            print('Hello World!')
        perm = [0, 1, 2, 3]
        self.assertAllClose(permutation.LinearOperatorPermutation(perm).to_dense(), linalg_ops.eye(4))
        perm = [1, 0, 3, 2]
        self.assertAllClose(permutation.LinearOperatorPermutation(perm).to_dense(), [[0.0, 1, 0, 0], [1.0, 0, 0, 0], [0.0, 0, 0, 1], [0.0, 0, 1, 0]])
        perm = [3, 2, 0, 1]
        self.assertAllClose(permutation.LinearOperatorPermutation(perm).to_dense(), [[0.0, 0, 0, 1], [0.0, 0, 1, 0], [1.0, 0, 0, 0], [0.0, 1, 0, 0]])
if __name__ == '__main__':
    linear_operator_test_util.add_tests(LinearOperatorPermutationTest)
    test.main()