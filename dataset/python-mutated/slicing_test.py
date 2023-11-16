from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.linalg import linalg as linalg_lib
from tensorflow.python.ops.linalg import linear_operator_test_util
from tensorflow.python.ops.linalg import slicing
from tensorflow.python.platform import test
linalg = linalg_lib

class _MakeSlices(object):

    def __getitem__(self, slices):
        if False:
            print('Hello World!')
        return slices if isinstance(slices, tuple) else (slices,)
make_slices = _MakeSlices()

@test_util.run_all_in_graph_and_eager_modes
class SlicingTest(test.TestCase):
    """Tests for slicing LinearOperators."""

    def test_single_param_slice_withstep_broadcastdim(self):
        if False:
            while True:
                i = 10
        event_dim = 3
        sliced = slicing._slice_single_param(array_ops.zeros([1, 1, event_dim]), param_ndims_to_matrix_ndims=1, slices=make_slices[44:-52:-3, -94:], batch_shape=constant_op.constant([2, 7], dtype=dtypes.int32))
        self.assertAllEqual((1, 1, event_dim), self.evaluate(sliced).shape)

    def test_single_param_slice_stop_leadingdim(self):
        if False:
            i = 10
            return i + 15
        sliced = slicing._slice_single_param(array_ops.zeros([7, 6, 5, 4, 3]), param_ndims_to_matrix_ndims=2, slices=make_slices[:2], batch_shape=constant_op.constant([7, 6, 5], dtype=dtypes.int32))
        self.assertAllEqual((2, 6, 5, 4, 3), self.evaluate(sliced).shape)

    def test_single_param_slice_stop_trailingdim(self):
        if False:
            print('Hello World!')
        sliced = slicing._slice_single_param(array_ops.zeros([7, 6, 5, 4, 3]), param_ndims_to_matrix_ndims=2, slices=make_slices[..., :2], batch_shape=constant_op.constant([7, 6, 5]))
        self.assertAllEqual((7, 6, 2, 4, 3), self.evaluate(sliced).shape)

    def test_single_param_slice_stop_broadcastdim(self):
        if False:
            return 10
        sliced = slicing._slice_single_param(array_ops.zeros([7, 1, 5, 4, 3]), param_ndims_to_matrix_ndims=2, slices=make_slices[:, :2], batch_shape=constant_op.constant([7, 6, 5]))
        self.assertAllEqual((7, 1, 5, 4, 3), self.evaluate(sliced).shape)

    def test_single_param_slice_newaxis_leading(self):
        if False:
            while True:
                i = 10
        sliced = slicing._slice_single_param(array_ops.zeros([7, 6, 5, 4, 3]), param_ndims_to_matrix_ndims=2, slices=make_slices[:, array_ops.newaxis], batch_shape=constant_op.constant([7, 6, 5]))
        self.assertAllEqual((7, 1, 6, 5, 4, 3), self.evaluate(sliced).shape)

    def test_single_param_slice_newaxis_trailing(self):
        if False:
            while True:
                i = 10
        sliced = slicing._slice_single_param(array_ops.zeros([7, 6, 5, 4, 3]), param_ndims_to_matrix_ndims=2, slices=make_slices[..., array_ops.newaxis, :], batch_shape=constant_op.constant([7, 6, 5]))
        self.assertAllEqual((7, 6, 1, 5, 4, 3), self.evaluate(sliced).shape)

    def test_single_param_slice_start(self):
        if False:
            i = 10
            return i + 15
        sliced = slicing._slice_single_param(array_ops.zeros([7, 6, 5, 4, 3]), param_ndims_to_matrix_ndims=2, slices=make_slices[:, 2:], batch_shape=constant_op.constant([7, 6, 5]))
        self.assertAllEqual((7, 4, 5, 4, 3), self.evaluate(sliced).shape)

    def test_single_param_slice_start_broadcastdim(self):
        if False:
            while True:
                i = 10
        sliced = slicing._slice_single_param(array_ops.zeros([7, 1, 5, 4, 3]), param_ndims_to_matrix_ndims=2, slices=make_slices[:, 2:], batch_shape=constant_op.constant([7, 6, 5]))
        self.assertAllEqual((7, 1, 5, 4, 3), self.evaluate(sliced).shape)

    def test_single_param_slice_int(self):
        if False:
            print('Hello World!')
        sliced = slicing._slice_single_param(array_ops.zeros([7, 6, 5, 4, 3]), param_ndims_to_matrix_ndims=2, slices=make_slices[:, 2], batch_shape=constant_op.constant([7, 6, 5]))
        self.assertAllEqual((7, 5, 4, 3), self.evaluate(sliced).shape)

    def test_single_param_slice_int_broadcastdim(self):
        if False:
            print('Hello World!')
        sliced = slicing._slice_single_param(array_ops.zeros([7, 1, 5, 4, 3]), param_ndims_to_matrix_ndims=2, slices=make_slices[:, 2], batch_shape=constant_op.constant([7, 6, 5]))
        self.assertAllEqual((7, 5, 4, 3), self.evaluate(sliced).shape)

    def test_single_param_slice_tensor(self):
        if False:
            for i in range(10):
                print('nop')
        param = array_ops.placeholder_with_default(array_ops.zeros([7, 6, 5, 4, 3]), shape=None)
        idx = array_ops.placeholder_with_default(constant_op.constant(2, dtype=dtypes.int32), shape=[])
        sliced = slicing._slice_single_param(param, param_ndims_to_matrix_ndims=2, slices=make_slices[:, idx], batch_shape=constant_op.constant([7, 6, 5]))
        self.assertAllEqual((7, 5, 4, 3), self.evaluate(sliced).shape)

    def test_single_param_slice_tensor_broadcastdim(self):
        if False:
            print('Hello World!')
        param = array_ops.placeholder_with_default(array_ops.zeros([7, 1, 5, 4, 3]), shape=None)
        idx = array_ops.placeholder_with_default(constant_op.constant(2, dtype=dtypes.int32), shape=[])
        sliced = slicing._slice_single_param(param, param_ndims_to_matrix_ndims=2, slices=make_slices[:, idx], batch_shape=constant_op.constant([7, 6, 5]))
        self.assertAllEqual((7, 5, 4, 3), self.evaluate(sliced).shape)

    def test_single_param_slice_broadcast_batch(self):
        if False:
            return 10
        sliced = slicing._slice_single_param(array_ops.zeros([4, 3, 1]), param_ndims_to_matrix_ndims=1, slices=make_slices[..., array_ops.newaxis, 2:, array_ops.newaxis], batch_shape=constant_op.constant([7, 4, 3]))
        self.assertAllEqual(list(array_ops.zeros([1, 4, 3])[..., array_ops.newaxis, 2:, array_ops.newaxis].shape) + [1], self.evaluate(sliced).shape)

    def test_single_param_slice_broadcast_batch_leading_newaxis(self):
        if False:
            i = 10
            return i + 15
        sliced = slicing._slice_single_param(array_ops.zeros([4, 3, 1]), param_ndims_to_matrix_ndims=1, slices=make_slices[array_ops.newaxis, ..., array_ops.newaxis, 2:, array_ops.newaxis], batch_shape=constant_op.constant([7, 4, 3]))
        expected = array_ops.zeros([1, 4, 3])[array_ops.newaxis, ..., array_ops.newaxis, 2:, array_ops.newaxis].shape + [1]
        self.assertAllEqual(expected, self.evaluate(sliced).shape)

    def test_single_param_multi_ellipsis(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegexp(ValueError, 'Found multiple `...`'):
            slicing._slice_single_param(array_ops.zeros([7, 6, 5, 4, 3]), param_ndims_to_matrix_ndims=2, slices=make_slices[:, ..., 2, ...], batch_shape=constant_op.constant([7, 6, 5]))

    def test_single_param_too_many_slices(self):
        if False:
            while True:
                i = 10
        with self.assertRaises((IndexError, ValueError, errors.InvalidArgumentError)):
            slicing._slice_single_param(array_ops.zeros([7, 6, 5, 4, 3]), param_ndims_to_matrix_ndims=2, slices=make_slices[:, :3, ..., -2:, :], batch_shape=constant_op.constant([7, 6, 5]))

    def test_slice_single_param_operator(self):
        if False:
            print('Hello World!')
        matrix = linear_operator_test_util.random_normal(shape=[1, 4, 3, 2, 2], dtype=dtypes.float32)
        operator = linalg.LinearOperatorFullMatrix(matrix, is_square=True)
        sliced = operator[..., array_ops.newaxis, 2:, array_ops.newaxis]
        self.assertAllEqual(list(array_ops.zeros([1, 4, 3])[..., array_ops.newaxis, 2:, array_ops.newaxis].shape), sliced.batch_shape_tensor())

    def test_slice_nested_operator(self):
        if False:
            for i in range(10):
                print('nop')
        linop = linalg.LinearOperatorKronecker([linalg.LinearOperatorBlockDiag([linalg.LinearOperatorDiag(array_ops.ones([1, 2, 2])), linalg.LinearOperatorDiag(array_ops.ones([3, 5, 2, 2]))]), linalg.LinearOperatorFullMatrix(linear_operator_test_util.random_normal(shape=[4, 1, 1, 1, 3, 3], dtype=dtypes.float32))])
        self.assertAllEqual(linop[0, ...].batch_shape_tensor(), [3, 5, 2])
        self.assertAllEqual(linop[0, ..., array_ops.newaxis].batch_shape_tensor(), [3, 5, 2, 1])
        self.assertAllEqual(linop[..., array_ops.newaxis].batch_shape_tensor(), [4, 3, 5, 2, 1])
if __name__ == '__main__':
    test.main()