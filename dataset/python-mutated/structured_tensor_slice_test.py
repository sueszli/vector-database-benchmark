"""Tests for StructuredTensor."""
from absl.testing import parameterized
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.structured import structured_tensor
from tensorflow.python.platform import googletest

class _SliceBuilder:
    """Helper to construct arguments for __getitem__.

  Usage: _SliceBuilder()[<expr>] slice_spec Python generates for <expr>.
  """

    def __getitem__(self, slice_spec):
        if False:
            i = 10
            return i + 15
        return slice_spec
SLICE_BUILDER = _SliceBuilder()

def _make_tensor_slice_spec(slice_spec, use_constant=True):
    if False:
        for i in range(10):
            print('nop')
    'Wraps all integers in an extended slice spec w/ a tensor.\n\n  This function is used to help test slicing when the slice spec contains\n  tensors, rather than integers.\n\n  Args:\n    slice_spec: The extended slice spec.\n    use_constant: If true, then wrap each integer with a tf.constant.  If false,\n      then wrap each integer with a tf.placeholder.\n\n  Returns:\n    A copy of slice_spec, but with each integer i replaced with tf.constant(i).\n  '

    def make_piece_scalar(piece):
        if False:
            print('Hello World!')
        if isinstance(piece, int):
            scalar = constant_op.constant(piece)
            if use_constant:
                return scalar
            else:
                return array_ops.placeholder_with_default(scalar, [])
        elif isinstance(piece, slice):
            return slice(make_piece_scalar(piece.start), make_piece_scalar(piece.stop), make_piece_scalar(piece.step))
        else:
            return piece
    if isinstance(slice_spec, tuple):
        return tuple((make_piece_scalar(piece) for piece in slice_spec))
    else:
        return make_piece_scalar(slice_spec)
EXAMPLE_STRUCT = {'f1': 1, 'f2': [[1, 2], [3, 4]], 'f3': {'f3_1': 1}, 'f4': [{'f4_1': 1, 'f4_2': b'a'}, {'f4_1': 2, 'f4_2': b'b'}], 'f5': [[{'f5_1': 1}, {'f5_1': 2}], [{'f5_1': 3}, {'f5_1': 4}]]}
EXAMPLE_STRUCT_2 = {'f1': 5, 'f2': [[6, 7], [8, 9]], 'f3': {'f3_1': 9}, 'f4': [{'f4_1': 5, 'f4_2': b'A'}, {'f4_1': 6, 'f4_2': b'B'}], 'f5': [[{'f5_1': 6}, {'f5_1': 7}], [{'f5_1': 8}, {'f5_1': 9}]]}
EXAMPLE_STRUCT_VECTOR = [EXAMPLE_STRUCT] * 5 + [EXAMPLE_STRUCT_2]
EXAMPLE_STRUCT_SPEC1 = structured_tensor.StructuredTensorSpec([], {'f1': tensor_spec.TensorSpec([], dtypes.int32), 'f2': tensor_spec.TensorSpec([2, 2], dtypes.int32), 'f3': structured_tensor.StructuredTensorSpec([], {'f3_1': tensor_spec.TensorSpec([], dtypes.int32)}), 'f4': structured_tensor.StructuredTensorSpec([2], {'f4_1': tensor_spec.TensorSpec([2], dtypes.int32), 'f4_2': tensor_spec.TensorSpec([2], dtypes.string)}), 'f5': structured_tensor.StructuredTensorSpec([2, 2], {'f5_1': tensor_spec.TensorSpec([2, 2], dtypes.int32)})})

@test_util.run_all_in_graph_and_eager_modes
class StructuredTensorSliceTest(test_util.TensorFlowTestCase, parameterized.TestCase):

    def assertAllEqual(self, a, b, msg=None):
        if False:
            while True:
                i = 10
        if not (isinstance(a, structured_tensor.StructuredTensor) or isinstance(b, structured_tensor.StructuredTensor)):
            super(StructuredTensorSliceTest, self).assertAllEqual(a, b, msg)
        elif isinstance(a, structured_tensor.StructuredTensor) and isinstance(b, structured_tensor.StructuredTensor):
            a_shape = tensor_shape.as_shape(a.shape)
            b_shape = tensor_shape.as_shape(b.shape)
            a_shape.assert_is_compatible_with(b_shape)
            self.assertEqual(set(a.field_names()), set(b.field_names()))
            for field in a.field_names():
                self.assertAllEqual(a.field_value(field), b.field_value(field))
        elif isinstance(b, structured_tensor.StructuredTensor):
            self.assertAllEqual(b, a, msg)
        elif a.rank == 0:
            self.assertIsInstance(b, dict)
            self.assertEqual(set(a.field_names()), set(b))
            for (key, b_val) in b.items():
                a_val = a.field_value(key)
                self.assertAllEqual(a_val, b_val)
        else:
            self.assertIsInstance(b, (list, tuple))
            a.shape[:1].assert_is_compatible_with([len(b)])
            for i in range(len(b)):
                self.assertAllEqual(a[i], b[i])

    def _TestGetItem(self, struct, slice_spec, expected):
        if False:
            print('Hello World!')
        'Helper function for testing StructuredTensor.__getitem__.\n\n    Checks that calling `struct.__getitem__(slice_spec) returns the expected\n    value.  Checks three different configurations for each slice spec:\n\n      * Call __getitem__ with the slice spec as-is (with int values)\n      * Call __getitem__ with int values in the slice spec wrapped in\n        `tf.constant()`.\n      * Call __getitem__ with int values in the slice spec wrapped in\n        `tf.compat.v1.placeholder()` (so value is not known at graph\n        construction time).\n\n    Args:\n      struct: The StructuredTensor to test.\n      slice_spec: The slice spec.\n      expected: The expected value of struct.__getitem__(slice_spec), as a\n        python list.\n    '
        tensor_slice_spec1 = _make_tensor_slice_spec(slice_spec, True)
        tensor_slice_spec2 = _make_tensor_slice_spec(slice_spec, False)
        value1 = struct.__getitem__(slice_spec)
        value2 = struct.__getitem__(tensor_slice_spec1)
        value3 = struct.__getitem__(tensor_slice_spec2)
        self.assertAllEqual(value1, expected, 'slice_spec=%s' % (slice_spec,))
        self.assertAllEqual(value2, expected, 'slice_spec=%s' % (slice_spec,))
        self.assertAllEqual(value3, expected, 'slice_spec=%s' % (slice_spec,))

    @parameterized.parameters([(SLICE_BUILDER['f1'], EXAMPLE_STRUCT['f1']), (SLICE_BUILDER['f2'], EXAMPLE_STRUCT['f2']), (SLICE_BUILDER['f3'], EXAMPLE_STRUCT['f3']), (SLICE_BUILDER['f4'], EXAMPLE_STRUCT['f4']), (SLICE_BUILDER['f5'], EXAMPLE_STRUCT['f5']), (SLICE_BUILDER['f2', 1], EXAMPLE_STRUCT['f2'][1]), (SLICE_BUILDER['f3', 'f3_1'], EXAMPLE_STRUCT['f3']['f3_1']), (SLICE_BUILDER['f4', 1], EXAMPLE_STRUCT['f4'][1]), (SLICE_BUILDER['f4', 1, 'f4_2'], EXAMPLE_STRUCT['f4'][1]['f4_2']), (SLICE_BUILDER['f5', 0, 1], EXAMPLE_STRUCT['f5'][0][1]), (SLICE_BUILDER['f5', 0, 1, 'f5_1'], EXAMPLE_STRUCT['f5'][0][1]['f5_1']), (SLICE_BUILDER['f2', 1:], EXAMPLE_STRUCT['f2'][1:]), (SLICE_BUILDER['f4', :1], EXAMPLE_STRUCT['f4'][:1]), (SLICE_BUILDER['f4', 1:, 'f4_2'], [b'b']), (SLICE_BUILDER['f4', :, 'f4_2'], [b'a', b'b']), (SLICE_BUILDER['f5', :, :, 'f5_1'], [[1, 2], [3, 4]]), (SLICE_BUILDER[:], EXAMPLE_STRUCT), (['f2', 1], EXAMPLE_STRUCT['f2'][1])])
    def testGetitemFromScalarStruct(self, slice_spec, expected):
        if False:
            i = 10
            return i + 15
        struct = structured_tensor.StructuredTensor.from_pyval(EXAMPLE_STRUCT)
        self._TestGetItem(struct, slice_spec, expected)
        struct2 = structured_tensor.StructuredTensor.from_pyval(EXAMPLE_STRUCT, EXAMPLE_STRUCT_SPEC1)
        self._TestGetItem(struct2, slice_spec, expected)

    @parameterized.parameters([(SLICE_BUILDER[2], EXAMPLE_STRUCT_VECTOR[2]), (SLICE_BUILDER[5], EXAMPLE_STRUCT_VECTOR[5]), (SLICE_BUILDER[-2], EXAMPLE_STRUCT_VECTOR[-2]), (SLICE_BUILDER[-1], EXAMPLE_STRUCT_VECTOR[-1]), (SLICE_BUILDER[2, 'f1'], EXAMPLE_STRUCT_VECTOR[2]['f1']), (SLICE_BUILDER[-1, 'f1'], EXAMPLE_STRUCT_VECTOR[-1]['f1']), (SLICE_BUILDER[5:], EXAMPLE_STRUCT_VECTOR[5:]), (SLICE_BUILDER[3:, 'f1'], [1, 1, 5]), (SLICE_BUILDER[::2, 'f1'], [1, 1, 1]), (SLICE_BUILDER[1::2, 'f1'], [1, 1, 5]), (SLICE_BUILDER[4:, 'f5', 0, 1, 'f5_1'], [2, 7], True), (SLICE_BUILDER[4:, 'f5', :, :, 'f5_1'], [[[1, 2], [3, 4]], [[6, 7], [8, 9]]])])
    def testGetitemFromVectorStruct(self, slice_spec, expected, test_requires_typespec=False):
        if False:
            while True:
                i = 10
        if not test_requires_typespec:
            struct_vector = structured_tensor.StructuredTensor.from_pyval(EXAMPLE_STRUCT_VECTOR)
            self._TestGetItem(struct_vector, slice_spec, expected)
        struct_vector2 = structured_tensor.StructuredTensor.from_pyval(EXAMPLE_STRUCT_VECTOR, EXAMPLE_STRUCT_SPEC1._batch(6))
        self._TestGetItem(struct_vector2, slice_spec, expected)

    @parameterized.parameters([(SLICE_BUILDER[:2], "Key for indexing a StructuredTensor must be a string or a full slice \\(':'\\)"), (SLICE_BUILDER['f4', ...], 'Slicing not supported for Ellipsis'), (SLICE_BUILDER['f4', None], 'Slicing not supported for tf.newaxis'), (SLICE_BUILDER['f4', :, 0], 'Key for indexing a StructuredTensor must be a string')])
    def testGetItemError(self, slice_spec, error, exception=ValueError):
        if False:
            i = 10
            return i + 15
        struct = structured_tensor.StructuredTensor.from_pyval(EXAMPLE_STRUCT)
        with self.assertRaisesRegex(exception, error):
            struct.__getitem__(slice_spec)

    @parameterized.parameters([(SLICE_BUILDER[:, 1], 'Key for indexing a StructuredTensor must be a string')])
    def testGetItemFromVectorError(self, slice_spec, error, exception=ValueError):
        if False:
            return 10
        struct = structured_tensor.StructuredTensor.from_pyval(EXAMPLE_STRUCT_VECTOR)
        with self.assertRaisesRegex(exception, error):
            struct.__getitem__(slice_spec)
if __name__ == '__main__':
    googletest.main()