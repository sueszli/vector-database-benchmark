"""Tests for UnicodeEncode op from ragged_string_ops."""
from absl.testing import parameterized
import numpy as np
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors_impl as errors
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_string_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_tensor_value
from tensorflow.python.platform import test

class UnicodeEncodeOpTest(test.TestCase, parameterized.TestCase):

    def assertAllEqual(self, rt, expected):
        if False:
            return 10
        with self.cached_session() as sess:
            value = sess.run(rt)
            if isinstance(value, np.ndarray):
                value = value.tolist()
            elif isinstance(value, ragged_tensor_value.RaggedTensorValue):
                value = value.to_list()
            self.assertEqual(value, expected)

    def testScalar(self):
        if False:
            while True:
                i = 10
        with self.cached_session():
            with self.assertRaises(ValueError):
                ragged_string_ops.unicode_encode(72, 'UTF-8')
        with self.cached_session():
            with self.assertRaises(ValueError):
                ragged_string_ops.unicode_encode(constant_op.constant(72), 'UTF-8')

    def testRequireParams(self):
        if False:
            while True:
                i = 10
        with self.cached_session():
            with self.assertRaises(TypeError):
                ragged_string_ops.unicode_encode()
        with self.cached_session():
            with self.assertRaises(TypeError):
                ragged_string_ops.unicode_encode(72)
        with self.cached_session():
            with self.assertRaises(TypeError):
                ragged_string_ops.unicode_encode(encoding='UTF-8')

    @parameterized.parameters('UTF-8', 'UTF-16-BE', 'UTF-32-BE')
    def testStrictErrors(self, encoding):
        if False:
            while True:
                i = 10
        test_value = np.array([ord('H'), ord('e'), 2147483647, -1, ord('o')], np.int32)
        with self.cached_session() as session:
            with self.assertRaises(errors.InvalidArgumentError):
                session.run(ragged_string_ops.unicode_encode(test_value, encoding, 'strict'))

    @parameterized.parameters('UTF-8', 'UTF-16-BE', 'UTF-32-BE')
    @test_util.run_v1_only('b/120545219')
    def testIgnoreErrors(self, encoding):
        if False:
            for i in range(10):
                print('nop')
        test_value = np.array([ord('H'), ord('e'), 2147483647, -1, ord('o')], np.int32)
        expected_value = u'Heo'.encode(encoding)
        unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding, 'ignore')
        self.assertAllEqual(unicode_encode_op, expected_value)

    @parameterized.parameters('UTF-8', 'UTF-16-BE', 'UTF-32-BE')
    @test_util.run_v1_only('b/120545219')
    def testReplaceErrors(self, encoding):
        if False:
            return 10
        test_value = np.array([ord('H'), ord('e'), 2147483647, -1, ord('o')], np.int32)
        expected_value = u'He��o'.encode(encoding)
        unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding, 'replace')
        self.assertAllEqual(unicode_encode_op, expected_value)
        test_value = np.array([ord('H'), ord('e'), 2147483647, -1, ord('o')], np.int32)
        expected_value = u'Heooo'.encode(encoding)
        unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding, 'replace', ord('o'))
        self.assertAllEqual(unicode_encode_op, expected_value)
        test_value = np.array([ord('H'), ord('e'), 2147483647, -1, ord('o')], np.int32)
        expected_value = u'He��o'.encode(encoding)
        unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding)
        self.assertAllEqual(unicode_encode_op, expected_value)
        test_value = np.array([55297], np.int32)
        expected_value = u'A'.encode(encoding)
        unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding, 'replace', 65)
        self.assertAllEqual(unicode_encode_op, expected_value)
        test_value = np.array([131071], np.int32)
        expected_value = u'A'.encode(encoding)
        unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding, 'replace', 65)
        self.assertAllEqual(unicode_encode_op, expected_value)
        test_value = np.array([ord('H'), ord('e'), 2147483647, -1, ord('o')], np.int32)
        unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding, 'replace', 1114112)
        with self.assertRaises(errors.InvalidArgumentError):
            self.evaluate(unicode_encode_op)

    @parameterized.parameters('UTF-8', 'UTF-16-BE', 'UTF-32-BE')
    @test_util.run_v1_only('b/120545219')
    def testVector(self, encoding):
        if False:
            i = 10
            return i + 15
        test_value = np.array([ord('H'), ord('e'), ord('l'), ord('l'), ord('o')], np.int32)
        expected_value = u'Hello'.encode(encoding)
        unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding)
        self.assertAllEqual(unicode_encode_op, expected_value)
        test_value = np.array([ord('H'), ord('e'), 195, 195, 128516], np.int32)
        expected_value = u'HeÃÃ😄'.encode(encoding)
        unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding)
        self.assertAllEqual(unicode_encode_op, expected_value)
        test_value = np.array([ord('H')], np.int32)
        expected_value = u'H'.encode(encoding)
        unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding)
        self.assertAllEqual(unicode_encode_op, expected_value)
        test_value = np.array([128516], np.int32)
        expected_value = u'😄'.encode(encoding)
        unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding)
        self.assertAllEqual(unicode_encode_op, expected_value)

    @parameterized.parameters('UTF-8', 'UTF-16-BE', 'UTF-32-BE')
    @test_util.run_v1_only('b/120545219')
    def testMatrix(self, encoding):
        if False:
            i = 10
            return i + 15
        test_value = np.array([[72, 128516, 108, 108, 111], [87, 128516, 114, 108, 100]], np.int32)
        expected_value = [u'H😄llo'.encode(encoding), u'W😄rld'.encode(encoding)]
        unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding)
        self.assertAllEqual(unicode_encode_op, expected_value)

    @parameterized.parameters('UTF-8', 'UTF-16-BE', 'UTF-32-BE')
    @test_util.run_v1_only('b/120545219')
    def test3DimMatrix(self, encoding):
        if False:
            print('Hello World!')
        test_value = constant_op.constant([[[72, 101, 108, 108, 111], [87, 111, 114, 108, 100]], [[102, 105, 120, 101, 100], [119, 111, 114, 100, 115]], [[72, 121, 112, 101, 114], [99, 117, 98, 101, 46]]], np.int32)
        expected_value = [[u'Hello'.encode(encoding), u'World'.encode(encoding)], [u'fixed'.encode(encoding), u'words'.encode(encoding)], [u'Hyper'.encode(encoding), u'cube.'.encode(encoding)]]
        unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding)
        self.assertAllEqual(unicode_encode_op, expected_value)

    @parameterized.parameters('UTF-8', 'UTF-16-BE', 'UTF-32-BE')
    @test_util.run_v1_only('b/120545219')
    def test4DimMatrix(self, encoding):
        if False:
            print('Hello World!')
        test_value = constant_op.constant([[[[72, 101, 108, 108, 111]], [[87, 111, 114, 108, 100]]], [[[102, 105, 120, 101, 100]], [[119, 111, 114, 100, 115]]], [[[72, 121, 112, 101, 114]], [[99, 117, 98, 101, 46]]]], np.int32)
        expected_value = [[[u'Hello'.encode(encoding)], [u'World'.encode(encoding)]], [[u'fixed'.encode(encoding)], [u'words'.encode(encoding)]], [[u'Hyper'.encode(encoding)], [u'cube.'.encode(encoding)]]]
        unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding)
        self.assertAllEqual(unicode_encode_op, expected_value)

    @parameterized.parameters('UTF-8', 'UTF-16-BE', 'UTF-32-BE')
    @test_util.run_v1_only('b/120545219')
    def testRaggedMatrix(self, encoding):
        if False:
            i = 10
            return i + 15
        test_value = ragged_factory_ops.constant([[ord('H'), 195, ord('l'), ord('l'), ord('o')], [ord('W'), 128516, ord('r'), ord('l'), ord('d'), ord('.')]], np.int32)
        expected_value = [u'HÃllo'.encode(encoding), u'W😄rld.'.encode(encoding)]
        unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding)
        self.assertAllEqual(unicode_encode_op, expected_value)

    @parameterized.parameters('UTF-8', 'UTF-16-BE', 'UTF-32-BE')
    @test_util.run_v1_only('b/120545219')
    def test3DimMatrixWithRagged2ndDim(self, encoding):
        if False:
            return 10
        test_value = ragged_factory_ops.constant([[[72, 101, 108, 108, 111], [87, 111, 114, 108, 100]], [[102, 105, 120, 101, 100]], [[72, 121, 112, 101, 114], [119, 111, 114, 100, 115], [99, 117, 98, 101, 46]]], np.int32)
        expected_value = [[u'Hello'.encode(encoding), u'World'.encode(encoding)], [u'fixed'.encode(encoding)], [u'Hyper'.encode(encoding), u'words'.encode(encoding), u'cube.'.encode(encoding)]]
        unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding)
        self.assertAllEqual(unicode_encode_op, expected_value)

    @parameterized.parameters('UTF-8', 'UTF-16-BE', 'UTF-32-BE')
    @test_util.run_v1_only('b/120545219')
    def test3DimMatrixWithRagged3rdDim(self, encoding):
        if False:
            i = 10
            return i + 15
        test_value = ragged_factory_ops.constant([[[72, 101, 108, 108, 111], [87, 111, 114, 108, 100, 46]], [[68, 111, 110, 39, 116], [119, 195, 114, 114, 121, 44, 32, 98, 101]], [[128516], []]], np.int32)
        expected_value = [[u'Hello'.encode(encoding), u'World.'.encode(encoding)], [u"Don't".encode(encoding), u'wÃrry, be'.encode(encoding)], [u'😄'.encode(encoding), u''.encode(encoding)]]
        unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding)
        self.assertAllEqual(unicode_encode_op, expected_value)

    @parameterized.parameters('UTF-8', 'UTF-16-BE', 'UTF-32-BE')
    @test_util.run_v1_only('b/120545219')
    def test3DimMatrixWithRagged2ndAnd3rdDim(self, encoding):
        if False:
            print('Hello World!')
        test_value = ragged_factory_ops.constant([[[72, 101, 108, 108, 111], [87, 111, 114, 108, 100, 46]], [], [[128516]]], np.int32)
        expected_value = [[u'Hello'.encode(encoding), u'World.'.encode(encoding)], [], [u'😄'.encode(encoding)]]
        unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding)
        self.assertAllEqual(unicode_encode_op, expected_value)

    @parameterized.parameters('UTF-8', 'UTF-16-BE', 'UTF-32-BE')
    @test_util.run_v1_only('b/120545219')
    def test4DimRaggedMatrix(self, encoding):
        if False:
            for i in range(10):
                print('nop')
        test_value = ragged_factory_ops.constant([[[[72, 101, 108, 108, 111], [87, 111, 114, 108, 100]]], [[[]], [[72, 121, 112, 101]]]], np.int32)
        expected_value = [[[u'Hello'.encode(encoding), u'World'.encode(encoding)]], [[u''.encode(encoding)], [u'Hype'.encode(encoding)]]]
        unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding)
        self.assertAllEqual(unicode_encode_op, expected_value)

    @parameterized.parameters('UTF-8', 'UTF-16-BE', 'UTF-32-BE')
    @test_util.run_v1_only('b/120545219')
    def testRaggedMatrixWithMultiDimensionInnerValues(self, encoding):
        if False:
            for i in range(10):
                print('nop')
        test_flat_values = constant_op.constant([[[72, 101, 108, 108, 111], [87, 111, 114, 108, 100]], [[102, 105, 120, 101, 100], [119, 111, 114, 100, 115]], [[72, 121, 112, 101, 114], [99, 117, 98, 101, 46]]])
        test_row_splits = [constant_op.constant([0, 2, 3], dtype=np.int64), constant_op.constant([0, 1, 1, 3], dtype=np.int64)]
        test_value = ragged_tensor.RaggedTensor.from_nested_row_splits(test_flat_values, test_row_splits)
        expected_value = [[[[u'Hello'.encode(encoding), u'World'.encode(encoding)]], []], [[[u'fixed'.encode(encoding), u'words'.encode(encoding)], [u'Hyper'.encode(encoding), u'cube.'.encode(encoding)]]]]
        unicode_encode_op = ragged_string_ops.unicode_encode(test_value, encoding)
        self.assertAllEqual(unicode_encode_op, expected_value)

    def testUnknownInputRankError(self):
        if False:
            i = 10
            return i + 15

        @def_function.function(input_signature=[tensor_spec.TensorSpec(None)])
        def f(v):
            if False:
                print('Hello World!')
            return ragged_string_ops.unicode_encode(v, 'UTF-8')
        with self.assertRaisesRegex(ValueError, 'Rank of input_tensor must be statically known.'):
            f([72, 101, 108, 108, 111])
if __name__ == '__main__':
    test.main()