"""Tests for tf.strings.to_bytes op."""
from absl.testing import parameterized
from tensorflow.python.eager import def_function
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_string_ops
from tensorflow.python.platform import test

class StringsToBytesOpTest(test_util.TensorFlowTestCase, parameterized.TestCase):

    @parameterized.parameters((b'hello', [b'h', b'e', b'l', b'l', b'o']), ([b'hello', b'123'], [[b'h', b'e', b'l', b'l', b'o'], [b'1', b'2', b'3']]), ([[b'abc', b'de'], [b'fgh', b'']], [[[b'a', b'b', b'c'], [b'd', b'e']], [[b'f', b'g', b'h'], []]]), (ragged_factory_ops.constant_value([[b'abc', b'de'], [b'f']]), [[[b'a', b'b', b'c'], [b'd', b'e']], [[b'f']]]), (ragged_factory_ops.constant_value([[[b'big', b'small'], [b'red']], [[b'cat', b'dog'], [b'ox']]]), [[[[b'b', b'i', b'g'], [b's', b'm', b'a', b'l', b'l']], [[b'r', b'e', b'd']]], [[[b'c', b'a', b't'], [b'd', b'o', b'g']], [[b'o', b'x']]]]), (b'', []), (b'\x00', [b'\x00']), (u'仅今年前'.encode('utf-8'), [b'\xe4', b'\xbb', b'\x85', b'\xe4', b'\xbb', b'\x8a', b'\xe5', b'\xb9', b'\xb4', b'\xe5', b'\x89', b'\x8d']))
    def testStringToBytes(self, source, expected):
        if False:
            i = 10
            return i + 15
        expected = ragged_factory_ops.constant_value(expected, dtype=object)
        result = ragged_string_ops.string_bytes_split(source)
        self.assertAllEqual(expected, result)

    def testUnknownInputRankError(self):
        if False:
            print('Hello World!')

        @def_function.function(input_signature=[tensor_spec.TensorSpec(None)])
        def f(v):
            if False:
                print('Hello World!')
            return ragged_string_ops.string_bytes_split(v)
        with self.assertRaisesRegex(TypeError, 'Binding inputs to tf.function failed'):
            f(['foo'])
if __name__ == '__main__':
    test.main()