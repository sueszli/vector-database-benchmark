"""Tests for tf_doctest."""
import doctest
from absl.testing import absltest
from absl.testing import parameterized
from tensorflow.tools.docs import tf_doctest_lib

class TfDoctestOutputCheckerTest(parameterized.TestCase):

    @parameterized.parameters(['result = 1', []], ['0.0', [0.0]], ['text 1.0 text', [1.0]], ['text 1. text', [1.0]], ['text .1 text', [0.1]], ['text 1e3 text', [1000.0]], ['text 1.e3 text', [1000.0]], ['text +1. text', [1.0]], ['text -1. text', [-1.0]], ['text 1e+3 text', [1000.0]], ['text 1e-3 text', [0.001]], ['text +1E3 text', [1000.0]], ['text -1E3 text', [-1000.0]], ['text +1e-3 text', [0.001]], ['text -1e+3 text', [-1000.0]], ['.1', [0.1]], ['.1 text', [0.1]], ['text .1', [0.1]], ['0.1 text', [0.1]], ['text 0.1', [0.1]], ['0. text', [0.0]], ['text 0.', [0.0]], ['1e-1 text', [0.1]], ['text 1e-1', [0.1]], ['text1.0 text', []], ['text 1.0text', []], ['text1.0text', []], ['0x12e4', []], ['TensorBoard: http://128.0.0.1:8888', []], ['1.0 text\n 2.0 3.0 text', [1.0, 2.0, 3.0]], ['shape (1,2,3) value -1e9', [-1000000000.0]], ['No floats at end of sentence: 1.0.', []], ['No floats with ellipsis: 1.0...', []], ['array([[1., 2., 3.],\n                 [4., 5., 6.]], dtype=float32)', [1, 2, 3, 4, 5, 6]], ['(0.0002+30000j)', [0.0002, 30000]], ['(2.3e-10-3.34e+9j)', [2.3e-10, -3340000000.0]], ['array([1.27+5.j])', [1.27, 5]], ['(2.3e-10+3.34e+9j)', [2.3e-10, 3340000000.0]], ['array([1.27e-09+5.e+00j,\n                 2.30e+01-1.e-03j])', [1.27e-09, 5.0, 23.0, -0.001]], ['1e-6', [0]], ['0.0', [1e-06]], ['1.000001e9', [1000000000.0]], ['1e9', [1000001000.0]])
    def test_extract_floats(self, text, expected_floats):
        if False:
            return 10
        extract_floats = tf_doctest_lib._FloatExtractor()
        output_checker = tf_doctest_lib.TfDoctestOutputChecker()
        (text_parts, extracted_floats) = extract_floats(text)
        text_with_wildcards = '...'.join(text_parts)
        try:
            self.assertLen(extracted_floats, len(expected_floats))
        except AssertionError as e:
            msg = '\n\n  expected: {}\n  found:     {}'.format(expected_floats, extracted_floats)
            e.args = (e.args[0] + msg,)
            raise e
        try:
            self.assertTrue(output_checker._allclose(expected_floats, extracted_floats))
        except AssertionError as e:
            msg = '\n\nexpected:  {}\nfound:     {}'.format(expected_floats, extracted_floats)
            e.args = (e.args[0] + msg,)
            raise e
        try:
            self.assertTrue(doctest.OutputChecker().check_output(want=text_with_wildcards, got=text, optionflags=doctest.ELLIPSIS))
        except AssertionError as e:
            msg = '\n\n  expected: {}\n  found:     {}'.format(text_with_wildcards, text)
            e.args = (e.args[0] + msg,)
            raise e

    @parameterized.parameters(['1.001e-2', [0]], ['0.0', [0.001001]])
    def test_fail_tolerences(self, text, expected_floats):
        if False:
            i = 10
            return i + 15
        extract_floats = tf_doctest_lib._FloatExtractor()
        output_checker = tf_doctest_lib.TfDoctestOutputChecker()
        (_, extracted_floats) = extract_floats(text)
        try:
            self.assertFalse(output_checker._allclose(expected_floats, extracted_floats))
        except AssertionError as e:
            msg = '\n\nThese matched! They should not have.\n\n\n  Expected:  {}\n  found:     {}'.format(expected_floats, extracted_floats)
            e.args = (e.args[0] + msg,)
            raise e

    def test_want_no_floats(self):
        if False:
            return 10
        want = 'text ... text'
        got = 'text 1.0 1.2 1.9 text'
        output_checker = tf_doctest_lib.TfDoctestOutputChecker()
        self.assertTrue(output_checker.check_output(want=want, got=got, optionflags=doctest.ELLIPSIS))

    @parameterized.parameters(['text [1.0 ] text', 'text [1.00] text'], ['text [ 1.0] text', 'text [1.0 ] text'], ['text [ 1.0 ] text', 'text [ 1.0] text'], ['text [1.000] text', 'text [ 1.0 ] text'])
    def test_extra_spaces(self, want, got):
        if False:
            return 10
        output_checker = tf_doctest_lib.TfDoctestOutputChecker()
        self.assertTrue(output_checker.check_output(want=want, got=got, optionflags=doctest.ELLIPSIS))

    @parameterized.parameters(['Hello. 2.0', 'Hello. 2.0000001'], ['Hello... 2.0', 'Hello   2.0000001'])
    def test_extra_dots(self, want, got):
        if False:
            while True:
                i = 10
        output_checker = tf_doctest_lib.TfDoctestOutputChecker()
        self.assertTrue(output_checker.check_output(want=want, got=got, optionflags=doctest.ELLIPSIS))

    @parameterized.parameters(['1.0, ..., 1.0', '1.0, 1.0, 1.0'], ['1.0, 1.0..., 1.0', '1.0, 1.002, 1.0'])
    def test_wrong_float_counts(self, want, got):
        if False:
            for i in range(10):
                print('nop')
        output_checker = tf_doctest_lib.TfDoctestOutputChecker()
        output_checker.check_output(want=want, got=got, optionflags=doctest.ELLIPSIS)
        example = doctest.Example('None', want=want)
        result = output_checker.output_difference(example=example, got=got, optionflags=doctest.ELLIPSIS)
        self.assertIn("doesn't work if *some* of the", result)

    @parameterized.parameters(['<...>', ('<...>', False)], ['TensorFlow', ('TensorFlow', False)], ['tf.Variable([[1, 2], [3, 4]])', ('tf.Variable([[1, 2], [3, 4]])', False)], ['<tf.Tensor: shape=(), dtype=float32, numpy=inf>', ('inf', True)], ['<tf.RaggedTensor:... shape=(2, 2), numpy=1>', ('<tf.RaggedTensor:... shape=(2, 2), numpy=1>', False)], ['<tf.Tensor: shape=(2, 2), dtype=int32, numpy=\n              array([[2, 2],\n                     [3, 5]], dtype=int32)>', ('\n              array([[2, 2],\n                     [3, 5]], dtype=int32)', True)], ['[<tf.Tensor: shape=(2,), dtype=int32, numpy=array([1, 2], dtype=int32)>, <tf.Tensor: shape=(2,), dtype=int32, numpy=array([3, 4], dtype=int32)>]', ('[array([1, 2], dtype=int32), array([3, 4], dtype=int32)]', True)])
    def test_tf_tensor_numpy_output(self, string, expected_output):
        if False:
            print('Hello World!')
        output_checker = tf_doctest_lib.TfDoctestOutputChecker()
        output = output_checker._tf_tensor_numpy_output(string)
        self.assertEqual(expected_output, output)
if __name__ == '__main__':
    absltest.main()