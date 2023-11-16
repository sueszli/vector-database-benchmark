"""Tests for string_length_op."""
from tensorflow.python.framework import test_util
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test

class StringLengthOpTest(test.TestCase):

    def testStringLength(self):
        if False:
            print('Hello World!')
        strings = [[['1', '12'], ['123', '1234'], ['12345', '123456']]]
        with self.cached_session() as sess:
            lengths = string_ops.string_length(strings)
            values = self.evaluate(lengths)
            self.assertAllEqual(values, [[[1, 2], [3, 4], [5, 6]]])

    @test_util.run_deprecated_v1
    def testUnit(self):
        if False:
            i = 10
            return i + 15
        unicode_strings = [u'HÃllo', u'😄']
        utf8_strings = [s.encode('utf-8') for s in unicode_strings]
        expected_utf8_byte_lengths = [6, 4]
        expected_utf8_char_lengths = [5, 1]
        with self.session() as sess:
            utf8_byte_lengths = string_ops.string_length(utf8_strings, unit='BYTE')
            utf8_char_lengths = string_ops.string_length(utf8_strings, unit='UTF8_CHAR')
            self.assertAllEqual(self.evaluate(utf8_byte_lengths), expected_utf8_byte_lengths)
            self.assertAllEqual(self.evaluate(utf8_char_lengths), expected_utf8_char_lengths)
            with self.assertRaisesRegex(ValueError, 'Attr \'unit\' of \'StringLength\' Op passed string \'XYZ\' not in: "BYTE", "UTF8_CHAR"'):
                string_ops.string_length(utf8_strings, unit='XYZ')

    @test_util.run_deprecated_v1
    def testLegacyPositionalName(self):
        if False:
            i = 10
            return i + 15
        strings = [[['1', '12'], ['123', '1234'], ['12345', '123456']]]
        lengths = string_ops.string_length(strings, 'some_name')
        with self.session():
            self.assertAllEqual(lengths, [[[1, 2], [3, 4], [5, 6]]])
if __name__ == '__main__':
    test.main()