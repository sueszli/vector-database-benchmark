"""Tests for string_upper_op."""
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test

class StringUpperOpTest(test.TestCase):
    """Test cases for tf.strings.upper."""

    def test_string_upper(self):
        if False:
            i = 10
            return i + 15
        strings = ['Pigs on The Wing', 'aNimals']
        with self.cached_session():
            output = string_ops.string_upper(strings)
            output = self.evaluate(output)
            self.assertAllEqual(output, [b'PIGS ON THE WING', b'ANIMALS'])

    def test_string_upper_2d(self):
        if False:
            i = 10
            return i + 15
        strings = [['pigS on THE wIng', 'aniMals'], [' hello ', '\n\tWorld! \r \n']]
        with self.cached_session():
            output = string_ops.string_upper(strings)
            output = self.evaluate(output)
            self.assertAllEqual(output, [[b'PIGS ON THE WING', b'ANIMALS'], [b' HELLO ', b'\n\tWORLD! \r \n']])

    def test_string_upper_unicode(self):
        if False:
            print('Hello World!')
        strings = [['óósschloë']]
        with self.cached_session():
            output = string_ops.string_upper(strings, encoding='utf-8')
            output = self.evaluate(output)
            self.assertAllEqual(output, [[b'\xc3\x93\xc3\x93SSCHLO\xc3\x8b']])
if __name__ == '__main__':
    test.main()