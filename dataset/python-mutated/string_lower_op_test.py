"""Tests for string_lower_op."""
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test

class StringLowerOpTest(test.TestCase):
    """Test cases for tf.strings.lower."""

    def test_string_lower(self):
        if False:
            print('Hello World!')
        strings = ['Pigs on The Wing', 'aNimals']
        with self.cached_session():
            output = string_ops.string_lower(strings)
            output = self.evaluate(output)
            self.assertAllEqual(output, [b'pigs on the wing', b'animals'])

    def test_string_lower_2d(self):
        if False:
            print('Hello World!')
        strings = [['pigS on THE wIng', 'aniMals'], [' hello ', '\n\tWorld! \r \n']]
        with self.cached_session():
            output = string_ops.string_lower(strings)
            output = self.evaluate(output)
            self.assertAllEqual(output, [[b'pigs on the wing', b'animals'], [b' hello ', b'\n\tworld! \r \n']])

    def test_string_upper_unicode(self):
        if False:
            for i in range(10):
                print('nop')
        strings = [['ÓÓSSCHLOË']]
        with self.cached_session():
            output = string_ops.string_lower(strings, encoding='utf-8')
            output = self.evaluate(output)
            self.assertAllEqual(output, [[b'\xc3\xb3\xc3\xb3sschlo\xc3\xab']])
if __name__ == '__main__':
    test.main()