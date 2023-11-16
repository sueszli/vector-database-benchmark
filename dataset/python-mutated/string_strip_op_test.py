"""Tests for string_strip_op."""
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test

class StringStripOpTest(test.TestCase):
    """ Test cases for tf.strings.strip."""

    def test_string_strip(self):
        if False:
            for i in range(10):
                print('nop')
        strings = ['pigs on the wing', 'animals']
        with self.cached_session() as sess:
            output = string_ops.string_strip(strings)
            output = self.evaluate(output)
            self.assertAllEqual(output, [b'pigs on the wing', b'animals'])

    def test_string_strip_2d(self):
        if False:
            print('Hello World!')
        strings = [['pigs on the wing', 'animals'], [' hello ', '\n\tworld \r \n']]
        with self.cached_session() as sess:
            output = string_ops.string_strip(strings)
            output = self.evaluate(output)
            self.assertAllEqual(output, [[b'pigs on the wing', b'animals'], [b'hello', b'world']])

    def test_string_strip_with_empty_strings(self):
        if False:
            for i in range(10):
                print('nop')
        strings = [' hello ', '', 'world ', ' \t \r \n ']
        with self.cached_session() as sess:
            output = string_ops.string_strip(strings)
            output = self.evaluate(output)
            self.assertAllEqual(output, [b'hello', b'', b'world', b''])
if __name__ == '__main__':
    test.main()