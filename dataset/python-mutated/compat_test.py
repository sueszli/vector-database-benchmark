"""Compat tests."""
from tensorflow.python.platform import test
from tensorflow.python.util import compat

class CompatTest(test.TestCase):

    def testCompatValidEncoding(self):
        if False:
            return 10
        self.assertEqual(compat.as_bytes('hello', 'utf8'), b'hello')
        self.assertEqual(compat.as_text(b'hello', 'utf-8'), 'hello')

    def testCompatInvalidEncoding(self):
        if False:
            print('Hello World!')
        with self.assertRaises(LookupError):
            compat.as_bytes('hello', 'invalid')
        with self.assertRaises(LookupError):
            compat.as_text(b'hello', 'invalid')
if __name__ == '__main__':
    test.main()