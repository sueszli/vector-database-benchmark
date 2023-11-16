"""Tests for urlencoded unpacker."""
import unittest
from jsbeautifier.unpackers.urlencode import detect, unpack

class TestUrlencode(unittest.TestCase):
    """urlencode test case."""

    def test_detect(self):
        if False:
            i = 10
            return i + 15
        'Test detect() function.'

        def encoded(source):
            if False:
                while True:
                    i = 10
            return self.assertTrue(detect(source))

        def unencoded(source):
            if False:
                while True:
                    i = 10
            return self.assertFalse(detect(source))
        unencoded('')
        unencoded('var a = b')
        encoded('var%20a+=+b')
        encoded('var%20a=b')
        encoded('var%20%21%22')

    def test_unpack(self):
        if False:
            while True:
                i = 10
        'Test unpack function.'

        def equals(source, result):
            if False:
                i = 10
                return i + 15
            return self.assertEqual(unpack(source), result)
        equals('', '')
        equals('abcd', 'abcd')
        equals('var a = b', 'var a = b')
        equals('var%20a=b', 'var a=b')
        equals('var%20a+=+b', 'var a = b')
if __name__ == '__main__':
    unittest.main()