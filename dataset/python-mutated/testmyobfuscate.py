"""Tests for MyObfuscate unpacker."""
import unittest
import os
from jsbeautifier.unpackers.myobfuscate import detect, unpack
from jsbeautifier.unpackers.tests import __path__ as path
INPUT = os.path.join(path[0], 'test-myobfuscate-input.js')
OUTPUT = os.path.join(path[0], 'test-myobfuscate-output.js')

class TestMyObfuscate(unittest.TestCase):
    """MyObfuscate obfuscator testcase."""

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        'Load source files (encoded and decoded version) for tests.'
        with open(INPUT, 'r') as data:
            cls.input = data.read()
        with open(OUTPUT, 'r') as data:
            cls.output = data.read()

    def test_detect(self):
        if False:
            for i in range(10):
                print('nop')
        'Test detect() function.'

        def detected(source):
            if False:
                while True:
                    i = 10
            return self.assertTrue(detect(source))
        detected(self.input)

    def test_unpack(self):
        if False:
            for i in range(10):
                print('nop')
        'Test unpack() function.'

        def check(inp, out):
            if False:
                i = 10
                return i + 15
            return self.assertEqual(unpack(inp), out)
        check(self.input, self.output)
if __name__ == '__main__':
    unittest.main()