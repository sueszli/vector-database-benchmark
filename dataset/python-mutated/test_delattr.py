import unittest
from ctypes import *

class X(Structure):
    _fields_ = [('foo', c_int)]

class TestCase(unittest.TestCase):

    def test_simple(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(TypeError, delattr, c_int(42), 'value')

    def test_chararray(self):
        if False:
            while True:
                i = 10
        self.assertRaises(TypeError, delattr, (c_char * 5)(), 'value')

    def test_struct(self):
        if False:
            print('Hello World!')
        self.assertRaises(TypeError, delattr, X(), 'foo')
if __name__ == '__main__':
    unittest.main()