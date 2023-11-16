import unittest
from ctypes import *

class StructFieldsTestCase(unittest.TestCase):

    def test_1_A(self):
        if False:
            for i in range(10):
                print('nop')

        class X(Structure):
            pass
        self.assertEqual(sizeof(X), 0)
        X._fields_ = []
        self.assertRaises(AttributeError, setattr, X, '_fields_', [])

    def test_1_B(self):
        if False:
            i = 10
            return i + 15

        class X(Structure):
            _fields_ = []
        self.assertRaises(AttributeError, setattr, X, '_fields_', [])

    def test_2(self):
        if False:
            i = 10
            return i + 15

        class X(Structure):
            pass
        X()
        self.assertRaises(AttributeError, setattr, X, '_fields_', [])

    def test_3(self):
        if False:
            print('Hello World!')

        class X(Structure):
            pass

        class Y(Structure):
            _fields_ = [('x', X)]
        self.assertRaises(AttributeError, setattr, X, '_fields_', [])

    def test_4(self):
        if False:
            print('Hello World!')

        class X(Structure):
            pass

        class Y(X):
            pass
        self.assertRaises(AttributeError, setattr, X, '_fields_', [])
        Y._fields_ = []
        self.assertRaises(AttributeError, setattr, X, '_fields_', [])

    def test_5(self):
        if False:
            for i in range(10):
                print('nop')

        class X(Structure):
            _fields_ = (('char', c_char * 5),)
        x = X(b'#' * 5)
        x.char = b'a\x00b\x00'
        self.assertEqual(bytes(x), b'a\x00###')

    def test___set__(self):
        if False:
            while True:
                i = 10

        class MyCStruct(Structure):
            _fields_ = (('field', c_int),)
        self.assertRaises(TypeError, MyCStruct.field.__set__, 'wrong type self', 42)

        class MyCUnion(Union):
            _fields_ = (('field', c_int),)
        self.assertRaises(TypeError, MyCUnion.field.__set__, 'wrong type self', 42)

    def test___get__(self):
        if False:
            return 10

        class MyCStruct(Structure):
            _fields_ = (('field', c_int),)
        self.assertRaises(TypeError, MyCStruct.field.__get__, 'wrong type self', 42)

        class MyCUnion(Union):
            _fields_ = (('field', c_int),)
        self.assertRaises(TypeError, MyCUnion.field.__get__, 'wrong type self', 42)
if __name__ == '__main__':
    unittest.main()