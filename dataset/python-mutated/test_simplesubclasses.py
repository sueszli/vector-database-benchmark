import unittest
from ctypes import *

class MyInt(c_int):

    def __eq__(self, other):
        if False:
            print('Hello World!')
        if type(other) != MyInt:
            return NotImplementedError
        return self.value == other.value

class Test(unittest.TestCase):

    def test_compare(self):
        if False:
            return 10
        self.assertEqual(MyInt(3), MyInt(3))
        self.assertNotEqual(MyInt(42), MyInt(43))

    def test_ignore_retval(self):
        if False:
            for i in range(10):
                print('nop')
        proto = CFUNCTYPE(None)

        def func():
            if False:
                return 10
            return (1, 'abc', None)
        cb = proto(func)
        self.assertEqual(None, cb())

    def test_int_callback(self):
        if False:
            for i in range(10):
                print('nop')
        args = []

        def func(arg):
            if False:
                i = 10
                return i + 15
            args.append(arg)
            return arg
        cb = CFUNCTYPE(None, MyInt)(func)
        self.assertEqual(None, cb(42))
        self.assertEqual(type(args[-1]), MyInt)
        cb = CFUNCTYPE(c_int, c_int)(func)
        self.assertEqual(42, cb(42))
        self.assertEqual(type(args[-1]), int)

    def test_int_struct(self):
        if False:
            for i in range(10):
                print('nop')

        class X(Structure):
            _fields_ = [('x', MyInt)]
        self.assertEqual(X().x, MyInt())
        s = X()
        s.x = MyInt(42)
        self.assertEqual(s.x, MyInt(42))
if __name__ == '__main__':
    unittest.main()