from ctypes import *
from ctypes.test import need_symbol
import unittest
import _ctypes_test
testdll = CDLL(_ctypes_test.__file__)

def positive_address(a):
    if False:
        for i in range(10):
            print('nop')
    if a >= 0:
        return a
    import struct
    num_bits = struct.calcsize('P') * 8
    a += 1 << num_bits
    assert a >= 0
    return a

def c_wbuffer(init):
    if False:
        i = 10
        return i + 15
    n = len(init) + 1
    return (c_wchar * n)(*init)

class CharPointersTestCase(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        func = testdll._testfunc_p_p
        func.restype = c_long
        func.argtypes = None

    def test_paramflags(self):
        if False:
            return 10
        prototype = CFUNCTYPE(c_void_p, c_void_p)
        func = prototype(('_testfunc_p_p', testdll), ((1, 'input'),))
        try:
            func()
        except TypeError as details:
            self.assertEqual(str(details), "required argument 'input' missing")
        else:
            self.fail('TypeError not raised')
        self.assertEqual(func(None), None)
        self.assertEqual(func(input=None), None)

    def test_int_pointer_arg(self):
        if False:
            print('Hello World!')
        func = testdll._testfunc_p_p
        if sizeof(c_longlong) == sizeof(c_void_p):
            func.restype = c_longlong
        else:
            func.restype = c_long
        self.assertEqual(0, func(0))
        ci = c_int(0)
        func.argtypes = (POINTER(c_int),)
        self.assertEqual(positive_address(addressof(ci)), positive_address(func(byref(ci))))
        func.argtypes = (c_char_p,)
        self.assertRaises(ArgumentError, func, byref(ci))
        func.argtypes = (POINTER(c_short),)
        self.assertRaises(ArgumentError, func, byref(ci))
        func.argtypes = (POINTER(c_double),)
        self.assertRaises(ArgumentError, func, byref(ci))

    def test_POINTER_c_char_arg(self):
        if False:
            print('Hello World!')
        func = testdll._testfunc_p_p
        func.restype = c_char_p
        func.argtypes = (POINTER(c_char),)
        self.assertEqual(None, func(None))
        self.assertEqual(b'123', func(b'123'))
        self.assertEqual(None, func(c_char_p(None)))
        self.assertEqual(b'123', func(c_char_p(b'123')))
        self.assertEqual(b'123', func(c_buffer(b'123')))
        ca = c_char(b'a')
        self.assertEqual(ord(b'a'), func(pointer(ca))[0])
        self.assertEqual(ord(b'a'), func(byref(ca))[0])

    def test_c_char_p_arg(self):
        if False:
            print('Hello World!')
        func = testdll._testfunc_p_p
        func.restype = c_char_p
        func.argtypes = (c_char_p,)
        self.assertEqual(None, func(None))
        self.assertEqual(b'123', func(b'123'))
        self.assertEqual(None, func(c_char_p(None)))
        self.assertEqual(b'123', func(c_char_p(b'123')))
        self.assertEqual(b'123', func(c_buffer(b'123')))
        ca = c_char(b'a')
        self.assertEqual(ord(b'a'), func(pointer(ca))[0])
        self.assertEqual(ord(b'a'), func(byref(ca))[0])

    def test_c_void_p_arg(self):
        if False:
            print('Hello World!')
        func = testdll._testfunc_p_p
        func.restype = c_char_p
        func.argtypes = (c_void_p,)
        self.assertEqual(None, func(None))
        self.assertEqual(b'123', func(b'123'))
        self.assertEqual(b'123', func(c_char_p(b'123')))
        self.assertEqual(None, func(c_char_p(None)))
        self.assertEqual(b'123', func(c_buffer(b'123')))
        ca = c_char(b'a')
        self.assertEqual(ord(b'a'), func(pointer(ca))[0])
        self.assertEqual(ord(b'a'), func(byref(ca))[0])
        func(byref(c_int()))
        func(pointer(c_int()))
        func((c_int * 3)())

    @need_symbol('c_wchar_p')
    def test_c_void_p_arg_with_c_wchar_p(self):
        if False:
            while True:
                i = 10
        func = testdll._testfunc_p_p
        func.restype = c_wchar_p
        func.argtypes = (c_void_p,)
        self.assertEqual(None, func(c_wchar_p(None)))
        self.assertEqual('123', func(c_wchar_p('123')))

    def test_instance(self):
        if False:
            print('Hello World!')
        func = testdll._testfunc_p_p
        func.restype = c_void_p

        class X:
            _as_parameter_ = None
        func.argtypes = (c_void_p,)
        self.assertEqual(None, func(X()))
        func.argtypes = None
        self.assertEqual(None, func(X()))

@need_symbol('c_wchar')
class WCharPointersTestCase(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        func = testdll._testfunc_p_p
        func.restype = c_int
        func.argtypes = None

    def test_POINTER_c_wchar_arg(self):
        if False:
            while True:
                i = 10
        func = testdll._testfunc_p_p
        func.restype = c_wchar_p
        func.argtypes = (POINTER(c_wchar),)
        self.assertEqual(None, func(None))
        self.assertEqual('123', func('123'))
        self.assertEqual(None, func(c_wchar_p(None)))
        self.assertEqual('123', func(c_wchar_p('123')))
        self.assertEqual('123', func(c_wbuffer('123')))
        ca = c_wchar('a')
        self.assertEqual('a', func(pointer(ca))[0])
        self.assertEqual('a', func(byref(ca))[0])

    def test_c_wchar_p_arg(self):
        if False:
            print('Hello World!')
        func = testdll._testfunc_p_p
        func.restype = c_wchar_p
        func.argtypes = (c_wchar_p,)
        c_wchar_p.from_param('123')
        self.assertEqual(None, func(None))
        self.assertEqual('123', func('123'))
        self.assertEqual(None, func(c_wchar_p(None)))
        self.assertEqual('123', func(c_wchar_p('123')))
        self.assertEqual('123', func(c_wbuffer('123')))
        ca = c_wchar('a')
        self.assertEqual('a', func(pointer(ca))[0])
        self.assertEqual('a', func(byref(ca))[0])

class ArrayTest(unittest.TestCase):

    def test(self):
        if False:
            return 10
        func = testdll._testfunc_ai8
        func.restype = POINTER(c_int)
        func.argtypes = (c_int * 8,)
        func((c_int * 8)(1, 2, 3, 4, 5, 6, 7, 8))

        def func():
            if False:
                i = 10
                return i + 15
            pass
        CFUNCTYPE(None, c_int * 3)(func)
if __name__ == '__main__':
    unittest.main()