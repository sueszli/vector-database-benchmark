import unittest
from test.support import import_helper
from test.support import warnings_helper
import_helper.import_module('_testcapi')
from _testcapi import _test_structmembersType, CHAR_MAX, CHAR_MIN, UCHAR_MAX, SHRT_MAX, SHRT_MIN, USHRT_MAX, INT_MAX, INT_MIN, UINT_MAX, LONG_MAX, LONG_MIN, ULONG_MAX, LLONG_MAX, LLONG_MIN, ULLONG_MAX, PY_SSIZE_T_MAX, PY_SSIZE_T_MIN
ts = _test_structmembersType(False, 1, 2, 3, 4, 5, 6, 7, 8, 23, 9.99999, 10.101010101, 'hi')

class ReadWriteTests(unittest.TestCase):

    def test_bool(self):
        if False:
            while True:
                i = 10
        ts.T_BOOL = True
        self.assertEqual(ts.T_BOOL, True)
        ts.T_BOOL = False
        self.assertEqual(ts.T_BOOL, False)
        self.assertRaises(TypeError, setattr, ts, 'T_BOOL', 1)

    def test_byte(self):
        if False:
            return 10
        ts.T_BYTE = CHAR_MAX
        self.assertEqual(ts.T_BYTE, CHAR_MAX)
        ts.T_BYTE = CHAR_MIN
        self.assertEqual(ts.T_BYTE, CHAR_MIN)
        ts.T_UBYTE = UCHAR_MAX
        self.assertEqual(ts.T_UBYTE, UCHAR_MAX)

    def test_short(self):
        if False:
            while True:
                i = 10
        ts.T_SHORT = SHRT_MAX
        self.assertEqual(ts.T_SHORT, SHRT_MAX)
        ts.T_SHORT = SHRT_MIN
        self.assertEqual(ts.T_SHORT, SHRT_MIN)
        ts.T_USHORT = USHRT_MAX
        self.assertEqual(ts.T_USHORT, USHRT_MAX)

    def test_int(self):
        if False:
            return 10
        ts.T_INT = INT_MAX
        self.assertEqual(ts.T_INT, INT_MAX)
        ts.T_INT = INT_MIN
        self.assertEqual(ts.T_INT, INT_MIN)
        ts.T_UINT = UINT_MAX
        self.assertEqual(ts.T_UINT, UINT_MAX)

    def test_long(self):
        if False:
            print('Hello World!')
        ts.T_LONG = LONG_MAX
        self.assertEqual(ts.T_LONG, LONG_MAX)
        ts.T_LONG = LONG_MIN
        self.assertEqual(ts.T_LONG, LONG_MIN)
        ts.T_ULONG = ULONG_MAX
        self.assertEqual(ts.T_ULONG, ULONG_MAX)

    def test_py_ssize_t(self):
        if False:
            i = 10
            return i + 15
        ts.T_PYSSIZET = PY_SSIZE_T_MAX
        self.assertEqual(ts.T_PYSSIZET, PY_SSIZE_T_MAX)
        ts.T_PYSSIZET = PY_SSIZE_T_MIN
        self.assertEqual(ts.T_PYSSIZET, PY_SSIZE_T_MIN)

    @unittest.skipUnless(hasattr(ts, 'T_LONGLONG'), 'long long not present')
    def test_longlong(self):
        if False:
            return 10
        ts.T_LONGLONG = LLONG_MAX
        self.assertEqual(ts.T_LONGLONG, LLONG_MAX)
        ts.T_LONGLONG = LLONG_MIN
        self.assertEqual(ts.T_LONGLONG, LLONG_MIN)
        ts.T_ULONGLONG = ULLONG_MAX
        self.assertEqual(ts.T_ULONGLONG, ULLONG_MAX)
        ts.T_LONGLONG = 3
        self.assertEqual(ts.T_LONGLONG, 3)
        ts.T_ULONGLONG = 4
        self.assertEqual(ts.T_ULONGLONG, 4)

    def test_bad_assignments(self):
        if False:
            print('Hello World!')
        integer_attributes = ['T_BOOL', 'T_BYTE', 'T_UBYTE', 'T_SHORT', 'T_USHORT', 'T_INT', 'T_UINT', 'T_LONG', 'T_ULONG', 'T_PYSSIZET']
        if hasattr(ts, 'T_LONGLONG'):
            integer_attributes.extend(['T_LONGLONG', 'T_ULONGLONG'])
        for nonint in (None, 3.2j, 'full of eels', {}, []):
            for attr in integer_attributes:
                self.assertRaises(TypeError, setattr, ts, attr, nonint)

    def test_inplace_string(self):
        if False:
            while True:
                i = 10
        self.assertEqual(ts.T_STRING_INPLACE, 'hi')
        self.assertRaises(TypeError, setattr, ts, 'T_STRING_INPLACE', 's')
        self.assertRaises(TypeError, delattr, ts, 'T_STRING_INPLACE')

class TestWarnings(unittest.TestCase):

    def test_byte_max(self):
        if False:
            i = 10
            return i + 15
        with warnings_helper.check_warnings(('', RuntimeWarning)):
            ts.T_BYTE = CHAR_MAX + 1

    def test_byte_min(self):
        if False:
            for i in range(10):
                print('nop')
        with warnings_helper.check_warnings(('', RuntimeWarning)):
            ts.T_BYTE = CHAR_MIN - 1

    def test_ubyte_max(self):
        if False:
            print('Hello World!')
        with warnings_helper.check_warnings(('', RuntimeWarning)):
            ts.T_UBYTE = UCHAR_MAX + 1

    def test_short_max(self):
        if False:
            i = 10
            return i + 15
        with warnings_helper.check_warnings(('', RuntimeWarning)):
            ts.T_SHORT = SHRT_MAX + 1

    def test_short_min(self):
        if False:
            return 10
        with warnings_helper.check_warnings(('', RuntimeWarning)):
            ts.T_SHORT = SHRT_MIN - 1

    def test_ushort_max(self):
        if False:
            while True:
                i = 10
        with warnings_helper.check_warnings(('', RuntimeWarning)):
            ts.T_USHORT = USHRT_MAX + 1
if __name__ == '__main__':
    unittest.main()