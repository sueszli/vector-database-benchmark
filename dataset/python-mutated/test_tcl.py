import unittest
import locale
import re
import subprocess
import sys
import os
import warnings
from test import support
from test.support import import_helper
from test.support import os_helper
_tkinter = import_helper.import_module('_tkinter')
import tkinter
from tkinter import Tcl
from _tkinter import TclError
try:
    from _testcapi import INT_MAX, PY_SSIZE_T_MAX
except ImportError:
    INT_MAX = PY_SSIZE_T_MAX = sys.maxsize
tcl_version = tuple(map(int, _tkinter.TCL_VERSION.split('.')))
_tk_patchlevel = None

def get_tk_patchlevel():
    if False:
        print('Hello World!')
    global _tk_patchlevel
    if _tk_patchlevel is None:
        tcl = Tcl()
        patchlevel = tcl.call('info', 'patchlevel')
        m = re.fullmatch('(\\d+)\\.(\\d+)([ab.])(\\d+)', patchlevel)
        (major, minor, releaselevel, serial) = m.groups()
        (major, minor, serial) = (int(major), int(minor), int(serial))
        releaselevel = {'a': 'alpha', 'b': 'beta', '.': 'final'}[releaselevel]
        if releaselevel == 'final':
            _tk_patchlevel = (major, minor, serial, releaselevel, 0)
        else:
            _tk_patchlevel = (major, minor, 0, releaselevel, serial)
    return _tk_patchlevel

class TkinterTest(unittest.TestCase):

    def testFlattenLen(self):
        if False:
            return 10
        self.assertRaises(TypeError, _tkinter._flatten, True)
        self.assertRaises(TypeError, _tkinter._flatten, {})
        self.assertRaises(TypeError, _tkinter._flatten, 'string')
        self.assertRaises(TypeError, _tkinter._flatten, {'set'})

class TclTest(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.interp = Tcl()
        self.wantobjects = self.interp.tk.wantobjects()

    def testEval(self):
        if False:
            i = 10
            return i + 15
        tcl = self.interp
        tcl.eval('set a 1')
        self.assertEqual(tcl.eval('set a'), '1')

    def test_eval_null_in_result(self):
        if False:
            return 10
        tcl = self.interp
        self.assertEqual(tcl.eval('set a "a\\0b"'), 'a\x00b')

    def test_eval_surrogates_in_result(self):
        if False:
            while True:
                i = 10
        tcl = self.interp
        self.assertIn(tcl.eval('set a "<\\ud83d\\udcbb>"'), '<💻>')

    def testEvalException(self):
        if False:
            while True:
                i = 10
        tcl = self.interp
        self.assertRaises(TclError, tcl.eval, 'set a')

    def testEvalException2(self):
        if False:
            while True:
                i = 10
        tcl = self.interp
        self.assertRaises(TclError, tcl.eval, 'this is wrong')

    def testCall(self):
        if False:
            while True:
                i = 10
        tcl = self.interp
        tcl.call('set', 'a', '1')
        self.assertEqual(tcl.call('set', 'a'), '1')

    def testCallException(self):
        if False:
            return 10
        tcl = self.interp
        self.assertRaises(TclError, tcl.call, 'set', 'a')

    def testCallException2(self):
        if False:
            while True:
                i = 10
        tcl = self.interp
        self.assertRaises(TclError, tcl.call, 'this', 'is', 'wrong')

    def testSetVar(self):
        if False:
            while True:
                i = 10
        tcl = self.interp
        tcl.setvar('a', '1')
        self.assertEqual(tcl.eval('set a'), '1')

    def testSetVarArray(self):
        if False:
            i = 10
            return i + 15
        tcl = self.interp
        tcl.setvar('a(1)', '1')
        self.assertEqual(tcl.eval('set a(1)'), '1')

    def testGetVar(self):
        if False:
            print('Hello World!')
        tcl = self.interp
        tcl.eval('set a 1')
        self.assertEqual(tcl.getvar('a'), '1')

    def testGetVarArray(self):
        if False:
            for i in range(10):
                print('nop')
        tcl = self.interp
        tcl.eval('set a(1) 1')
        self.assertEqual(tcl.getvar('a(1)'), '1')

    def testGetVarException(self):
        if False:
            print('Hello World!')
        tcl = self.interp
        self.assertRaises(TclError, tcl.getvar, 'a')

    def testGetVarArrayException(self):
        if False:
            i = 10
            return i + 15
        tcl = self.interp
        self.assertRaises(TclError, tcl.getvar, 'a(1)')

    def testUnsetVar(self):
        if False:
            while True:
                i = 10
        tcl = self.interp
        tcl.setvar('a', 1)
        self.assertEqual(tcl.eval('info exists a'), '1')
        tcl.unsetvar('a')
        self.assertEqual(tcl.eval('info exists a'), '0')

    def testUnsetVarArray(self):
        if False:
            i = 10
            return i + 15
        tcl = self.interp
        tcl.setvar('a(1)', 1)
        tcl.setvar('a(2)', 2)
        self.assertEqual(tcl.eval('info exists a(1)'), '1')
        self.assertEqual(tcl.eval('info exists a(2)'), '1')
        tcl.unsetvar('a(1)')
        self.assertEqual(tcl.eval('info exists a(1)'), '0')
        self.assertEqual(tcl.eval('info exists a(2)'), '1')

    def testUnsetVarException(self):
        if False:
            print('Hello World!')
        tcl = self.interp
        self.assertRaises(TclError, tcl.unsetvar, 'a')

    def get_integers(self):
        if False:
            return 10
        integers = (0, 1, -1, 2 ** 31 - 1, -2 ** 31, 2 ** 31, -2 ** 31 - 1, 2 ** 63 - 1, -2 ** 63)
        if tcl_version >= (8, 5):
            v = get_tk_patchlevel()
            if v >= (8, 6, 0, 'final') or (8, 5, 8) <= v < (8, 6):
                integers += (2 ** 63, -2 ** 63 - 1, 2 ** 1000, -2 ** 1000)
        return integers

    def test_getint(self):
        if False:
            i = 10
            return i + 15
        tcl = self.interp.tk
        for i in self.get_integers():
            self.assertEqual(tcl.getint(' %d ' % i), i)
            if tcl_version >= (8, 5):
                self.assertEqual(tcl.getint(' %#o ' % i), i)
            self.assertEqual(tcl.getint((' %#o ' % i).replace('o', '')), i)
            self.assertEqual(tcl.getint(' %#x ' % i), i)
        if tcl_version < (8, 5):
            self.assertRaises(TclError, tcl.getint, str(2 ** 1000))
        self.assertEqual(tcl.getint(42), 42)
        self.assertRaises(TypeError, tcl.getint)
        self.assertRaises(TypeError, tcl.getint, '42', '10')
        self.assertRaises(TypeError, tcl.getint, b'42')
        self.assertRaises(TypeError, tcl.getint, 42.0)
        self.assertRaises(TclError, tcl.getint, 'a')
        self.assertRaises((TypeError, ValueError, TclError), tcl.getint, '42\x00')
        self.assertRaises((UnicodeEncodeError, ValueError, TclError), tcl.getint, '42\ud800')

    def test_getdouble(self):
        if False:
            for i in range(10):
                print('nop')
        tcl = self.interp.tk
        self.assertEqual(tcl.getdouble(' 42 '), 42.0)
        self.assertEqual(tcl.getdouble(' 42.5 '), 42.5)
        self.assertEqual(tcl.getdouble(42.5), 42.5)
        self.assertEqual(tcl.getdouble(42), 42.0)
        self.assertRaises(TypeError, tcl.getdouble)
        self.assertRaises(TypeError, tcl.getdouble, '42.5', '10')
        self.assertRaises(TypeError, tcl.getdouble, b'42.5')
        self.assertRaises(TclError, tcl.getdouble, 'a')
        self.assertRaises((TypeError, ValueError, TclError), tcl.getdouble, '42.5\x00')
        self.assertRaises((UnicodeEncodeError, ValueError, TclError), tcl.getdouble, '42.5\ud800')

    def test_getboolean(self):
        if False:
            print('Hello World!')
        tcl = self.interp.tk
        self.assertIs(tcl.getboolean('on'), True)
        self.assertIs(tcl.getboolean('1'), True)
        self.assertIs(tcl.getboolean(42), True)
        self.assertIs(tcl.getboolean(0), False)
        self.assertRaises(TypeError, tcl.getboolean)
        self.assertRaises(TypeError, tcl.getboolean, 'on', '1')
        self.assertRaises(TypeError, tcl.getboolean, b'on')
        self.assertRaises(TypeError, tcl.getboolean, 1.0)
        self.assertRaises(TclError, tcl.getboolean, 'a')
        self.assertRaises((TypeError, ValueError, TclError), tcl.getboolean, 'on\x00')
        self.assertRaises((UnicodeEncodeError, ValueError, TclError), tcl.getboolean, 'on\ud800')

    def testEvalFile(self):
        if False:
            return 10
        tcl = self.interp
        filename = os_helper.TESTFN_ASCII
        self.addCleanup(os_helper.unlink, filename)
        with open(filename, 'w') as f:
            f.write('set a 1\n            set b 2\n            set c [ expr $a + $b ]\n            ')
        tcl.evalfile(filename)
        self.assertEqual(tcl.eval('set a'), '1')
        self.assertEqual(tcl.eval('set b'), '2')
        self.assertEqual(tcl.eval('set c'), '3')

    def test_evalfile_null_in_result(self):
        if False:
            i = 10
            return i + 15
        tcl = self.interp
        filename = os_helper.TESTFN_ASCII
        self.addCleanup(os_helper.unlink, filename)
        with open(filename, 'w') as f:
            f.write('\n            set a "a\x00b"\n            set b "a\\0b"\n            ')
        tcl.evalfile(filename)
        self.assertEqual(tcl.eval('set a'), 'a\x00b')
        self.assertEqual(tcl.eval('set b'), 'a\x00b')

    def test_evalfile_surrogates_in_result(self):
        if False:
            for i in range(10):
                print('nop')
        tcl = self.interp
        encoding = tcl.call('encoding', 'system')
        self.addCleanup(tcl.call, 'encoding', 'system', encoding)
        tcl.call('encoding', 'system', 'utf-8')
        filename = os_helper.TESTFN_ASCII
        self.addCleanup(os_helper.unlink, filename)
        with open(filename, 'wb') as f:
            f.write(b'\n            set a "<\xed\xa0\xbd\xed\xb2\xbb>"\n            set b "<\\ud83d\\udcbb>"\n            ')
        tcl.evalfile(filename)
        self.assertEqual(tcl.eval('set a'), '<💻>')
        self.assertEqual(tcl.eval('set b'), '<💻>')

    def testEvalFileException(self):
        if False:
            return 10
        tcl = self.interp
        filename = 'doesnotexists'
        try:
            os.remove(filename)
        except Exception as e:
            pass
        self.assertRaises(TclError, tcl.evalfile, filename)

    def testPackageRequireException(self):
        if False:
            for i in range(10):
                print('nop')
        tcl = self.interp
        self.assertRaises(TclError, tcl.eval, 'package require DNE')

    @unittest.skipUnless(sys.platform == 'win32', 'Requires Windows')
    def testLoadWithUNC(self):
        if False:
            i = 10
            return i + 15
        fullname = os.path.abspath(sys.executable)
        if fullname[1] != ':':
            raise unittest.SkipTest('Absolute path should have drive part')
        unc_name = '\\\\%s\\%s$\\%s' % (os.environ['COMPUTERNAME'], fullname[0], fullname[3:])
        if not os.path.exists(unc_name):
            raise unittest.SkipTest('Cannot connect to UNC Path')
        with os_helper.EnvironmentVarGuard() as env:
            env.unset('TCL_LIBRARY')
            stdout = subprocess.check_output([unc_name, '-c', 'import tkinter; print(tkinter)'])
        self.assertIn(b'tkinter', stdout)

    def test_exprstring(self):
        if False:
            return 10
        tcl = self.interp
        tcl.call('set', 'a', 3)
        tcl.call('set', 'b', 6)

        def check(expr, expected):
            if False:
                return 10
            result = tcl.exprstring(expr)
            self.assertEqual(result, expected)
            self.assertIsInstance(result, str)
        self.assertRaises(TypeError, tcl.exprstring)
        self.assertRaises(TypeError, tcl.exprstring, '8.2', '+6')
        self.assertRaises(TypeError, tcl.exprstring, b'8.2 + 6')
        self.assertRaises(TclError, tcl.exprstring, 'spam')
        check('', '0')
        check('8.2 + 6', '14.2')
        check('3.1 + $a', '6.1')
        check('2 + "$a.$b"', '5.6')
        check('4*[llength "6 2"]', '8')
        check('{word one} < "word $a"', '0')
        check('4*2 < 7', '0')
        check('hypot($a, 4)', '5.0')
        check('5 / 4', '1')
        check('5 / 4.0', '1.25')
        check('5 / ( [string length "abcd"] + 0.0 )', '1.25')
        check('20.0/5.0', '4.0')
        check('"0x03" > "2"', '1')
        check('[string length "a½€"]', '3')
        check('[string length "a\\xbd\\u20ac"]', '3')
        check('"abc"', 'abc')
        check('"a½€"', 'a½€')
        check('"a\\xbd\\u20ac"', 'a½€')
        check('"a\\0b"', 'a\x00b')
        if tcl_version >= (8, 5):
            check('2**64', str(2 ** 64))

    def test_exprdouble(self):
        if False:
            print('Hello World!')
        tcl = self.interp
        tcl.call('set', 'a', 3)
        tcl.call('set', 'b', 6)

        def check(expr, expected):
            if False:
                print('Hello World!')
            result = tcl.exprdouble(expr)
            self.assertEqual(result, expected)
            self.assertIsInstance(result, float)
        self.assertRaises(TypeError, tcl.exprdouble)
        self.assertRaises(TypeError, tcl.exprdouble, '8.2', '+6')
        self.assertRaises(TypeError, tcl.exprdouble, b'8.2 + 6')
        self.assertRaises(TclError, tcl.exprdouble, 'spam')
        check('', 0.0)
        check('8.2 + 6', 14.2)
        check('3.1 + $a', 6.1)
        check('2 + "$a.$b"', 5.6)
        check('4*[llength "6 2"]', 8.0)
        check('{word one} < "word $a"', 0.0)
        check('4*2 < 7', 0.0)
        check('hypot($a, 4)', 5.0)
        check('5 / 4', 1.0)
        check('5 / 4.0', 1.25)
        check('5 / ( [string length "abcd"] + 0.0 )', 1.25)
        check('20.0/5.0', 4.0)
        check('"0x03" > "2"', 1.0)
        check('[string length "a½€"]', 3.0)
        check('[string length "a\\xbd\\u20ac"]', 3.0)
        self.assertRaises(TclError, tcl.exprdouble, '"abc"')
        if tcl_version >= (8, 5):
            check('2**64', float(2 ** 64))

    def test_exprlong(self):
        if False:
            return 10
        tcl = self.interp
        tcl.call('set', 'a', 3)
        tcl.call('set', 'b', 6)

        def check(expr, expected):
            if False:
                while True:
                    i = 10
            result = tcl.exprlong(expr)
            self.assertEqual(result, expected)
            self.assertIsInstance(result, int)
        self.assertRaises(TypeError, tcl.exprlong)
        self.assertRaises(TypeError, tcl.exprlong, '8.2', '+6')
        self.assertRaises(TypeError, tcl.exprlong, b'8.2 + 6')
        self.assertRaises(TclError, tcl.exprlong, 'spam')
        check('', 0)
        check('8.2 + 6', 14)
        check('3.1 + $a', 6)
        check('2 + "$a.$b"', 5)
        check('4*[llength "6 2"]', 8)
        check('{word one} < "word $a"', 0)
        check('4*2 < 7', 0)
        check('hypot($a, 4)', 5)
        check('5 / 4', 1)
        check('5 / 4.0', 1)
        check('5 / ( [string length "abcd"] + 0.0 )', 1)
        check('20.0/5.0', 4)
        check('"0x03" > "2"', 1)
        check('[string length "a½€"]', 3)
        check('[string length "a\\xbd\\u20ac"]', 3)
        self.assertRaises(TclError, tcl.exprlong, '"abc"')
        if tcl_version >= (8, 5):
            self.assertRaises(TclError, tcl.exprlong, '2**64')

    def test_exprboolean(self):
        if False:
            i = 10
            return i + 15
        tcl = self.interp
        tcl.call('set', 'a', 3)
        tcl.call('set', 'b', 6)

        def check(expr, expected):
            if False:
                return 10
            result = tcl.exprboolean(expr)
            self.assertEqual(result, expected)
            self.assertIsInstance(result, int)
            self.assertNotIsInstance(result, bool)
        self.assertRaises(TypeError, tcl.exprboolean)
        self.assertRaises(TypeError, tcl.exprboolean, '8.2', '+6')
        self.assertRaises(TypeError, tcl.exprboolean, b'8.2 + 6')
        self.assertRaises(TclError, tcl.exprboolean, 'spam')
        check('', False)
        for value in ('0', 'false', 'no', 'off'):
            check(value, False)
            check('"%s"' % value, False)
            check('{%s}' % value, False)
        for value in ('1', 'true', 'yes', 'on'):
            check(value, True)
            check('"%s"' % value, True)
            check('{%s}' % value, True)
        check('8.2 + 6', True)
        check('3.1 + $a', True)
        check('2 + "$a.$b"', True)
        check('4*[llength "6 2"]', True)
        check('{word one} < "word $a"', False)
        check('4*2 < 7', False)
        check('hypot($a, 4)', True)
        check('5 / 4', True)
        check('5 / 4.0', True)
        check('5 / ( [string length "abcd"] + 0.0 )', True)
        check('20.0/5.0', True)
        check('"0x03" > "2"', True)
        check('[string length "a½€"]', True)
        check('[string length "a\\xbd\\u20ac"]', True)
        self.assertRaises(TclError, tcl.exprboolean, '"abc"')
        if tcl_version >= (8, 5):
            check('2**64', True)

    @unittest.skipUnless(tcl_version >= (8, 5), 'requires Tcl version >= 8.5')
    def test_booleans(self):
        if False:
            while True:
                i = 10
        tcl = self.interp

        def check(expr, expected):
            if False:
                return 10
            result = tcl.call('expr', expr)
            if tcl.wantobjects():
                self.assertEqual(result, expected)
                self.assertIsInstance(result, int)
            else:
                self.assertIn(result, (expr, str(int(expected))))
                self.assertIsInstance(result, str)
        check('true', True)
        check('yes', True)
        check('on', True)
        check('false', False)
        check('no', False)
        check('off', False)
        check('1 < 2', True)
        check('1 > 2', False)

    def test_expr_bignum(self):
        if False:
            while True:
                i = 10
        tcl = self.interp
        for i in self.get_integers():
            result = tcl.call('expr', str(i))
            if self.wantobjects:
                self.assertEqual(result, i)
                self.assertIsInstance(result, int)
            else:
                self.assertEqual(result, str(i))
                self.assertIsInstance(result, str)
        if get_tk_patchlevel() < (8, 5):
            self.assertRaises(TclError, tcl.call, 'expr', str(2 ** 1000))

    def test_passing_values(self):
        if False:
            print('Hello World!')

        def passValue(value):
            if False:
                return 10
            return self.interp.call('set', '_', value)
        self.assertEqual(passValue(True), True if self.wantobjects else '1')
        self.assertEqual(passValue(False), False if self.wantobjects else '0')
        self.assertEqual(passValue('string'), 'string')
        self.assertEqual(passValue('string€'), 'string€')
        self.assertEqual(passValue('string💻'), 'string💻')
        self.assertEqual(passValue('str\x00ing'), 'str\x00ing')
        self.assertEqual(passValue('str\x00ing½'), 'str\x00ing½')
        self.assertEqual(passValue('str\x00ing€'), 'str\x00ing€')
        self.assertEqual(passValue('str\x00ing💻'), 'str\x00ing💻')
        if sys.platform != 'win32':
            self.assertEqual(passValue('<\udce2\udc82\udcac>'), '<€>')
            self.assertEqual(passValue('<\udced\udca0\udcbd\udced\udcb2\udcbb>'), '<💻>')
        self.assertEqual(passValue(b'str\x00ing'), b'str\x00ing' if self.wantobjects else 'str\x00ing')
        self.assertEqual(passValue(b'str\xc0\x80ing'), b'str\xc0\x80ing' if self.wantobjects else 'strÀ\x80ing')
        self.assertEqual(passValue(b'str\xbding'), b'str\xbding' if self.wantobjects else 'str½ing')
        for i in self.get_integers():
            self.assertEqual(passValue(i), i if self.wantobjects else str(i))
        if tcl_version < (8, 5):
            self.assertEqual(passValue(2 ** 1000), str(2 ** 1000))
        for f in (0.0, 1.0, -1.0, 1 / 3, sys.float_info.min, sys.float_info.max, -sys.float_info.min, -sys.float_info.max):
            if self.wantobjects:
                self.assertEqual(passValue(f), f)
            else:
                self.assertEqual(float(passValue(f)), f)
        if self.wantobjects:
            f = passValue(float('nan'))
            self.assertNotEqual(f, f)
            self.assertEqual(passValue(float('inf')), float('inf'))
            self.assertEqual(passValue(-float('inf')), -float('inf'))
        else:
            self.assertEqual(float(passValue(float('inf'))), float('inf'))
            self.assertEqual(float(passValue(-float('inf'))), -float('inf'))
        self.assertEqual(passValue((1, '2', (3.4,))), (1, '2', (3.4,)) if self.wantobjects else '1 2 3.4')
        self.assertEqual(passValue(['a', ['b', 'c']]), ('a', ('b', 'c')) if self.wantobjects else 'a {b c}')

    def test_user_command(self):
        if False:
            return 10
        result = None

        def testfunc(arg):
            if False:
                i = 10
                return i + 15
            nonlocal result
            result = arg
            return arg
        self.interp.createcommand('testfunc', testfunc)
        self.addCleanup(self.interp.tk.deletecommand, 'testfunc')

        def check(value, expected=None, *, eq=self.assertEqual):
            if False:
                print('Hello World!')
            if expected is None:
                expected = value
            nonlocal result
            result = None
            r = self.interp.call('testfunc', value)
            self.assertIsInstance(result, str)
            eq(result, expected)
            self.assertIsInstance(r, str)
            eq(r, expected)

        def float_eq(actual, expected):
            if False:
                for i in range(10):
                    print('nop')
            self.assertAlmostEqual(float(actual), expected, delta=abs(expected) * 1e-10)
        check(True, '1')
        check(False, '0')
        check('string')
        check('string½')
        check('string€')
        check('string💻')
        if sys.platform != 'win32':
            check('<\udce2\udc82\udcac>', '<€>')
            check('<\udced\udca0\udcbd\udced\udcb2\udcbb>', '<💻>')
        check('')
        check(b'string', 'string')
        check(b'string\xe2\x82\xac', 'stringâ\x82¬')
        check(b'string\xbd', 'string½')
        check(b'', '')
        check('str\x00ing')
        check('str\x00ing½')
        check('str\x00ing€')
        check(b'str\x00ing', 'str\x00ing')
        check(b'str\xc0\x80ing', 'strÀ\x80ing')
        check(b'str\xc0\x80ing\xe2\x82\xac', 'strÀ\x80ingâ\x82¬')
        for i in self.get_integers():
            check(i, str(i))
        if tcl_version < (8, 5):
            check(2 ** 1000, str(2 ** 1000))
        for f in (0.0, 1.0, -1.0):
            check(f, repr(f))
        for f in (1 / 3.0, sys.float_info.min, sys.float_info.max, -sys.float_info.min, -sys.float_info.max):
            check(f, eq=float_eq)
        check(float('inf'), eq=float_eq)
        check(-float('inf'), eq=float_eq)
        check((), '')
        check((1, (2,), (3, 4), '5 6', ()), '1 2 {3 4} {5 6} {}')
        check([1, [2], [3, 4], '5 6', []], '1 2 {3 4} {5 6} {}')

    def test_splitlist(self):
        if False:
            print('Hello World!')
        splitlist = self.interp.tk.splitlist
        call = self.interp.tk.call
        self.assertRaises(TypeError, splitlist)
        self.assertRaises(TypeError, splitlist, 'a', 'b')
        self.assertRaises(TypeError, splitlist, 2)
        testcases = [('2', ('2',)), ('', ()), ('{}', ('',)), ('""', ('',)), ('a\n b\t\r c\n ', ('a', 'b', 'c')), (b'a\n b\t\r c\n ', ('a', 'b', 'c')), ('a €', ('a', '€')), ('a 💻', ('a', '💻')), (b'a \xe2\x82\xac', ('a', '€')), (b'a \xf0\x9f\x92\xbb', ('a', '💻')), (b'a \xed\xa0\xbd\xed\xb2\xbb', ('a', '💻')), (b'a\xc0\x80b c\xc0\x80d', ('a\x00b', 'c\x00d')), ('a {b c}', ('a', 'b c')), ('a b\\ c', ('a', 'b c')), (('a', 'b c'), ('a', 'b c')), ('a 2', ('a', '2')), (('a', 2), ('a', 2)), ('a 3.4', ('a', '3.4')), (('a', 3.4), ('a', 3.4)), ((), ()), ([], ()), (['a', ['b', 'c']], ('a', ['b', 'c'])), (call('list', 1, '2', (3.4,)), (1, '2', (3.4,)) if self.wantobjects else ('1', '2', '3.4'))]
        tk_patchlevel = get_tk_patchlevel()
        if tcl_version >= (8, 5):
            if not self.wantobjects or tk_patchlevel < (8, 5, 5):
                expected = ('12', '€', 'â\x82¬', '3.4')
            else:
                expected = (12, '€', b'\xe2\x82\xac', (3.4,))
            testcases += [(call('dict', 'create', 12, '€', b'\xe2\x82\xac', (3.4,)), expected)]
        dbg_info = 'want objects? %s, Tcl version: %s, Tk patchlevel: %s' % (self.wantobjects, tcl_version, tk_patchlevel)
        for (arg, res) in testcases:
            self.assertEqual(splitlist(arg), res, 'arg=%a, %s' % (arg, dbg_info))
        self.assertRaises(TclError, splitlist, '{')

    def test_split(self):
        if False:
            i = 10
            return i + 15
        split = self.interp.tk.split
        call = self.interp.tk.call
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', '\\bsplit\\b.*\\bsplitlist\\b', DeprecationWarning)
            self.assertRaises(TypeError, split)
            self.assertRaises(TypeError, split, 'a', 'b')
            self.assertRaises(TypeError, split, 2)
        testcases = [('2', '2'), ('', ''), ('{}', ''), ('""', ''), ('{', '{'), ('a\n b\t\r c\n ', ('a', 'b', 'c')), (b'a\n b\t\r c\n ', ('a', 'b', 'c')), ('a €', ('a', '€')), (b'a \xe2\x82\xac', ('a', '€')), (b'a\xc0\x80b', 'a\x00b'), (b'a\xc0\x80b c\xc0\x80d', ('a\x00b', 'c\x00d')), (b'{a\xc0\x80b c\xc0\x80d', '{a\x00b c\x00d'), ('a {b c}', ('a', ('b', 'c'))), ('a b\\ c', ('a', ('b', 'c'))), (('a', b'b c'), ('a', ('b', 'c'))), (('a', 'b c'), ('a', ('b', 'c'))), ('a 2', ('a', '2')), (('a', 2), ('a', 2)), ('a 3.4', ('a', '3.4')), (('a', 3.4), ('a', 3.4)), (('a', (2, 3.4)), ('a', (2, 3.4))), ((), ()), ([], ()), (['a', 'b c'], ('a', ('b', 'c'))), (['a', ['b', 'c']], ('a', ('b', 'c'))), (call('list', 1, '2', (3.4,)), (1, '2', (3.4,)) if self.wantobjects else ('1', '2', '3.4'))]
        if tcl_version >= (8, 5):
            if not self.wantobjects or get_tk_patchlevel() < (8, 5, 5):
                expected = ('12', '€', 'â\x82¬', '3.4')
            else:
                expected = (12, '€', b'\xe2\x82\xac', (3.4,))
            testcases += [(call('dict', 'create', 12, '€', b'\xe2\x82\xac', (3.4,)), expected)]
        for (arg, res) in testcases:
            with self.assertWarns(DeprecationWarning):
                self.assertEqual(split(arg), res, msg=arg)

    def test_splitdict(self):
        if False:
            for i in range(10):
                print('nop')
        splitdict = tkinter._splitdict
        tcl = self.interp.tk
        arg = '-a {1 2 3} -something foo status {}'
        self.assertEqual(splitdict(tcl, arg, False), {'-a': '1 2 3', '-something': 'foo', 'status': ''})
        self.assertEqual(splitdict(tcl, arg), {'a': '1 2 3', 'something': 'foo', 'status': ''})
        arg = ('-a', (1, 2, 3), '-something', 'foo', 'status', '{}')
        self.assertEqual(splitdict(tcl, arg, False), {'-a': (1, 2, 3), '-something': 'foo', 'status': '{}'})
        self.assertEqual(splitdict(tcl, arg), {'a': (1, 2, 3), 'something': 'foo', 'status': '{}'})
        self.assertRaises(RuntimeError, splitdict, tcl, '-a b -c ')
        self.assertRaises(RuntimeError, splitdict, tcl, ('-a', 'b', '-c'))
        arg = tcl.call('list', '-a', (1, 2, 3), '-something', 'foo', 'status', ())
        self.assertEqual(splitdict(tcl, arg), {'a': (1, 2, 3) if self.wantobjects else '1 2 3', 'something': 'foo', 'status': ''})
        if tcl_version >= (8, 5):
            arg = tcl.call('dict', 'create', '-a', (1, 2, 3), '-something', 'foo', 'status', ())
            if not self.wantobjects or get_tk_patchlevel() < (8, 5, 5):
                expected = {'a': '1 2 3', 'something': 'foo', 'status': ''}
            else:
                expected = {'a': (1, 2, 3), 'something': 'foo', 'status': ''}
            self.assertEqual(splitdict(tcl, arg), expected)

    def test_join(self):
        if False:
            i = 10
            return i + 15
        join = tkinter._join
        tcl = self.interp.tk

        def unpack(s):
            if False:
                print('Hello World!')
            return tcl.call('lindex', s, 0)

        def check(value):
            if False:
                for i in range(10):
                    print('nop')
            self.assertEqual(unpack(join([value])), value)
            self.assertEqual(unpack(join([value, 0])), value)
            self.assertEqual(unpack(unpack(join([[value]]))), value)
            self.assertEqual(unpack(unpack(join([[value, 0]]))), value)
            self.assertEqual(unpack(unpack(join([[value], 0]))), value)
            self.assertEqual(unpack(unpack(join([[value, 0], 0]))), value)
        check('')
        check('spam')
        check('sp am')
        check('sp\tam')
        check('sp\nam')
        check(' \t\n')
        check('{spam}')
        check('{sp am}')
        check('"spam"')
        check('"sp am"')
        check('{"spam"}')
        check('"{spam}"')
        check('sp\\am')
        check('"sp\\am"')
        check('"{}" "{}"')
        check('"\\')
        check('"{')
        check('"}')
        check('\n\\')
        check('\n{')
        check('\n}')
        check('\\\n')
        check('{\n')
        check('}\n')

    @support.cpython_only
    def test_new_tcl_obj(self):
        if False:
            for i in range(10):
                print('nop')
        support.check_disallow_instantiation(self, _tkinter.Tcl_Obj)
        support.check_disallow_instantiation(self, _tkinter.TkttType)
        support.check_disallow_instantiation(self, _tkinter.TkappType)

class BigmemTclTest(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.interp = Tcl()

    @support.cpython_only
    @unittest.skipUnless(INT_MAX < PY_SSIZE_T_MAX, 'needs UINT_MAX < SIZE_MAX')
    @support.bigmemtest(size=INT_MAX + 1, memuse=5, dry_run=False)
    def test_huge_string_call(self, size):
        if False:
            i = 10
            return i + 15
        value = ' ' * size
        self.assertRaises(OverflowError, self.interp.call, 'string', 'index', value, 0)

    @support.cpython_only
    @unittest.skipUnless(INT_MAX < PY_SSIZE_T_MAX, 'needs UINT_MAX < SIZE_MAX')
    @support.bigmemtest(size=INT_MAX + 1, memuse=2, dry_run=False)
    def test_huge_string_builtins(self, size):
        if False:
            return 10
        tk = self.interp.tk
        value = '1' + ' ' * size
        self.assertRaises(OverflowError, tk.getint, value)
        self.assertRaises(OverflowError, tk.getdouble, value)
        self.assertRaises(OverflowError, tk.getboolean, value)
        self.assertRaises(OverflowError, tk.eval, value)
        self.assertRaises(OverflowError, tk.evalfile, value)
        self.assertRaises(OverflowError, tk.record, value)
        self.assertRaises(OverflowError, tk.adderrorinfo, value)
        self.assertRaises(OverflowError, tk.setvar, value, 'x', 'a')
        self.assertRaises(OverflowError, tk.setvar, 'x', value, 'a')
        self.assertRaises(OverflowError, tk.unsetvar, value)
        self.assertRaises(OverflowError, tk.unsetvar, 'x', value)
        self.assertRaises(OverflowError, tk.adderrorinfo, value)
        self.assertRaises(OverflowError, tk.exprstring, value)
        self.assertRaises(OverflowError, tk.exprlong, value)
        self.assertRaises(OverflowError, tk.exprboolean, value)
        self.assertRaises(OverflowError, tk.splitlist, value)
        self.assertRaises(OverflowError, tk.split, value)
        self.assertRaises(OverflowError, tk.createcommand, value, max)
        self.assertRaises(OverflowError, tk.deletecommand, value)

    @support.cpython_only
    @unittest.skipUnless(INT_MAX < PY_SSIZE_T_MAX, 'needs UINT_MAX < SIZE_MAX')
    @support.bigmemtest(size=INT_MAX + 1, memuse=6, dry_run=False)
    def test_huge_string_builtins2(self, size):
        if False:
            print('Hello World!')
        tk = self.interp.tk
        value = '1' + ' ' * size
        self.assertRaises(OverflowError, tk.evalfile, value)
        self.assertRaises(OverflowError, tk.unsetvar, value)
        self.assertRaises(OverflowError, tk.unsetvar, 'x', value)

def setUpModule():
    if False:
        while True:
            i = 10
    if support.verbose:
        tcl = Tcl()
        print('patchlevel =', tcl.call('info', 'patchlevel'))
if __name__ == '__main__':
    unittest.main()