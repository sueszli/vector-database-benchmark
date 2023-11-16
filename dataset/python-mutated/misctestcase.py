__author__ = 'Yaroslav Halchenko'
__copyright__ = 'Copyright (c) 2013 Yaroslav Halchenko'
__license__ = 'GPL'
import logging
import os
import sys
import unittest
import tempfile
import shutil
import fnmatch
from glob import glob
from io import StringIO
from .utils import LogCaptureTestCase, logSys as DefLogSys
from ..helpers import formatExceptionInfo, mbasename, TraceBack, FormatterWithTraceBack, getLogger, getVerbosityFormat, splitwords, uni_decode, uni_string
from ..server.mytime import MyTime

class HelpersTest(unittest.TestCase):

    def testFormatExceptionInfoBasic(self):
        if False:
            i = 10
            return i + 15
        try:
            raise ValueError('Very bad exception')
        except:
            (name, args) = formatExceptionInfo()
            self.assertEqual(name, 'ValueError')
            self.assertEqual(args, 'Very bad exception')

    def testFormatExceptionConvertArgs(self):
        if False:
            return 10
        try:
            raise ValueError('Very bad', None)
        except:
            (name, args) = formatExceptionInfo()
            self.assertEqual(name, 'ValueError')
            self.assertEqual(args, "('Very bad', None)")

    def testsplitwords(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(splitwords(None), [])
        self.assertEqual(splitwords(''), [])
        self.assertEqual(splitwords('  '), [])
        self.assertEqual(splitwords('1'), ['1'])
        self.assertEqual(splitwords(' 1 2 '), ['1', '2'])
        self.assertEqual(splitwords(' 1, 2 , '), ['1', '2'])
        self.assertEqual(splitwords(' 1\n  2'), ['1', '2'])
        self.assertEqual(splitwords(' 1\n  2, 3'), ['1', '2', '3'])
        self.assertEqual(splitwords('\t1\t  2,\r\n 3\n'), ['1', '2', '3'])

def _sh_call(cmd):
    if False:
        i = 10
        return i + 15
    import subprocess
    ret = subprocess.check_output(cmd, shell=True)
    return uni_decode(ret).rstrip()

def _getSysPythonVersion():
    if False:
        while True:
            i = 10
    return _sh_call("fail2ban-python -c 'import sys; print(tuple(sys.version_info))'")

class SetupTest(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        super(SetupTest, self).setUp()
        unittest.F2B.SkipIfFast()
        setup = os.path.join(os.path.dirname(__file__), '..', '..', 'setup.py')
        self.setup = os.path.exists(setup) and setup or None
        if not self.setup:
            raise unittest.SkipTest('Seems to be running not out of source distribution -- cannot locate setup.py')
        sysVer = _getSysPythonVersion()
        if sysVer != str(tuple(sys.version_info)):
            raise unittest.SkipTest('Seems to be running with python distribution %s -- install can be tested only with system distribution %s' % (str(tuple(sys.version_info)), sysVer))

    def testSetupInstallDryRun(self):
        if False:
            i = 10
            return i + 15
        if not self.setup:
            return
        tmp = tempfile.mkdtemp()
        supdbgout = ' >/dev/null 2>&1' if unittest.F2B.log_level >= logging.DEBUG else ''
        try:
            os.system('%s %s --dry-run install --root=%s%s' % (sys.executable, self.setup, tmp, supdbgout))
            self.assertTrue(not os.listdir(tmp))
        finally:
            shutil.rmtree(tmp)

    def testSetupInstallRoot(self):
        if False:
            return 10
        if not self.setup:
            return
        tmp = tempfile.mkdtemp()
        remove_build = not os.path.exists('build')
        supdbgout = ' >/dev/null' if unittest.F2B.log_level >= logging.DEBUG else ''
        try:
            self.assertEqual(os.system('%s %s install --root=%s%s' % (sys.executable, self.setup, tmp, supdbgout)), 0)

            def strippath(l):
                if False:
                    while True:
                        i = 10
                return [x[len(tmp) + 1:] for x in l]
            got = strippath(sorted(glob('%s/*' % tmp)))
            need = ['etc', 'usr', 'var']
            if set(need).difference(got):

                def recursive_glob(treeroot, pattern):
                    if False:
                        return 10
                    results = []
                    for (base, dirs, files) in os.walk(treeroot):
                        goodfiles = fnmatch.filter(dirs + files, pattern)
                        results.extend((os.path.join(base, f) for f in goodfiles))
                    return results
                files = {}
                for missing in set(got).difference(need):
                    missing_full = os.path.join(tmp, missing)
                    files[missing] = os.path.exists(missing_full) and strippath(recursive_glob(missing_full, '*')) or None
                self.assertEqual(got, need, msg='Got: %s Needed: %s under %s. Files under new paths: %s' % (got, need, tmp, files))
            for f in ('etc/fail2ban/fail2ban.conf', 'etc/fail2ban/jail.conf'):
                self.assertTrue(os.path.exists(os.path.join(tmp, f)), msg="Can't find %s" % f)
            installedPath = _sh_call('find ' + tmp + ' -name fail2ban-python').split('\n')
            self.assertTrue(len(installedPath) > 0)
            for installedPath in installedPath:
                self.assertEqual(os.path.realpath(installedPath), os.path.realpath(sys.executable))
        finally:
            shutil.rmtree(tmp)
            os.system('%s %s clean --all%s' % (sys.executable, self.setup, supdbgout + ' 2>&1' if supdbgout else ''))
            if remove_build and os.path.exists('build'):
                shutil.rmtree('build')

class TestsUtilsTest(LogCaptureTestCase):

    def testmbasename(self):
        if False:
            return 10
        self.assertEqual(mbasename('sample.py'), 'sample')
        self.assertEqual(mbasename('/long/path/sample.py'), 'sample')
        self.assertEqual(mbasename('/long/path/__init__.py'), 'path.__init__')
        self.assertEqual(mbasename('/long/path/base.py'), 'path.base')
        self.assertEqual(mbasename('/long/path/base'), 'path.base')

    def testUniConverters(self):
        if False:
            return 10
        self.assertRaises(Exception, uni_decode, b'test' if sys.version_info >= (3,) else 'test', 'f2b-test::non-existing-encoding')
        uni_decode(b'test\xcf' if sys.version_info >= (3,) else 'testÏ')
        uni_string(b'test\xcf')
        uni_string('testÏ')
        if sys.version_info < (3,) and 'PyPy' not in sys.version:
            uni_string('testÏ')

    def testSafeLogging(self):
        if False:
            while True:
                i = 10
        logSys = DefLogSys

        class Test:

            def __init__(self, err=1):
                if False:
                    for i in range(10):
                        print('nop')
                self.err = err

            def __repr__(self):
                if False:
                    return 10
                if self.err:
                    raise Exception('no represenation for test!')
                else:
                    return 'conv-error (òðåòèé), unterminated utf Ï'
        test = Test()
        logSys.log(logging.NOTICE, 'test 1a: %r', test)
        self.assertLogged('Traceback', 'no represenation for test!')
        self.pruneLog()
        logSys.notice('test 1b: %r', test)
        self.assertLogged('Traceback', 'no represenation for test!')
        self.pruneLog('[phase 2] test error conversion by encoding %s' % sys.getdefaultencoding())
        test = Test(0)
        logSys.log(logging.NOTICE, 'test 2a: %r, %s', test, test)
        self.assertLogged('test 2a', 'Error by logging handler', all=False)
        logSys.notice('test 2b: %r, %s', test, test)
        self.assertLogged('test 2b', 'Error by logging handler', all=False)
        self.pruneLog('[phase 3] test unexpected error in handler')

        class _ErrorHandler(logging.Handler):

            def handle(self, record):
                if False:
                    i = 10
                    return i + 15
                raise Exception('error in handler test!')
        _org_handler = logSys.handlers
        try:
            logSys.handlers = list(logSys.handlers)
            logSys.handlers += [_ErrorHandler()]
            logSys.log(logging.NOTICE, 'test 3a')
            logSys.notice('test 3b')
        finally:
            logSys.handlers = _org_handler
        self.pruneLog('OK')

    def testTraceBack(self):
        if False:
            for i in range(10):
                print('nop')
        for compress in (True, False):
            tb = TraceBack(compress=compress)

            def func_raise():
                if False:
                    for i in range(10):
                        print('nop')
                raise ValueError()

            def deep_function(i):
                if False:
                    i = 10
                    return i + 15
                if i:
                    deep_function(i - 1)
                else:
                    func_raise()
            try:
                print(deep_function(3))
            except ValueError:
                s = tb()
            if not 'fail2ban-testcases' in s:
                self.assertIn('>', s)
            elif not 'coverage' in s:
                self.assertNotIn('>', s)
            self.assertIn(':', s)

    def _testAssertionErrorRE(self, regexp, fun, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaisesRegex(AssertionError, regexp, fun, *args, **kwargs)

    def testExtendedAssertRaisesRE(self):
        if False:
            i = 10
            return i + 15

        def _key_err(msg):
            if False:
                i = 10
                return i + 15
            raise KeyError(msg)
        self.assertRaises(KeyError, self._testAssertionErrorRE, '^failed$', _key_err, 'failed')
        self.assertRaises(AssertionError, self._testAssertionErrorRE, '^failed$', self.fail, '__failed__')
        self._testAssertionErrorRE('failed.* does not match .*__failed__', lambda : self._testAssertionErrorRE('^failed$', self.fail, '__failed__'))
        self.assertRaises(AssertionError, self._testAssertionErrorRE, '', int, 1)
        self._testAssertionErrorRE('0 AssertionError not raised X.* does not match .*AssertionError not raised', lambda : self._testAssertionErrorRE('^0 AssertionError not raised X$', lambda : self._testAssertionErrorRE('', int, 1)))

    def testExtendedAssertMethods(self):
        if False:
            print('Hello World!')
        self.assertIn('a', ['a', 'b', 'c', 'd'])
        self.assertIn('a', ('a', 'b', 'c', 'd'))
        self.assertIn('a', 'cba')
        self.assertIn('a', (c for c in 'cba' if c != 'b'))
        self.assertNotIn('a', ['b', 'c', 'd'])
        self.assertNotIn('a', ('b', 'c', 'd'))
        self.assertNotIn('a', 'cbd')
        self.assertNotIn('a', (c.upper() for c in 'cba' if c != 'b'))
        self._testAssertionErrorRE("'a' unexpectedly found in 'cba'", self.assertNotIn, 'a', 'cba')
        self._testAssertionErrorRE('1 unexpectedly found in \\[0, 1, 2\\]', self.assertNotIn, 1, range(3))
        self._testAssertionErrorRE("'A' unexpectedly found in \\['C', 'A'\\]", self.assertNotIn, 'A', (c.upper() for c in 'cba' if c != 'b'))
        self._testAssertionErrorRE("'a' was not found in 'xyz'", self.assertIn, 'a', 'xyz')
        self._testAssertionErrorRE('5 was not found in \\[0, 1, 2\\]', self.assertIn, 5, range(3))
        self._testAssertionErrorRE("'A' was not found in \\['C', 'B'\\]", self.assertIn, 'A', (c.upper() for c in 'cba' if c != 'a'))
        logSys = DefLogSys
        self.pruneLog()
        logSys.debug('test "xyz"')
        self.assertLogged('test "xyz"')
        self.assertLogged('test', 'xyz', all=True)
        self.assertNotLogged('test', 'zyx', all=False)
        self.assertNotLogged('test_zyx', 'zyx', all=True)
        self.assertLogged('test', 'zyx', all=False)
        self.pruneLog()
        logSys.debug('xxxx "xxx"')
        self.assertNotLogged('test "xyz"')
        self.assertNotLogged('test', 'xyz', all=False)
        self.assertNotLogged('test', 'xyz', 'zyx', all=True)
        (orgfast, unittest.F2B.fast) = (unittest.F2B.fast, False)
        self.assertFalse(isinstance(unittest.F2B.maxWaitTime(True), bool))
        self.assertEqual(unittest.F2B.maxWaitTime(lambda : 50)(), 50)
        self.assertEqual(unittest.F2B.maxWaitTime(25), 25)
        self.assertEqual(unittest.F2B.maxWaitTime(25.0), 25.0)
        unittest.F2B.fast = True
        try:
            self.assertEqual(unittest.F2B.maxWaitTime(lambda : 50)(), 50)
            self.assertEqual(unittest.F2B.maxWaitTime(25), 2.5)
            self.assertEqual(unittest.F2B.maxWaitTime(25.0), 25.0)
        finally:
            unittest.F2B.fast = orgfast
        self.assertFalse(unittest.F2B.maxWaitTime(False))
        self.pruneLog()
        logSys.debug('test "xyz"')
        self._testAssertionErrorRE('.* was found in the log', self.assertNotLogged, 'test "xyz"')
        self._testAssertionErrorRE('All of the .* were found present in the log', self.assertNotLogged, 'test "xyz"', 'test')
        self._testAssertionErrorRE('was found in the log', self.assertNotLogged, 'test', 'xyz', all=True)
        self._testAssertionErrorRE('was not found in the log', self.assertLogged, 'test', 'zyx', all=True)
        self._testAssertionErrorRE('was not found in the log, waited 1e-06', self.assertLogged, 'test', 'zyx', all=True, wait=1e-06)
        self._testAssertionErrorRE('None among .* was found in the log', self.assertLogged, 'test_zyx', 'zyx', all=False)
        self._testAssertionErrorRE('None among .* was found in the log, waited 1e-06', self.assertLogged, 'test_zyx', 'zyx', all=False, wait=1e-06)
        self._testAssertionErrorRE('All of the .* were found present in the log', self.assertNotLogged, 'test', 'xyz', all=False)
        self.assertDictEqual({'A': [1, 2]}, {'A': [1, 2]})
        self.assertRaises(AssertionError, self.assertDictEqual, {'A': [1, 2]}, {'A': [2, 1]})
        self.assertSortedEqual(['A', 'B'], ['B', 'A'])
        self.assertSortedEqual([['A', 'B']], [['B', 'A']], level=2)
        self.assertSortedEqual([['A', 'B']], [['B', 'A']], nestedOnly=False)
        self.assertRaises(AssertionError, lambda : self.assertSortedEqual([['A', 'B']], [['B', 'A']], level=1, nestedOnly=True))
        self.assertSortedEqual({'A': ['A', 'B']}, {'A': ['B', 'A']}, nestedOnly=False)
        self.assertRaises(AssertionError, lambda : self.assertSortedEqual({'A': ['A', 'B']}, {'A': ['B', 'A']}, level=1, nestedOnly=True))
        self.assertSortedEqual(['Z', {'A': ['B', 'C'], 'B': ['E', 'F']}], [{'B': ['F', 'E'], 'A': ['C', 'B']}, 'Z'], nestedOnly=False)
        self.assertSortedEqual(['Z', {'A': ['B', 'C'], 'B': ['E', 'F']}], [{'B': ['F', 'E'], 'A': ['C', 'B']}, 'Z'], level=-1)
        self.assertRaises(AssertionError, lambda : self.assertSortedEqual(['Z', {'A': ['B', 'C'], 'B': ['E', 'F']}], [{'B': ['F', 'E'], 'A': ['C', 'B']}, 'Z'], nestedOnly=True))
        self.assertSortedEqual((0, [['A1'], ['A2', 'A1'], []]), (0, [['A1'], ['A1', 'A2'], []]))
        self.assertSortedEqual(list('ABC'), list('CBA'))
        self.assertRaises(AssertionError, self.assertSortedEqual, ['ABC'], ['CBA'])
        self.assertRaises(AssertionError, self.assertSortedEqual, [['ABC']], [['CBA']])
        self._testAssertionErrorRE("\\['A'\\] != \\['C', 'B'\\]", self.assertSortedEqual, ['A'], ['C', 'B'])
        self._testAssertionErrorRE("\\['A', 'B'\\] != \\['B', 'C'\\]", self.assertSortedEqual, ['A', 'B'], ['C', 'B'])

    def testVerbosityFormat(self):
        if False:
            return 10
        self.assertEqual(getVerbosityFormat(1), '%(asctime)s %(name)-24s[%(process)d]: %(levelname)-7s %(message)s')
        self.assertEqual(getVerbosityFormat(1, padding=False), '%(asctime)s %(name)s[%(process)d]: %(levelname)s %(message)s')
        self.assertEqual(getVerbosityFormat(1, addtime=False, padding=False), '%(name)s[%(process)d]: %(levelname)s %(message)s')

    def testFormatterWithTraceBack(self):
        if False:
            for i in range(10):
                print('nop')
        strout = StringIO()
        Formatter = FormatterWithTraceBack
        fmt = ' %(tb)s | %(tbc)s : %(message)s'
        logSys = getLogger('fail2ban_tests')
        out = logging.StreamHandler(strout)
        out.setFormatter(Formatter(fmt))
        logSys.addHandler(out)
        logSys.error('XXX')
        s = strout.getvalue()
        self.assertTrue(s.rstrip().endswith(': XXX'))
        pindex = s.index('|')
        self.assertTrue(pindex > 10)
        self.assertEqual(s[:pindex], s[pindex + 1:pindex * 2 + 1])

    def testLazyLogging(self):
        if False:
            i = 10
            return i + 15
        logSys = DefLogSys
        logSys.debug('lazy logging: %r', unittest.F2B.log_lazy)
        logSys.notice('test', 1, 2, 3)
        self.assertLogged('not all arguments converted')

class MyTimeTest(unittest.TestCase):

    def testStr2Seconds(self):
        if False:
            i = 10
            return i + 15
        str2sec = MyTime.str2seconds
        self.assertEqual(str2sec('1y6mo30w15d12h35m25s'), 66821725)
        self.assertEqual(str2sec('2yy 3mo 4ww 10dd 5hh 30mm 20ss'), 74307620)
        self.assertEqual(str2sec('2 years 3 months 4 weeks 10 days 5 hours 30 minutes 20 seconds'), 74307620)
        self.assertEqual(str2sec('1 year + 1 month - 1 week + 1 day'), 33669000)
        self.assertEqual(str2sec('2 * 0.5 yea + 1*1 mon - 3*1/3 wee + 2/2 day - (2*12 hou 3*20 min 80 sec) '), 33578920.0)
        self.assertEqual(str2sec('2*.5y+1*1mo-3*1/3w+2/2d-(2*12h3*20m80s) '), 33578920.0)
        self.assertEqual(str2sec('1ye -2mo -3we -4da -5ho -6mi -7se'), 24119633)
        self.assertEqual(float(str2sec('1 month')) / 60 / 60 / 24, 30.4375)
        self.assertEqual(float(str2sec('1 year')) / 60 / 60 / 24, 365.25)

    def testSec2Str(self):
        if False:
            print('Hello World!')
        sec2str = lambda s: str(MyTime.seconds2str(s))
        self.assertEqual(sec2str(86400 * 390), '1y 3w 4d')
        self.assertEqual(sec2str(86400 * 368), '1y 3d')
        self.assertEqual(sec2str(86400 * 365.49), '1y')
        self.assertEqual(sec2str(86400 * 15), '2w 1d')
        self.assertEqual(sec2str(86400 * 14 - 10), '2w')
        self.assertEqual(sec2str(86400 * 2 + 3600 * 7 + 60 * 15), '2d 7h 15m')
        self.assertEqual(sec2str(86400 * 2 + 3599), '2d 1h')
        self.assertEqual(sec2str(3600 * 3.52), '3h 31m')
        self.assertEqual(sec2str(3600 * 2 - 5), '2h')
        self.assertEqual(sec2str(3600 - 5), '1h')
        self.assertEqual(sec2str(3600 - 10), '59m 50s')
        self.assertEqual(sec2str(59), '59s')
        self.assertEqual(sec2str(0), '0')