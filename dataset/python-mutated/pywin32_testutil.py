import gc
import os
import site
import sys
import unittest
import winerror

class LeakTestCase(unittest.TestCase):
    """An 'adaptor' which takes another test.  In debug builds we execute the
    test once to remove one-off side-effects, then capture the total
    reference count, then execute the test a few times.  If the total
    refcount at the end is greater than we first captured, we have a leak!

    In release builds the test is executed just once, as normal.

    Generally used automatically by the test runner - you can safely
    ignore this.
    """

    def __init__(self, real_test):
        if False:
            for i in range(10):
                print('nop')
        unittest.TestCase.__init__(self)
        self.real_test = real_test
        self.num_test_cases = 1
        self.num_leak_iters = 2
        if hasattr(sys, 'gettotalrefcount'):
            self.num_test_cases = self.num_test_cases + self.num_leak_iters

    def countTestCases(self):
        if False:
            return 10
        return self.num_test_cases

    def __call__(self, result=None):
        if False:
            i = 10
            return i + 15
        from pythoncom import _GetGatewayCount, _GetInterfaceCount
        gc.collect()
        ni = _GetInterfaceCount()
        ng = _GetGatewayCount()
        self.real_test(result)
        if result.shouldStop or not result.wasSuccessful():
            return
        self._do_leak_tests(result)
        gc.collect()
        lost_i = _GetInterfaceCount() - ni
        lost_g = _GetGatewayCount() - ng
        if lost_i or lost_g:
            msg = '%d interface objects and %d gateway objects leaked' % (lost_i, lost_g)
            exc = AssertionError(msg)
            result.addFailure(self.real_test, (exc.__class__, exc, None))

    def runTest(self):
        if False:
            while True:
                i = 10
        assert 0, 'not used'

    def _do_leak_tests(self, result=None):
        if False:
            for i in range(10):
                print('nop')
        try:
            gtrc = sys.gettotalrefcount
        except AttributeError:
            return
        gc.collect()
        trc = gtrc()
        for i in range(self.num_leak_iters):
            self.real_test(result)
            if result.shouldStop:
                break
        del i
        gc.collect()
        lost = (gtrc() - trc) // self.num_leak_iters
        if lost < 0:
            msg = 'LeakTest: %s appeared to gain %d references!!' % (self.real_test, -lost)
            result.addFailure(self.real_test, (AssertionError, msg, None))
        if lost > 0:
            msg = 'LeakTest: %s lost %d references' % (self.real_test, lost)
            exc = AssertionError(msg)
            result.addFailure(self.real_test, (exc.__class__, exc, None))

class TestLoader(unittest.TestLoader):

    def loadTestsFromTestCase(self, testCaseClass):
        if False:
            print('Hello World!')
        'Return a suite of all tests cases contained in testCaseClass'
        leak_tests = []
        for name in self.getTestCaseNames(testCaseClass):
            real_test = testCaseClass(name)
            leak_test = self._getTestWrapper(real_test)
            leak_tests.append(leak_test)
        return self.suiteClass(leak_tests)

    def fixupTestsForLeakTests(self, test):
        if False:
            return 10
        if isinstance(test, unittest.TestSuite):
            test._tests = [self.fixupTestsForLeakTests(t) for t in test._tests]
            return test
        else:
            return self._getTestWrapper(test)

    def _getTestWrapper(self, test):
        if False:
            return 10
        no_leak_tests = getattr(test, 'no_leak_tests', False)
        if no_leak_tests:
            print("Test says it doesn't want leak tests!")
            return test
        return LeakTestCase(test)

    def loadTestsFromModule(self, mod):
        if False:
            while True:
                i = 10
        if hasattr(mod, 'suite'):
            tests = mod.suite()
        else:
            tests = unittest.TestLoader.loadTestsFromModule(self, mod)
        return self.fixupTestsForLeakTests(tests)

    def loadTestsFromName(self, name, module=None):
        if False:
            i = 10
            return i + 15
        test = unittest.TestLoader.loadTestsFromName(self, name, module)
        if isinstance(test, unittest.TestSuite):
            pass
        elif isinstance(test, unittest.TestCase):
            test = self._getTestWrapper(test)
        else:
            print('XXX - what is', test)
        return test
non_admin_error_codes = [winerror.ERROR_ACCESS_DENIED, winerror.ERROR_PRIVILEGE_NOT_HELD]
_is_admin = None

def check_is_admin():
    if False:
        for i in range(10):
            print('nop')
    global _is_admin
    if _is_admin is None:
        import pythoncom
        from win32com.shell.shell import IsUserAnAdmin
        try:
            _is_admin = IsUserAnAdmin()
        except pythoncom.com_error as exc:
            if exc.hresult != winerror.E_NOTIMPL:
                raise
            _is_admin = True
    return _is_admin

def find_test_fixture(basename, extra_dir='.'):
    if False:
        print('Hello World!')
    candidates = [os.path.dirname(sys.argv[0]), extra_dir, '.']
    for candidate in candidates:
        fname = os.path.join(candidate, basename)
        if os.path.isfile(fname):
            return fname
    else:
        this_file = os.path.normcase(os.path.abspath(sys.argv[0]))
        dirs_to_check = site.getsitepackages()[:]
        if site.USER_SITE:
            dirs_to_check.append(site.USER_SITE)
        for d in dirs_to_check:
            d = os.path.normcase(d)
            if os.path.commonprefix([this_file, d]) == d:
                raise TestSkipped(f"Can't find test fixture '{fname}'")
        raise RuntimeError(f"Can't find test fixture '{fname}'")

class TestSkipped(Exception):
    pass
try:
    TextTestResult = unittest._TextTestResult
except AttributeError:
    TextTestResult = unittest.TextTestResult

class TestResult(TextTestResult):

    def __init__(self, *args, **kw):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kw)
        self.skips = {}

    def addError(self, test, err):
        if False:
            print('Hello World!')
        "Called when an error has occurred. 'err' is a tuple of values as\n        returned by sys.exc_info().\n        "
        import pywintypes
        exc_val = err[1]
        if isinstance(exc_val, pywintypes.error) and exc_val.winerror in non_admin_error_codes and (not check_is_admin()):
            exc_val = TestSkipped(exc_val)
        elif isinstance(exc_val, pywintypes.com_error) and exc_val.hresult in [winerror.CO_E_CLASSSTRING, winerror.REGDB_E_CLASSNOTREG, winerror.TYPE_E_LIBNOTREGISTERED]:
            exc_val = TestSkipped(exc_val)
        elif isinstance(exc_val, NotImplementedError):
            exc_val = TestSkipped(NotImplementedError)
        if isinstance(exc_val, TestSkipped):
            reason = exc_val.args[0]
            try:
                reason = tuple(reason.args)
            except (AttributeError, TypeError):
                pass
            self.skips.setdefault(reason, 0)
            self.skips[reason] += 1
            if self.showAll:
                self.stream.writeln(f'SKIP ({reason})')
            elif self.dots:
                self.stream.write('S')
                self.stream.flush()
            return
        super().addError(test, err)

    def printErrors(self):
        if False:
            print('Hello World!')
        super().printErrors()
        for (reason, num_skipped) in self.skips.items():
            self.stream.writeln('SKIPPED: %d tests - %s' % (num_skipped, reason))

class TestRunner(unittest.TextTestRunner):

    def _makeResult(self):
        if False:
            for i in range(10):
                print('nop')
        return TestResult(self.stream, self.descriptions, self.verbosity)

class TestProgram(unittest.TestProgram):

    def runTests(self):
        if False:
            print('Hello World!')
        self.testRunner = TestRunner(verbosity=self.verbosity)
        unittest.TestProgram.runTests(self)

def testmain(*args, **kw):
    if False:
        while True:
            i = 10
    new_kw = kw.copy()
    if 'testLoader' not in new_kw:
        new_kw['testLoader'] = TestLoader()
    program_class = new_kw.get('testProgram', TestProgram)
    program_class(*args, **new_kw)