"""
Sample test cases defined using the standard library L{unittest.TestCase}
class which are used as data by test cases which are actually part of the
trial test suite to verify handling of handling of such cases.
"""
import unittest
from sys import exc_info
from twisted.python.failure import Failure

class PyUnitTest(unittest.TestCase):

    def test_pass(self):
        if False:
            i = 10
            return i + 15
        '\n        A passing test.\n        '

    def test_error(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        A test which raises an exception to cause an error.\n        '
        raise Exception('pyunit error')

    def test_fail(self):
        if False:
            while True:
                i = 10
        '\n        A test which uses L{unittest.TestCase.fail} to cause a failure.\n        '
        self.fail('pyunit failure')

    @unittest.skip('pyunit skip')
    def test_skip(self):
        if False:
            print('Hello World!')
        '\n        A test which uses the L{unittest.skip} decorator to cause a skip.\n        '

class _NonStringId:
    """
    A class that looks a little like a TestCase, but not enough so to
    actually be used as one.  This helps L{BrokenRunInfrastructure} use some
    interfaces incorrectly to provoke certain failure conditions.
    """

    def id(self) -> object:
        if False:
            print('Hello World!')
        return object()

class BrokenRunInfrastructure(unittest.TestCase):
    """
    A test suite that is broken at the level of integration between
    L{TestCase.run} and the results object.
    """

    def run(self, result):
        if False:
            i = 10
            return i + 15
        '\n        Override the normal C{run} behavior to pass the result object\n        along to the test method.  Each test method needs the result object so\n        that it can implement its particular kind of brokenness.\n        '
        return getattr(self, self._testMethodName)(result)

    def test_addSuccess(self, result):
        if False:
            print('Hello World!')
        '\n        Violate the L{TestResult.addSuccess} interface.\n        '
        result.addSuccess(_NonStringId())

    def test_addError(self, result):
        if False:
            return 10
        '\n        Violate the L{TestResult.addError} interface.\n        '
        try:
            raise Exception('test_addError')
        except BaseException:
            err = exc_info()
        result.addError(_NonStringId(), err)

    def test_addFailure(self, result):
        if False:
            print('Hello World!')
        '\n        Violate the L{TestResult.addFailure} interface.\n        '
        try:
            raise Exception('test_addFailure')
        except BaseException:
            err = exc_info()
        result.addFailure(_NonStringId(), err)

    def test_addSkip(self, result):
        if False:
            print('Hello World!')
        '\n        Violate the L{TestResult.addSkip} interface.\n        '
        result.addSkip(_NonStringId(), 'test_addSkip')

    def test_addExpectedFailure(self, result):
        if False:
            print('Hello World!')
        '\n        Violate the L{TestResult.addExpectedFailure} interface.\n        '
        try:
            raise Exception('test_addExpectedFailure')
        except BaseException:
            err = Failure()
        result.addExpectedFailure(_NonStringId(), err)

    def test_addUnexpectedSuccess(self, result):
        if False:
            while True:
                i = 10
        '\n        Violate the L{TestResult.addUnexpectedSuccess} interface.\n        '
        result.addUnexpectedSuccess(_NonStringId())