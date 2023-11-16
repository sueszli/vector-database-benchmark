"""
Infrastructure for test running and suites.
"""
import doctest
import gc
import unittest as pyunit
from typing import Iterator, Union
from zope.interface import implementer
from twisted.python import components
from twisted.trial import itrial, reporter
from twisted.trial._synctest import _logObserver

class TestSuite(pyunit.TestSuite):
    """
    Extend the standard library's C{TestSuite} with a consistently overrideable
    C{run} method.
    """

    def run(self, result):
        if False:
            i = 10
            return i + 15
        '\n        Call C{run} on every member of the suite.\n        '
        for test in self._tests:
            if result.shouldStop:
                break
            test(result)
        return result

@implementer(itrial.ITestCase)
class TestDecorator(components.proxyForInterface(itrial.ITestCase, '_originalTest')):
    """
    Decorator for test cases.

    @param _originalTest: The wrapped instance of test.
    @type _originalTest: A provider of L{itrial.ITestCase}
    """

    def __call__(self, result):
        if False:
            while True:
                i = 10
        '\n        Run the unit test.\n\n        @param result: A TestResult object.\n        '
        return self.run(result)

    def run(self, result):
        if False:
            return 10
        '\n        Run the unit test.\n\n        @param result: A TestResult object.\n        '
        return self._originalTest.run(reporter._AdaptedReporter(result, self.__class__))

def _clearSuite(suite):
    if False:
        i = 10
        return i + 15
    '\n    Clear all tests from C{suite}.\n\n    This messes with the internals of C{suite}. In particular, it assumes that\n    the suite keeps all of its tests in a list in an instance variable called\n    C{_tests}.\n    '
    suite._tests = []

def decorate(test, decorator):
    if False:
        print('Hello World!')
    '\n    Decorate all test cases in C{test} with C{decorator}.\n\n    C{test} can be a test case or a test suite. If it is a test suite, then the\n    structure of the suite is preserved.\n\n    L{decorate} tries to preserve the class of the test suites it finds, but\n    assumes the presence of the C{_tests} attribute on the suite.\n\n    @param test: The C{TestCase} or C{TestSuite} to decorate.\n\n    @param decorator: A unary callable used to decorate C{TestCase}s.\n\n    @return: A decorated C{TestCase} or a C{TestSuite} containing decorated\n        C{TestCase}s.\n    '
    try:
        tests = iter(test)
    except TypeError:
        return decorator(test)
    _clearSuite(test)
    for case in tests:
        test.addTest(decorate(case, decorator))
    return test

class _PyUnitTestCaseAdapter(TestDecorator):
    """
    Adapt from pyunit.TestCase to ITestCase.
    """

class _BrokenIDTestCaseAdapter(_PyUnitTestCaseAdapter):
    """
    Adapter for pyunit-style C{TestCase} subclasses that have undesirable id()
    methods. That is C{unittest.FunctionTestCase} and C{unittest.DocTestCase}.
    """

    def id(self):
        if False:
            return 10
        '\n        Return the fully-qualified Python name of the doctest.\n        '
        testID = self._originalTest.shortDescription()
        if testID is not None:
            return testID
        return self._originalTest.id()

class _ForceGarbageCollectionDecorator(TestDecorator):
    """
    Forces garbage collection to be run before and after the test. Any errors
    logged during the post-test collection are added to the test result as
    errors.
    """

    def run(self, result):
        if False:
            for i in range(10):
                print('nop')
        gc.collect()
        TestDecorator.run(self, result)
        _logObserver._add()
        gc.collect()
        for error in _logObserver.getErrors():
            result.addError(self, error)
        _logObserver.flushErrors()
        _logObserver._remove()
components.registerAdapter(_PyUnitTestCaseAdapter, pyunit.TestCase, itrial.ITestCase)
components.registerAdapter(_BrokenIDTestCaseAdapter, pyunit.FunctionTestCase, itrial.ITestCase)
_docTestCase = getattr(doctest, 'DocTestCase', None)
if _docTestCase:
    components.registerAdapter(_BrokenIDTestCaseAdapter, _docTestCase, itrial.ITestCase)

def _iterateTests(testSuiteOrCase: Union[pyunit.TestCase, pyunit.TestSuite]) -> Iterator[itrial.ITestCase]:
    if False:
        print('Hello World!')
    '\n    Iterate through all of the test cases in C{testSuiteOrCase}.\n    '
    try:
        suite = iter(testSuiteOrCase)
    except TypeError:
        yield testSuiteOrCase
    else:
        for test in suite:
            yield from _iterateTests(test)