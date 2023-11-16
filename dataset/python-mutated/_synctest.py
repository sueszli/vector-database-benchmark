"""
Things likely to be used by writers of unit tests.

Maintainer: Jonathan Lange
"""
import inspect
import os
import sys
import tempfile
import types
import unittest as pyunit
import warnings
from dis import findlinestarts as _findlinestarts
from typing import Any, Callable, Coroutine, Generator, Iterable, List, NoReturn, Optional, Tuple, Type, TypeVar, Union
from unittest import SkipTest
from attrs import frozen
from typing_extensions import ParamSpec
from twisted.internet.defer import Deferred, ensureDeferred
from twisted.python import failure, log, monkey
from twisted.python.deprecate import DEPRECATION_WARNING_FORMAT, getDeprecationWarningString, getVersionString, warnAboutFunction
from twisted.python.reflect import fullyQualifiedName
from twisted.python.util import runWithWarningsSuppressed
from twisted.trial import itrial, util
_P = ParamSpec('_P')
T = TypeVar('T')

class FailTest(AssertionError):
    """
    Raised to indicate the current test has failed to pass.
    """

@frozen
class Todo:
    """
    Internal object used to mark a L{TestCase} as 'todo'. Tests marked 'todo'
    are reported differently in Trial L{TestResult}s. If todo'd tests fail,
    they do not fail the suite and the errors are reported in a separate
    category. If todo'd tests succeed, Trial L{TestResult}s will report an
    unexpected success.

    @ivar reason: A string explaining why the test is marked 'todo'

    @ivar errors: An iterable of exception types that the test is expected to
        raise. If one of these errors is raised by the test, it will be
        trapped. Raising any other kind of error will fail the test.  If
        L{None} then all errors will be trapped.
    """
    reason: str
    errors: Optional[Iterable[Type[BaseException]]] = None

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        return f'<Todo reason={self.reason!r} errors={self.errors!r}>'

    def expected(self, failure):
        if False:
            i = 10
            return i + 15
        '\n        @param failure: A L{twisted.python.failure.Failure}.\n\n        @return: C{True} if C{failure} is expected, C{False} otherwise.\n        '
        if self.errors is None:
            return True
        for error in self.errors:
            if failure.check(error):
                return True
        return False

def makeTodo(value: Union[str, Tuple[Union[Type[BaseException], Iterable[Type[BaseException]]], str]]) -> Todo:
    if False:
        for i in range(10):
            print('nop')
    '\n    Return a L{Todo} object built from C{value}.\n\n    If C{value} is a string, return a Todo that expects any exception with\n    C{value} as a reason. If C{value} is a tuple, the second element is used\n    as the reason and the first element as the excepted error(s).\n\n    @param value: A string or a tuple of C{(errors, reason)}, where C{errors}\n    is either a single exception class or an iterable of exception classes.\n\n    @return: A L{Todo} object.\n    '
    if isinstance(value, str):
        return Todo(reason=value)
    if isinstance(value, tuple):
        (errors, reason) = value
        if isinstance(errors, type):
            iterableErrors: Iterable[Type[BaseException]] = [errors]
        else:
            iterableErrors = errors
        return Todo(reason=reason, errors=iterableErrors)

class _Warning:
    """
    A L{_Warning} instance represents one warning emitted through the Python
    warning system (L{warnings}).  This is used to insulate callers of
    L{_collectWarnings} from changes to the Python warnings system which might
    otherwise require changes to the warning objects that function passes to
    the observer object it accepts.

    @ivar message: The string which was passed as the message parameter to
        L{warnings.warn}.

    @ivar category: The L{Warning} subclass which was passed as the category
        parameter to L{warnings.warn}.

    @ivar filename: The name of the file containing the definition of the code
        object which was C{stacklevel} frames above the call to
        L{warnings.warn}, where C{stacklevel} is the value of the C{stacklevel}
        parameter passed to L{warnings.warn}.

    @ivar lineno: The source line associated with the active instruction of the
        code object object which was C{stacklevel} frames above the call to
        L{warnings.warn}, where C{stacklevel} is the value of the C{stacklevel}
        parameter passed to L{warnings.warn}.
    """

    def __init__(self, message, category, filename, lineno):
        if False:
            while True:
                i = 10
        self.message = message
        self.category = category
        self.filename = filename
        self.lineno = lineno

def _setWarningRegistryToNone(modules):
    if False:
        print('Hello World!')
    '\n    Disable the per-module cache for every module found in C{modules}, typically\n    C{sys.modules}.\n\n    @param modules: Dictionary of modules, typically sys.module dict\n    '
    for v in list(modules.values()):
        if v is not None:
            try:
                v.__warningregistry__ = None
            except BaseException:
                pass

def _collectWarnings(observeWarning, f, *args, **kwargs):
    if False:
        return 10
    '\n    Call C{f} with C{args} positional arguments and C{kwargs} keyword arguments\n    and collect all warnings which are emitted as a result in a list.\n\n    @param observeWarning: A callable which will be invoked with a L{_Warning}\n        instance each time a warning is emitted.\n\n    @return: The return value of C{f(*args, **kwargs)}.\n    '

    def showWarning(message, category, filename, lineno, file=None, line=None):
        if False:
            i = 10
            return i + 15
        assert isinstance(message, Warning)
        observeWarning(_Warning(str(message), category, filename, lineno))
    _setWarningRegistryToNone(sys.modules)
    origFilters = warnings.filters[:]
    origShow = warnings.showwarning
    warnings.simplefilter('always')
    try:
        warnings.showwarning = showWarning
        result = f(*args, **kwargs)
    finally:
        warnings.filters[:] = origFilters
        warnings.showwarning = origShow
    return result

class UnsupportedTrialFeature(Exception):
    """A feature of twisted.trial was used that pyunit cannot support."""

class PyUnitResultAdapter:
    """
    Wrap a C{TestResult} from the standard library's C{unittest} so that it
    supports the extended result types from Trial, and also supports
    L{twisted.python.failure.Failure}s being passed to L{addError} and
    L{addFailure}.
    """

    def __init__(self, original):
        if False:
            return 10
        '\n        @param original: A C{TestResult} instance from C{unittest}.\n        '
        self.original = original

    def _exc_info(self, err):
        if False:
            for i in range(10):
                print('nop')
        return util.excInfoOrFailureToExcInfo(err)

    def startTest(self, method):
        if False:
            i = 10
            return i + 15
        self.original.startTest(method)

    def stopTest(self, method):
        if False:
            i = 10
            return i + 15
        self.original.stopTest(method)

    def addFailure(self, test, fail):
        if False:
            i = 10
            return i + 15
        self.original.addFailure(test, self._exc_info(fail))

    def addError(self, test, error):
        if False:
            return 10
        self.original.addError(test, self._exc_info(error))

    def _unsupported(self, test, feature, info):
        if False:
            i = 10
            return i + 15
        self.original.addFailure(test, (UnsupportedTrialFeature, UnsupportedTrialFeature(feature, info), None))

    def addSkip(self, test, reason):
        if False:
            while True:
                i = 10
        '\n        Report the skip as a failure.\n        '
        self.original.addSkip(test, reason)

    def addUnexpectedSuccess(self, test, todo=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Report the unexpected success as a failure.\n        '
        self._unsupported(test, 'unexpected success', todo)

    def addExpectedFailure(self, test, error):
        if False:
            print('Hello World!')
        '\n        Report the expected failure (i.e. todo) as a failure.\n        '
        self._unsupported(test, 'expected failure', error)

    def addSuccess(self, test):
        if False:
            while True:
                i = 10
        self.original.addSuccess(test)

    def upDownError(self, method, error, warn, printStatus):
        if False:
            return 10
        pass

class _AssertRaisesContext:
    """
    A helper for implementing C{assertRaises}.  This is a context manager and a
    helper method to support the non-context manager version of
    C{assertRaises}.

    @ivar _testCase: See C{testCase} parameter of C{__init__}

    @ivar _expected: See C{expected} parameter of C{__init__}

    @ivar _returnValue: The value returned by the callable being tested (only
        when not being used as a context manager).

    @ivar _expectedName: A short string describing the expected exception
        (usually the name of the exception class).

    @ivar exception: The exception which was raised by the function being
        tested (if it raised one).
    """

    def __init__(self, testCase, expected):
        if False:
            i = 10
            return i + 15
        '\n        @param testCase: The L{TestCase} instance which is used to raise a\n            test-failing exception when that is necessary.\n\n        @param expected: The exception type expected to be raised.\n        '
        self._testCase = testCase
        self._expected = expected
        self._returnValue = None
        try:
            self._expectedName = self._expected.__name__
        except AttributeError:
            self._expectedName = str(self._expected)

    def _handle(self, obj):
        if False:
            i = 10
            return i + 15
        '\n        Call the given object using this object as a context manager.\n\n        @param obj: The object to call and which is expected to raise some\n            exception.\n        @type obj: L{object}\n\n        @return: Whatever exception is raised by C{obj()}.\n        @rtype: L{BaseException}\n        '
        with self as context:
            self._returnValue = obj()
        return context.exception

    def __enter__(self):
        if False:
            print('Hello World!')
        return self

    def __exit__(self, exceptionType, exceptionValue, traceback):
        if False:
            i = 10
            return i + 15
        '\n        Check exit exception against expected exception.\n        '
        if exceptionType is None:
            self._testCase.fail('{} not raised ({} returned)'.format(self._expectedName, self._returnValue))
        if not isinstance(exceptionValue, exceptionType):
            if isinstance(exceptionValue, tuple):
                exceptionValue = exceptionType(*exceptionValue)
            else:
                exceptionValue = exceptionType(exceptionValue)
        self.exception = exceptionValue
        if not issubclass(exceptionType, self._expected):
            reason = failure.Failure(exceptionValue, exceptionType, traceback)
            self._testCase.fail('{} raised instead of {}:\n {}'.format(fullyQualifiedName(exceptionType), self._expectedName, reason.getTraceback()))
        return True

class _Assertions(pyunit.TestCase):
    """
    Replaces many of the built-in TestCase assertions. In general, these
    assertions provide better error messages and are easier to use in
    callbacks.
    """

    def fail(self, msg: Optional[object]=None) -> NoReturn:
        if False:
            for i in range(10):
                print('nop')
        '\n        Absolutely fail the test.  Do not pass go, do not collect $200.\n\n        @param msg: the message that will be displayed as the reason for the\n        failure\n        '
        raise self.failureException(msg)

    def assertFalse(self, condition, msg=None):
        if False:
            return 10
        '\n        Fail the test if C{condition} evaluates to True.\n\n        @param condition: any object that defines __nonzero__\n        '
        super().assertFalse(condition, msg)
        return condition
    assertNot = assertFalse
    failUnlessFalse = assertFalse
    failIf = assertFalse

    def assertTrue(self, condition, msg=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Fail the test if C{condition} evaluates to False.\n\n        @param condition: any object that defines __nonzero__\n        '
        super().assertTrue(condition, msg)
        return condition
    assert_ = assertTrue
    failUnlessTrue = assertTrue
    failUnless = assertTrue

    def assertRaises(self, exception, f=None, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Fail the test unless calling the function C{f} with the given\n        C{args} and C{kwargs} raises C{exception}. The failure will report\n        the traceback and call stack of the unexpected exception.\n\n        @param exception: exception type that is to be expected\n        @param f: the function to call\n\n        @return: If C{f} is L{None}, a context manager which will make an\n            assertion about the exception raised from the suite it manages.  If\n            C{f} is not L{None}, the exception raised by C{f}.\n\n        @raise self.failureException: Raised if the function call does\n            not raise an exception or if it raises an exception of a\n            different type.\n        '
        context = _AssertRaisesContext(self, exception)
        if f is None:
            return context
        return context._handle(lambda : f(*args, **kwargs))
    failUnlessRaises = assertRaises

    def assertEqual(self, first, second, msg=None):
        if False:
            i = 10
            return i + 15
        "\n        Fail the test if C{first} and C{second} are not equal.\n\n        @param msg: A string describing the failure that's included in the\n            exception.\n        "
        super().assertEqual(first, second, msg)
        return first
    failUnlessEqual = assertEqual
    failUnlessEquals = assertEqual
    assertEquals = assertEqual

    def assertIs(self, first, second, msg=None):
        if False:
            while True:
                i = 10
        "\n        Fail the test if C{first} is not C{second}.  This is an\n        obect-identity-equality test, not an object equality\n        (i.e. C{__eq__}) test.\n\n        @param msg: if msg is None, then the failure message will be\n        '%r is not %r' % (first, second)\n        "
        if first is not second:
            raise self.failureException(msg or f'{first!r} is not {second!r}')
        return first
    failUnlessIdentical = assertIs
    assertIdentical = assertIs

    def assertIsNot(self, first, second, msg=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        Fail the test if C{first} is C{second}.  This is an\n        obect-identity-equality test, not an object equality\n        (i.e. C{__eq__}) test.\n\n        @param msg: if msg is None, then the failure message will be\n        '%r is %r' % (first, second)\n        "
        if first is second:
            raise self.failureException(msg or f'{first!r} is {second!r}')
        return first
    failIfIdentical = assertIsNot
    assertNotIdentical = assertIsNot

    def assertNotEqual(self, first, second, msg=None):
        if False:
            i = 10
            return i + 15
        "\n        Fail the test if C{first} == C{second}.\n\n        @param msg: if msg is None, then the failure message will be\n        '%r == %r' % (first, second)\n        "
        if not first != second:
            raise self.failureException(msg or f'{first!r} == {second!r}')
        return first
    assertNotEquals = assertNotEqual
    failIfEquals = assertNotEqual
    failIfEqual = assertNotEqual

    def assertIn(self, containee, container, msg=None):
        if False:
            while True:
                i = 10
        "\n        Fail the test if C{containee} is not found in C{container}.\n\n        @param containee: the value that should be in C{container}\n        @param container: a sequence type, or in the case of a mapping type,\n                          will follow semantics of 'if key in dict.keys()'\n        @param msg: if msg is None, then the failure message will be\n                    '%r not in %r' % (first, second)\n        "
        if containee not in container:
            raise self.failureException(msg or f'{containee!r} not in {container!r}')
        return containee
    failUnlessIn = assertIn

    def assertNotIn(self, containee, container, msg=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        Fail the test if C{containee} is found in C{container}.\n\n        @param containee: the value that should not be in C{container}\n        @param container: a sequence type, or in the case of a mapping type,\n                          will follow semantics of 'if key in dict.keys()'\n        @param msg: if msg is None, then the failure message will be\n                    '%r in %r' % (first, second)\n        "
        if containee in container:
            raise self.failureException(msg or f'{containee!r} in {container!r}')
        return containee
    failIfIn = assertNotIn

    def assertNotAlmostEqual(self, first, second, places=7, msg=None, delta=None):
        if False:
            while True:
                i = 10
        '\n        Fail if the two objects are equal as determined by their\n        difference rounded to the given number of decimal places\n        (default 7) and comparing to zero.\n\n        @note: decimal places (from zero) is usually not the same\n               as significant digits (measured from the most\n               significant digit).\n\n        @note: included for compatibility with PyUnit test cases\n        '
        if round(second - first, places) == 0:
            raise self.failureException(msg or f'{first!r} == {second!r} within {places!r} places')
        return first
    assertNotAlmostEquals = assertNotAlmostEqual
    failIfAlmostEqual = assertNotAlmostEqual
    failIfAlmostEquals = assertNotAlmostEqual

    def assertAlmostEqual(self, first, second, places=7, msg=None, delta=None):
        if False:
            print('Hello World!')
        '\n        Fail if the two objects are unequal as determined by their\n        difference rounded to the given number of decimal places\n        (default 7) and comparing to zero.\n\n        @note: decimal places (from zero) is usually not the same\n               as significant digits (measured from the most\n               significant digit).\n\n        @note: included for compatibility with PyUnit test cases\n        '
        if round(second - first, places) != 0:
            raise self.failureException(msg or f'{first!r} != {second!r} within {places!r} places')
        return first
    assertAlmostEquals = assertAlmostEqual
    failUnlessAlmostEqual = assertAlmostEqual

    def assertApproximates(self, first, second, tolerance, msg=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        Fail if C{first} - C{second} > C{tolerance}\n\n        @param msg: if msg is None, then the failure message will be\n                    '%r ~== %r' % (first, second)\n        "
        if abs(first - second) > tolerance:
            raise self.failureException(msg or f'{first} ~== {second}')
        return first
    failUnlessApproximates = assertApproximates

    def assertSubstring(self, substring, astring, msg=None):
        if False:
            print('Hello World!')
        '\n        Fail if C{substring} does not exist within C{astring}.\n        '
        return self.failUnlessIn(substring, astring, msg)
    failUnlessSubstring = assertSubstring

    def assertNotSubstring(self, substring, astring, msg=None):
        if False:
            return 10
        '\n        Fail if C{astring} contains C{substring}.\n        '
        return self.failIfIn(substring, astring, msg)
    failIfSubstring = assertNotSubstring

    def assertWarns(self, category, message, filename, f, *args, **kwargs):
        if False:
            while True:
                i = 10
        "\n        Fail if the given function doesn't generate the specified warning when\n        called. It calls the function, checks the warning, and forwards the\n        result of the function if everything is fine.\n\n        @param category: the category of the warning to check.\n        @param message: the output message of the warning to check.\n        @param filename: the filename where the warning should come from.\n        @param f: the function which is supposed to generate the warning.\n        @type f: any callable.\n        @param args: the arguments to C{f}.\n        @param kwargs: the keywords arguments to C{f}.\n\n        @return: the result of the original function C{f}.\n        "
        warningsShown = []
        result = _collectWarnings(warningsShown.append, f, *args, **kwargs)
        if not warningsShown:
            self.fail('No warnings emitted')
        first = warningsShown[0]
        for other in warningsShown[1:]:
            if (other.message, other.category) != (first.message, first.category):
                self.fail("Can't handle different warnings")
        self.assertEqual(first.message, message)
        self.assertIdentical(first.category, category)
        self.assertTrue(filename.startswith(first.filename), f'Warning in {first.filename!r}, expected {filename!r}')
        return result
    failUnlessWarns = assertWarns

    def assertIsInstance(self, instance, classOrTuple, message=None):
        if False:
            i = 10
            return i + 15
        '\n        Fail if C{instance} is not an instance of the given class or of\n        one of the given classes.\n\n        @param instance: the object to test the type (first argument of the\n            C{isinstance} call).\n        @type instance: any.\n        @param classOrTuple: the class or classes to test against (second\n            argument of the C{isinstance} call).\n        @type classOrTuple: class, type, or tuple.\n\n        @param message: Custom text to include in the exception text if the\n            assertion fails.\n        '
        if not isinstance(instance, classOrTuple):
            if message is None:
                suffix = ''
            else:
                suffix = ': ' + message
            self.fail(f'{instance!r} is not an instance of {classOrTuple}{suffix}')
    failUnlessIsInstance = assertIsInstance

    def assertNotIsInstance(self, instance, classOrTuple):
        if False:
            print('Hello World!')
        '\n        Fail if C{instance} is an instance of the given class or of one of the\n        given classes.\n\n        @param instance: the object to test the type (first argument of the\n            C{isinstance} call).\n        @type instance: any.\n        @param classOrTuple: the class or classes to test against (second\n            argument of the C{isinstance} call).\n        @type classOrTuple: class, type, or tuple.\n        '
        if isinstance(instance, classOrTuple):
            self.fail(f'{instance!r} is an instance of {classOrTuple}')
    failIfIsInstance = assertNotIsInstance

    def successResultOf(self, deferred: Union[Coroutine[Deferred[T], Any, T], Generator[Deferred[T], Any, T], Deferred[T]]) -> T:
        if False:
            i = 10
            return i + 15
        '\n        Return the current success result of C{deferred} or raise\n        C{self.failureException}.\n\n        @param deferred: A L{Deferred<twisted.internet.defer.Deferred>} or\n            I{coroutine} which has a success result.\n\n            For a L{Deferred<twisted.internet.defer.Deferred>} this means\n            L{Deferred.callback<twisted.internet.defer.Deferred.callback>} or\n            L{Deferred.errback<twisted.internet.defer.Deferred.errback>} has\n            been called on it and it has reached the end of its callback chain\n            and the last callback or errback returned a\n            non-L{failure.Failure}.\n\n            For a I{coroutine} this means all awaited values have a success\n            result.\n\n        @raise SynchronousTestCase.failureException: If the\n            L{Deferred<twisted.internet.defer.Deferred>} has no result or has a\n            failure result.\n\n        @return: The result of C{deferred}.\n        '
        deferred = ensureDeferred(deferred)
        results: List[Union[T, failure.Failure]] = []
        deferred.addBoth(results.append)
        if not results:
            self.fail('Success result expected on {!r}, found no result instead'.format(deferred))
        result = results[0]
        if isinstance(result, failure.Failure):
            self.fail('Success result expected on {!r}, found failure result instead:\n{}'.format(deferred, result.getTraceback()))
        return result

    def failureResultOf(self, deferred, *expectedExceptionTypes):
        if False:
            return 10
        '\n        Return the current failure result of C{deferred} or raise\n        C{self.failureException}.\n\n        @param deferred: A L{Deferred<twisted.internet.defer.Deferred>} which\n            has a failure result.  This means\n            L{Deferred.callback<twisted.internet.defer.Deferred.callback>} or\n            L{Deferred.errback<twisted.internet.defer.Deferred.errback>} has\n            been called on it and it has reached the end of its callback chain\n            and the last callback or errback raised an exception or returned a\n            L{failure.Failure}.\n        @type deferred: L{Deferred<twisted.internet.defer.Deferred>}\n\n        @param expectedExceptionTypes: Exception types to expect - if\n            provided, and the exception wrapped by the failure result is\n            not one of the types provided, then this test will fail.\n\n        @raise SynchronousTestCase.failureException: If the\n            L{Deferred<twisted.internet.defer.Deferred>} has no result, has a\n            success result, or has an unexpected failure result.\n\n        @return: The failure result of C{deferred}.\n        @rtype: L{failure.Failure}\n        '
        deferred = ensureDeferred(deferred)
        result = []
        deferred.addBoth(result.append)
        if not result:
            self.fail('Failure result expected on {!r}, found no result instead'.format(deferred))
        result = result[0]
        if not isinstance(result, failure.Failure):
            self.fail('Failure result expected on {!r}, found success result ({!r}) instead'.format(deferred, result))
        if expectedExceptionTypes and (not result.check(*expectedExceptionTypes)):
            expectedString = ' or '.join(['.'.join((t.__module__, t.__name__)) for t in expectedExceptionTypes])
            self.fail('Failure of type ({}) expected on {!r}, found type {!r} instead: {}'.format(expectedString, deferred, result.type, result.getTraceback()))
        return result

    def assertNoResult(self, deferred):
        if False:
            i = 10
            return i + 15
        '\n        Assert that C{deferred} does not have a result at this point.\n\n        If the assertion succeeds, then the result of C{deferred} is left\n        unchanged. Otherwise, any L{failure.Failure} result is swallowed.\n\n        @param deferred: A L{Deferred<twisted.internet.defer.Deferred>} without\n            a result.  This means that neither\n            L{Deferred.callback<twisted.internet.defer.Deferred.callback>} nor\n            L{Deferred.errback<twisted.internet.defer.Deferred.errback>} has\n            been called, or that the\n            L{Deferred<twisted.internet.defer.Deferred>} is waiting on another\n            L{Deferred<twisted.internet.defer.Deferred>} for a result.\n        @type deferred: L{Deferred<twisted.internet.defer.Deferred>}\n\n        @raise SynchronousTestCase.failureException: If the\n            L{Deferred<twisted.internet.defer.Deferred>} has a result.\n        '
        deferred = ensureDeferred(deferred)
        result = []

        def cb(res):
            if False:
                print('Hello World!')
            result.append(res)
            return res
        deferred.addBoth(cb)
        if result:
            deferred.addErrback(lambda _: None)
            self.fail('No result expected on {!r}, found {!r} instead'.format(deferred, result[0]))

class _LogObserver:
    """
    Observes the Twisted logs and catches any errors.

    @ivar _errors: A C{list} of L{Failure} instances which were received as
        error events from the Twisted logging system.

    @ivar _added: A C{int} giving the number of times C{_add} has been called
        less the number of times C{_remove} has been called; used to only add
        this observer to the Twisted logging since once, regardless of the
        number of calls to the add method.

    @ivar _ignored: A C{list} of exception types which will not be recorded.
    """

    def __init__(self):
        if False:
            print('Hello World!')
        self._errors = []
        self._added = 0
        self._ignored = []

    def _add(self):
        if False:
            return 10
        if self._added == 0:
            log.addObserver(self.gotEvent)
        self._added += 1

    def _remove(self):
        if False:
            i = 10
            return i + 15
        self._added -= 1
        if self._added == 0:
            log.removeObserver(self.gotEvent)

    def _ignoreErrors(self, *errorTypes):
        if False:
            return 10
        '\n        Do not store any errors with any of the given types.\n        '
        self._ignored.extend(errorTypes)

    def _clearIgnores(self):
        if False:
            while True:
                i = 10
        '\n        Stop ignoring any errors we might currently be ignoring.\n        '
        self._ignored = []

    def flushErrors(self, *errorTypes):
        if False:
            for i in range(10):
                print('nop')
        '\n        Flush errors from the list of caught errors. If no arguments are\n        specified, remove all errors. If arguments are specified, only remove\n        errors of those types from the stored list.\n        '
        if errorTypes:
            flushed = []
            remainder = []
            for f in self._errors:
                if f.check(*errorTypes):
                    flushed.append(f)
                else:
                    remainder.append(f)
            self._errors = remainder
        else:
            flushed = self._errors
            self._errors = []
        return flushed

    def getErrors(self):
        if False:
            print('Hello World!')
        '\n        Return a list of errors caught by this observer.\n        '
        return self._errors

    def gotEvent(self, event):
        if False:
            i = 10
            return i + 15
        '\n        The actual observer method. Called whenever a message is logged.\n\n        @param event: A dictionary containing the log message. Actual\n        structure undocumented (see source for L{twisted.python.log}).\n        '
        if event.get('isError', False) and 'failure' in event:
            f = event['failure']
            if len(self._ignored) == 0 or not f.check(*self._ignored):
                self._errors.append(f)
_logObserver = _LogObserver()

class SynchronousTestCase(_Assertions):
    """
    A unit test. The atom of the unit testing universe.

    This class extends C{unittest.TestCase} from the standard library.  A number
    of convenient testing helpers are added, including logging and warning
    integration, monkey-patching support, and more.

    To write a unit test, subclass C{SynchronousTestCase} and define a method
    (say, 'test_foo') on the subclass. To run the test, instantiate your
    subclass with the name of the method, and call L{run} on the instance,
    passing a L{TestResult} object.

    The C{trial} script will automatically find any C{SynchronousTestCase}
    subclasses defined in modules beginning with 'test_' and construct test
    cases for all methods beginning with 'test'.

    If an error is logged during the test run, the test will fail with an
    error. See L{log.err}.

    @ivar failureException: An exception class, defaulting to C{FailTest}. If
    the test method raises this exception, it will be reported as a failure,
    rather than an exception. All of the assertion methods raise this if the
    assertion fails.

    @ivar skip: L{None} or a string explaining why this test is to be
    skipped. If defined, the test will not be run. Instead, it will be
    reported to the result object as 'skipped' (if the C{TestResult} supports
    skipping).

    @ivar todo: L{None}, a string or a tuple of C{(errors, reason)} where
    C{errors} is either an exception class or an iterable of exception
    classes, and C{reason} is a string. See L{Todo} or L{makeTodo} for more
    information.

    @ivar suppress: L{None} or a list of tuples of C{(args, kwargs)} to be
    passed to C{warnings.filterwarnings}. Use these to suppress warnings
    raised in a test. Useful for testing deprecated code. See also
    L{util.suppress}.
    """
    failureException = FailTest

    def __init__(self, methodName='runTest'):
        if False:
            print('Hello World!')
        super().__init__(methodName)
        self._passed = False
        self._cleanups = []
        self._testMethodName = methodName
        testMethod = getattr(self, methodName)
        self._parents = [testMethod, self, sys.modules.get(self.__class__.__module__)]

    def __eq__(self, other: object) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Override the comparison defined by the base TestCase which considers\n        instances of the same class with the same _testMethodName to be\n        equal.  Since trial puts TestCase instances into a set, that\n        definition of comparison makes it impossible to run the same test\n        method twice.  Most likely, trial should stop using a set to hold\n        tests, but until it does, this is necessary on Python 2.6. -exarkun\n        '
        if isinstance(other, SynchronousTestCase):
            return self is other
        else:
            return NotImplemented

    def __hash__(self):
        if False:
            i = 10
            return i + 15
        return hash((self.__class__, self._testMethodName))

    def shortDescription(self):
        if False:
            i = 10
            return i + 15
        desc = super().shortDescription()
        if desc is None:
            return self._testMethodName
        return desc

    def getSkip(self) -> Tuple[bool, Optional[str]]:
        if False:
            return 10
        '\n        Return the skip reason set on this test, if any is set. Checks on the\n        instance first, then the class, then the module, then packages. As\n        soon as it finds something with a C{skip} attribute, returns that in\n        a tuple (L{True}, L{str}).\n        If the C{skip} attribute does not exist, look for C{__unittest_skip__}\n        and C{__unittest_skip_why__} attributes which are set by the standard\n        library L{unittest.skip} function.\n        Returns (L{False}, L{None}) if it cannot find anything.\n        See L{TestCase} docstring for more details.\n        '
        skipReason = util.acquireAttribute(self._parents, 'skip', None)
        doSkip = skipReason is not None
        if skipReason is None:
            doSkip = getattr(self, '__unittest_skip__', False)
            if doSkip:
                skipReason = getattr(self, '__unittest_skip_why__', '')
        return (doSkip, skipReason)

    def getTodo(self):
        if False:
            return 10
        '\n        Return a L{Todo} object if the test is marked todo. Checks on the\n        instance first, then the class, then the module, then packages. As\n        soon as it finds something with a C{todo} attribute, returns that.\n        Returns L{None} if it cannot find anything. See L{TestCase} docstring\n        for more details.\n        '
        todo = util.acquireAttribute(self._parents, 'todo', None)
        if todo is None:
            return None
        return makeTodo(todo)

    def runTest(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        If no C{methodName} argument is passed to the constructor, L{run} will\n        treat this method as the thing with the actual test inside.\n        '

    def run(self, result):
        if False:
            while True:
                i = 10
        '\n        Run the test case, storing the results in C{result}.\n\n        First runs C{setUp} on self, then runs the test method (defined in the\n        constructor), then runs C{tearDown}.  As with the standard library\n        L{unittest.TestCase}, the return value of these methods is disregarded.\n        In particular, returning a L{Deferred<twisted.internet.defer.Deferred>}\n        has no special additional consequences.\n\n        @param result: A L{TestResult} object.\n        '
        log.msg('--> %s <--' % self.id())
        new_result = itrial.IReporter(result, None)
        if new_result is None:
            result = PyUnitResultAdapter(result)
        else:
            result = new_result
        result.startTest(self)
        (doSkip, skipReason) = self.getSkip()
        if doSkip:
            result.addSkip(self, skipReason)
            result.stopTest(self)
            return
        self._passed = False
        self._warnings = []
        self._installObserver()
        _collectWarnings(self._warnings.append, self._runFixturesAndTest, result)
        for w in self.flushWarnings():
            try:
                warnings.warn_explicit(**w)
            except BaseException:
                result.addError(self, failure.Failure())
        result.stopTest(self)

    def addCleanup(self, f: Callable[_P, object], *args: _P.args, **kwargs: _P.kwargs) -> None:
        if False:
            while True:
                i = 10
        '\n        Add the given function to a list of functions to be called after the\n        test has run, but before C{tearDown}.\n\n        Functions will be run in reverse order of being added. This helps\n        ensure that tear down complements set up.\n\n        As with all aspects of L{SynchronousTestCase}, Deferreds are not\n        supported in cleanup functions.\n        '
        self._cleanups.append((f, args, kwargs))

    def patch(self, obj, attribute, value):
        if False:
            print('Hello World!')
        '\n        Monkey patch an object for the duration of the test.\n\n        The monkey patch will be reverted at the end of the test using the\n        L{addCleanup} mechanism.\n\n        The L{monkey.MonkeyPatcher} is returned so that users can restore and\n        re-apply the monkey patch within their tests.\n\n        @param obj: The object to monkey patch.\n        @param attribute: The name of the attribute to change.\n        @param value: The value to set the attribute to.\n        @return: A L{monkey.MonkeyPatcher} object.\n        '
        monkeyPatch = monkey.MonkeyPatcher((obj, attribute, value))
        monkeyPatch.patch()
        self.addCleanup(monkeyPatch.restore)
        return monkeyPatch

    def flushLoggedErrors(self, *errorTypes):
        if False:
            i = 10
            return i + 15
        '\n        Remove stored errors received from the log.\n\n        C{TestCase} stores each error logged during the run of the test and\n        reports them as errors during the cleanup phase (after C{tearDown}).\n\n        @param errorTypes: If unspecified, flush all errors. Otherwise, only\n        flush errors that match the given types.\n\n        @return: A list of failures that have been removed.\n        '
        return self._observer.flushErrors(*errorTypes)

    def flushWarnings(self, offendingFunctions=None):
        if False:
            print('Hello World!')
        "\n        Remove stored warnings from the list of captured warnings and return\n        them.\n\n        @param offendingFunctions: If L{None}, all warnings issued during the\n            currently running test will be flushed.  Otherwise, only warnings\n            which I{point} to a function included in this list will be flushed.\n            All warnings include a filename and source line number; if these\n            parts of a warning point to a source line which is part of a\n            function, then the warning I{points} to that function.\n        @type offendingFunctions: L{None} or L{list} of functions or methods.\n\n        @raise ValueError: If C{offendingFunctions} is not L{None} and includes\n            an object which is not a L{types.FunctionType} or\n            L{types.MethodType} instance.\n\n        @return: A C{list}, each element of which is a C{dict} giving\n            information about one warning which was flushed by this call.  The\n            keys of each C{dict} are:\n\n                - C{'message'}: The string which was passed as the I{message}\n                  parameter to L{warnings.warn}.\n\n                - C{'category'}: The warning subclass which was passed as the\n                  I{category} parameter to L{warnings.warn}.\n\n                - C{'filename'}: The name of the file containing the definition\n                  of the code object which was C{stacklevel} frames above the\n                  call to L{warnings.warn}, where C{stacklevel} is the value of\n                  the C{stacklevel} parameter passed to L{warnings.warn}.\n\n                - C{'lineno'}: The source line associated with the active\n                  instruction of the code object object which was C{stacklevel}\n                  frames above the call to L{warnings.warn}, where\n                  C{stacklevel} is the value of the C{stacklevel} parameter\n                  passed to L{warnings.warn}.\n        "
        if offendingFunctions is None:
            toFlush = self._warnings[:]
            self._warnings[:] = []
        else:
            toFlush = []
            for aWarning in self._warnings:
                for aFunction in offendingFunctions:
                    if not isinstance(aFunction, (types.FunctionType, types.MethodType)):
                        raise ValueError(f'{aFunction!r} is not a function or method')
                    aModule = sys.modules[aFunction.__module__]
                    filename = inspect.getabsfile(aModule)
                    if filename != os.path.normcase(aWarning.filename):
                        continue
                    lineNumbers = [lineNumber for (_, lineNumber) in _findlinestarts(aFunction.__code__)]
                    if not min(lineNumbers) <= aWarning.lineno <= max(lineNumbers):
                        continue
                    toFlush.append(aWarning)
                    break
            list(map(self._warnings.remove, toFlush))
        return [{'message': w.message, 'category': w.category, 'filename': w.filename, 'lineno': w.lineno} for w in toFlush]

    def getDeprecatedModuleAttribute(self, moduleName, name, version, message=None):
        if False:
            print('Hello World!')
        '\n        Retrieve a module attribute which should have been deprecated,\n        and assert that we saw the appropriate deprecation warning.\n\n        @type moduleName: C{str}\n        @param moduleName: Fully-qualified Python name of the module containing\n            the deprecated attribute; if called from the same module as the\n            attributes are being deprecated in, using the C{__name__} global can\n            be helpful\n\n        @type name: C{str}\n        @param name: Attribute name which we expect to be deprecated\n\n        @param version: The first L{version<twisted.python.versions.Version>} that\n            the module attribute was deprecated.\n\n        @type message: C{str}\n        @param message: (optional) The expected deprecation message for the module attribute\n\n        @return: The given attribute from the named module\n\n        @raise FailTest: if no warnings were emitted on getattr, or if the\n            L{DeprecationWarning} emitted did not produce the canonical\n            please-use-something-else message that is standard for Twisted\n            deprecations according to the given version and replacement.\n\n        @since: Twisted 21.2.0\n        '
        fqpn = moduleName + '.' + name
        module = sys.modules[moduleName]
        attr = getattr(module, name)
        warningsShown = self.flushWarnings([self.getDeprecatedModuleAttribute])
        if len(warningsShown) == 0:
            self.fail(f'{fqpn} is not deprecated.')
        observedWarning = warningsShown[0]['message']
        expectedWarning = DEPRECATION_WARNING_FORMAT % {'fqpn': fqpn, 'version': getVersionString(version)}
        if message is not None:
            expectedWarning = expectedWarning + ': ' + message
        self.assert_(observedWarning.startswith(expectedWarning), f'Expected {observedWarning!r} to start with {expectedWarning!r}')
        return attr

    def callDeprecated(self, version, f, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Call a function that should have been deprecated at a specific version\n        and in favor of a specific alternative, and assert that it was thusly\n        deprecated.\n\n        @param version: A 2-sequence of (since, replacement), where C{since} is\n            a the first L{version<incremental.Version>} that C{f}\n            should have been deprecated since, and C{replacement} is a suggested\n            replacement for the deprecated functionality, as described by\n            L{twisted.python.deprecate.deprecated}.  If there is no suggested\n            replacement, this parameter may also be simply a\n            L{version<incremental.Version>} by itself.\n\n        @param f: The deprecated function to call.\n\n        @param args: The arguments to pass to C{f}.\n\n        @param kwargs: The keyword arguments to pass to C{f}.\n\n        @return: Whatever C{f} returns.\n\n        @raise Exception: Whatever C{f} raises.  If any exception is\n            raised by C{f}, though, no assertions will be made about emitted\n            deprecations.\n\n        @raise FailTest: if no warnings were emitted by C{f}, or if the\n            L{DeprecationWarning} emitted did not produce the canonical\n            please-use-something-else message that is standard for Twisted\n            deprecations according to the given version and replacement.\n        '
        result = f(*args, **kwargs)
        warningsShown = self.flushWarnings([self.callDeprecated])
        try:
            info = list(version)
        except TypeError:
            since = version
            replacement = None
        else:
            [since, replacement] = info
        if len(warningsShown) == 0:
            self.fail(f'{f!r} is not deprecated.')
        observedWarning = warningsShown[0]['message']
        expectedWarning = getDeprecationWarningString(f, since, replacement=replacement)
        self.assertEqual(expectedWarning, observedWarning)
        return result

    def mktemp(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a new path name which can be used for a new file or directory.\n\n        The result is a relative path that is guaranteed to be unique within the\n        current working directory.  The parent of the path will exist, but the\n        path will not.\n\n        For a temporary directory call os.mkdir on the path.  For a temporary\n        file just create the file (e.g. by opening the path for writing and then\n        closing it).\n\n        @return: The newly created path\n        @rtype: C{str}\n        '
        MAX_FILENAME = 32
        base = os.path.join(self.__class__.__module__[:MAX_FILENAME], self.__class__.__name__[:MAX_FILENAME], self._testMethodName[:MAX_FILENAME])
        if not os.path.exists(base):
            os.makedirs(base)
        dirname = os.path.relpath(tempfile.mkdtemp('', '', base))
        return os.path.join(dirname, 'temp')

    def _getSuppress(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns any warning suppressions set for this test. Checks on the\n        instance first, then the class, then the module, then packages. As\n        soon as it finds something with a C{suppress} attribute, returns that.\n        Returns any empty list (i.e. suppress no warnings) if it cannot find\n        anything. See L{TestCase} docstring for more details.\n        '
        return util.acquireAttribute(self._parents, 'suppress', [])

    def _getSkipReason(self, method, skip):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the reason to use for skipping a test method.\n\n        @param method: The method which produced the skip.\n        @param skip: A L{unittest.SkipTest} instance raised by C{method}.\n        '
        if len(skip.args) > 0:
            return skip.args[0]
        warnAboutFunction(method, 'Do not raise unittest.SkipTest with no arguments! Give a reason for skipping tests!')
        return skip

    def _run(self, suppress, todo, method, result):
        if False:
            for i in range(10):
                print('nop')
        '\n        Run a single method, either a test method or fixture.\n\n        @param suppress: Any warnings to suppress, as defined by the C{suppress}\n            attribute on this method, test case, or the module it is defined in.\n\n        @param todo: Any expected failure or failures, as defined by the C{todo}\n            attribute on this method, test case, or the module it is defined in.\n\n        @param method: The method to run.\n\n        @param result: The TestResult instance to which to report results.\n\n        @return: C{True} if the method fails and no further method/fixture calls\n            should be made, C{False} otherwise.\n        '
        if inspect.isgeneratorfunction(method):
            exc = TypeError('{!r} is a generator function and therefore will never run'.format(method))
            result.addError(self, failure.Failure(exc))
            return True
        try:
            runWithWarningsSuppressed(suppress, method)
        except SkipTest as e:
            result.addSkip(self, self._getSkipReason(method, e))
        except BaseException:
            reason = failure.Failure()
            if todo is None or not todo.expected(reason):
                if reason.check(self.failureException):
                    addResult = result.addFailure
                else:
                    addResult = result.addError
                addResult(self, reason)
            else:
                result.addExpectedFailure(self, reason, todo)
        else:
            return False
        return True

    def _runFixturesAndTest(self, result):
        if False:
            i = 10
            return i + 15
        '\n        Run C{setUp}, a test method, test cleanups, and C{tearDown}.\n\n        @param result: The TestResult instance to which to report results.\n        '
        suppress = self._getSuppress()
        try:
            if self._run(suppress, None, self.setUp, result):
                return
            todo = self.getTodo()
            method = getattr(self, self._testMethodName)
            failed = self._run(suppress, todo, method, result)
        finally:
            self._runCleanups(result)
        if todo and (not failed):
            result.addUnexpectedSuccess(self, todo)
        if self._run(suppress, None, self.tearDown, result):
            failed = True
        for error in self._observer.getErrors():
            result.addError(self, error)
            failed = True
        self._observer.flushErrors()
        self._removeObserver()
        if not (failed or todo):
            result.addSuccess(self)

    def _runCleanups(self, result):
        if False:
            for i in range(10):
                print('nop')
        '\n        Synchronously run any cleanups which have been added.\n        '
        while len(self._cleanups) > 0:
            (f, args, kwargs) = self._cleanups.pop()
            try:
                f(*args, **kwargs)
            except BaseException:
                f = failure.Failure()
                result.addError(self, f)

    def _installObserver(self):
        if False:
            print('Hello World!')
        self._observer = _logObserver
        self._observer._add()

    def _removeObserver(self):
        if False:
            print('Hello World!')
        self._observer._remove()