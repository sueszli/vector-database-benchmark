from __future__ import annotations
import sys
import traceback
import unittest as pyunit
from unittest import skipIf
from zope.interface import implementer
from twisted.python.failure import Failure
from twisted.trial.itrial import IReporter, ITestCase
from twisted.trial.test import pyunitcases
from twisted.trial.unittest import PyUnitResultAdapter, SynchronousTestCase

class PyUnitTestTests(SynchronousTestCase):

    def setUp(self) -> None:
        if False:
            while True:
                i = 10
        self.original = pyunitcases.PyUnitTest('test_pass')
        self.test = ITestCase(self.original)

    def test_callable(self) -> None:
        if False:
            print('Hello World!')
        "\n        Tests must be callable in order to be used with Python's unittest.py.\n        "
        self.assertTrue(callable(self.test), f'{self.test!r} is not callable.')

class PyUnitResultTests(SynchronousTestCase):
    """
    Tests to show that PyUnitResultAdapter wraps TestResult objects from the
    standard library 'unittest' module in such a way as to make them usable and
    useful from Trial.
    """

    class ErrorTest(SynchronousTestCase):
        """
        A test case which has a L{test_foo} which will raise an error.

        @ivar ran: boolean indicating whether L{test_foo} has been run.
        """
        ran = False

        def test_foo(self) -> None:
            if False:
                i = 10
                return i + 15
            '\n            Set C{self.ran} to True and raise a C{ZeroDivisionError}\n            '
            self.ran = True
            1 / 0

    def test_dontUseAdapterWhenReporterProvidesIReporter(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        The L{PyUnitResultAdapter} is only used when the result passed to\n        C{run} does *not* provide L{IReporter}.\n        '

        @implementer(IReporter)
        class StubReporter:
            """
            A reporter which records data about calls made to it.

            @ivar errors: Errors passed to L{addError}.
            @ivar failures: Failures passed to L{addFailure}.
            """

            def __init__(self) -> None:
                if False:
                    while True:
                        i = 10
                self.errors: list[Failure] = []
                self.failures: list[None] = []

            def startTest(self, test: object) -> None:
                if False:
                    print('Hello World!')
                '\n                Do nothing.\n                '

            def stopTest(self, test: object) -> None:
                if False:
                    print('Hello World!')
                '\n                Do nothing.\n                '

            def addError(self, test: object, error: Failure) -> None:
                if False:
                    for i in range(10):
                        print('nop')
                '\n                Record the error.\n                '
                self.errors.append(error)
        test = self.ErrorTest('test_foo')
        result = StubReporter()
        test.run(result)
        self.assertIsInstance(result.errors[0], Failure)

    def test_success(self) -> None:
        if False:
            print('Hello World!')

        class SuccessTest(SynchronousTestCase):
            ran = False

            def test_foo(s) -> None:
                if False:
                    return 10
                s.ran = True
        test = SuccessTest('test_foo')
        result = pyunit.TestResult()
        test.run(result)
        self.assertTrue(test.ran)
        self.assertEqual(1, result.testsRun)
        self.assertTrue(result.wasSuccessful())

    def test_failure(self) -> None:
        if False:
            for i in range(10):
                print('nop')

        class FailureTest(SynchronousTestCase):
            ran = False

            def test_foo(s) -> None:
                if False:
                    return 10
                s.ran = True
                s.fail('boom!')
        test = FailureTest('test_foo')
        result = pyunit.TestResult()
        test.run(result)
        self.assertTrue(test.ran)
        self.assertEqual(1, result.testsRun)
        self.assertEqual(1, len(result.failures))
        self.assertFalse(result.wasSuccessful())

    def test_error(self) -> None:
        if False:
            i = 10
            return i + 15
        test = self.ErrorTest('test_foo')
        result = pyunit.TestResult()
        test.run(result)
        self.assertTrue(test.ran)
        self.assertEqual(1, result.testsRun)
        self.assertEqual(1, len(result.errors))
        self.assertFalse(result.wasSuccessful())

    def test_setUpError(self) -> None:
        if False:
            while True:
                i = 10

        class ErrorTest(SynchronousTestCase):
            ran = False

            def setUp(self) -> None:
                if False:
                    return 10
                1 / 0

            def test_foo(s) -> None:
                if False:
                    i = 10
                    return i + 15
                s.ran = True
        test = ErrorTest('test_foo')
        result = pyunit.TestResult()
        test.run(result)
        self.assertFalse(test.ran)
        self.assertEqual(1, result.testsRun)
        self.assertEqual(1, len(result.errors))
        self.assertFalse(result.wasSuccessful())

    def test_tracebackFromFailure(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Errors added through the L{PyUnitResultAdapter} have the same traceback\n        information as if there were no adapter at all.\n        '
        try:
            1 / 0
        except ZeroDivisionError:
            exc_info = sys.exc_info()
            f = Failure()
        pyresult = pyunit.TestResult()
        result = PyUnitResultAdapter(pyresult)
        result.addError(self, f)
        self.assertEqual(pyresult.errors[0][1], ''.join(traceback.format_exception(*exc_info)))

    def test_traceback(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        As test_tracebackFromFailure, but covering more code.\n        '

        class ErrorTest(SynchronousTestCase):
            exc_info = None

            def test_foo(self) -> None:
                if False:
                    while True:
                        i = 10
                try:
                    1 / 0
                except ZeroDivisionError:
                    self.exc_info = sys.exc_info()
                    raise
        test = ErrorTest('test_foo')
        result = pyunit.TestResult()
        test.run(result)
        assert test.exc_info is not None
        expected_stack = ''.join(traceback.format_tb(test.exc_info[2]))
        observed_stack = '\n'.join(result.errors[0][1].splitlines()[:-1])
        self.assertEqual(expected_stack.strip(), observed_stack[-len(expected_stack):].strip())

    def test_tracebackFromCleanFailure(self) -> None:
        if False:
            print('Hello World!')
        '\n        Errors added through the L{PyUnitResultAdapter} have the same\n        traceback information as if there were no adapter at all, even\n        if the Failure that held the information has been cleaned.\n        '
        try:
            1 / 0
        except ZeroDivisionError:
            exc_info = sys.exc_info()
            f = Failure()
        f.cleanFailure()
        pyresult = pyunit.TestResult()
        result = PyUnitResultAdapter(pyresult)
        result.addError(self, f)
        tback = ''.join(traceback.format_exception(*exc_info))
        self.assertEqual(pyresult.errors[0][1].endswith('ZeroDivisionError: division by zero\n'), tback.endswith('ZeroDivisionError: division by zero\n'))

    def test_trialSkip(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        "\n        Skips using trial's skipping functionality are reported as skips in\n        the L{pyunit.TestResult}.\n        "

        class SkipTest(SynchronousTestCase):

            @skipIf(True, "Let's skip!")
            def test_skip(self) -> None:
                if False:
                    while True:
                        i = 10
                1 / 0
        test = SkipTest('test_skip')
        result = pyunit.TestResult()
        test.run(result)
        self.assertEqual(result.skipped, [(test, "Let's skip!")])

    def test_pyunitSkip(self) -> None:
        if False:
            i = 10
            return i + 15
        "\n        Skips using pyunit's skipping functionality are reported as skips in\n        the L{pyunit.TestResult}.\n        "

        class SkipTest(SynchronousTestCase):

            @pyunit.skip('skippy')
            def test_skip(self) -> None:
                if False:
                    return 10
                1 / 0
        test = SkipTest('test_skip')
        result = pyunit.TestResult()
        test.run(result)
        self.assertEqual(result.skipped, [(test, 'skippy')])