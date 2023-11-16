"""
Tests for L{twisted.trial._dist.workerreporter}.
"""
from __future__ import annotations
from typing import Sized
from unittest import TestCase
from hamcrest import assert_that, equal_to, has_length
from hamcrest.core.matcher import Matcher
from twisted.internet.defer import Deferred
from twisted.test.iosim import connectedServerAndClient
from twisted.trial._dist.worker import LocalWorkerAMP, WorkerProtocol
from twisted.trial.reporter import TestResult
from twisted.trial.test import erroneous, pyunitcases, sample, skipping
from twisted.trial.unittest import SynchronousTestCase
from .matchers import matches_result

def run(case: SynchronousTestCase, target: TestCase) -> TestResult:
    if False:
        i = 10
        return i + 15
    '\n    Run C{target} and return a test result as populated by a worker reporter.\n\n    @param case: A test case to use to help run the target.\n    '
    result = TestResult()
    (worker, local, pump) = connectedServerAndClient(LocalWorkerAMP, WorkerProtocol)
    d = Deferred.fromCoroutine(local.run(target, result))
    pump.flush()
    assert_that(case.successResultOf(d), equal_to({'success': True}))
    return result

class WorkerReporterTests(SynchronousTestCase):
    """
    Tests for L{WorkerReporter}.
    """

    def assertTestRun(self, target: TestCase, **expectations: Matcher[Sized]) -> None:
        if False:
            return 10
        '\n        Run the given test and assert that the result matches the given\n        expectations.\n        '
        assert_that(run(self, target), matches_result(**expectations))

    def test_outsideReportingContext(self) -> None:
        if False:
            return 10
        "\n        L{WorkerReporter}'s implementation of test result methods raise\n        L{ValueError} when called outside of the\n        L{WorkerReporter.gatherReportingResults} context manager.\n        "
        (worker, local, pump) = connectedServerAndClient(LocalWorkerAMP, WorkerProtocol)
        case = sample.FooTest('test_foo')
        with self.assertRaises(ValueError):
            worker._result.addSuccess(case)

    def test_addSuccess(self) -> None:
        if False:
            print('Hello World!')
        '\n        L{WorkerReporter} propagates successes.\n        '
        self.assertTestRun(sample.FooTest('test_foo'), successes=equal_to(1))

    def test_addError(self) -> None:
        if False:
            while True:
                i = 10
        "\n        L{WorkerReporter} propagates errors from trial's TestCases.\n        "
        self.assertTestRun(erroneous.TestAsynchronousFail('test_exception'), errors=has_length(1))

    def test_addErrorGreaterThan64k(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        L{WorkerReporter} propagates errors with large string representations.\n        '
        self.assertTestRun(erroneous.TestAsynchronousFail('test_exceptionGreaterThan64k'), errors=has_length(1))

    def test_addErrorGreaterThan64kEncoded(self) -> None:
        if False:
            return 10
        '\n        L{WorkerReporter} propagates errors with a string representation that\n        is smaller than an implementation-specific limit but which encode to a\n        byte representation that exceeds this limit.\n        '
        self.assertTestRun(erroneous.TestAsynchronousFail('test_exceptionGreaterThan64kEncoded'), errors=has_length(1))

    def test_addErrorTuple(self) -> None:
        if False:
            i = 10
            return i + 15
        "\n        L{WorkerReporter} propagates errors from pyunit's TestCases.\n        "
        self.assertTestRun(pyunitcases.PyUnitTest('test_error'), errors=has_length(1))

    def test_addFailure(self) -> None:
        if False:
            while True:
                i = 10
        "\n        L{WorkerReporter} propagates test failures from trial's TestCases.\n        "
        self.assertTestRun(erroneous.TestRegularFail('test_fail'), failures=has_length(1))

    def test_addFailureGreaterThan64k(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        L{WorkerReporter} propagates test failures with large string representations.\n        '
        self.assertTestRun(erroneous.TestAsynchronousFail('test_failGreaterThan64k'), failures=has_length(1))

    def test_addFailureTuple(self) -> None:
        if False:
            print('Hello World!')
        "\n        L{WorkerReporter} propagates test failures from pyunit's TestCases.\n        "
        self.assertTestRun(pyunitcases.PyUnitTest('test_fail'), failures=has_length(1))

    def test_addSkip(self) -> None:
        if False:
            print('Hello World!')
        '\n        L{WorkerReporter} propagates skips.\n        '
        self.assertTestRun(skipping.SynchronousSkipping('test_skip1'), skips=has_length(1))

    def test_addSkipPyunit(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        L{WorkerReporter} propagates skips from L{unittest.TestCase} cases.\n        '
        self.assertTestRun(pyunitcases.PyUnitTest('test_skip'), skips=has_length(1))

    def test_addExpectedFailure(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        L{WorkerReporter} propagates expected failures.\n        '
        self.assertTestRun(skipping.SynchronousStrictTodo('test_todo1'), expectedFailures=has_length(1))

    def test_addExpectedFailureGreaterThan64k(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        WorkerReporter propagates expected failures with large string representations.\n        '
        self.assertTestRun(skipping.ExpectedFailure('test_expectedFailureGreaterThan64k'), expectedFailures=has_length(1))

    def test_addUnexpectedSuccess(self) -> None:
        if False:
            while True:
                i = 10
        '\n        L{WorkerReporter} propagates unexpected successes.\n        '
        self.assertTestRun(skipping.SynchronousTodo('test_todo3'), unexpectedSuccesses=has_length(1))