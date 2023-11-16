"""
Definitions of test cases with various interesting error-related behaviors, to
be used by test modules to exercise different features of trial's test runner.

See the L{twisted.trial.test.test_tests} module docstring for details about how
this code is arranged.

Some of these tests are also used by L{twisted.trial._dist.test}.
"""
from unittest import skipIf
from twisted.internet import defer, protocol, reactor
from twisted.internet.task import deferLater
from twisted.trial import unittest, util

class FoolishError(Exception):
    pass

class LargeError(Exception):
    """
    An exception which has a string representation of at least a specified
    number of characters.
    """

    def __init__(self, minSize: int) -> None:
        if False:
            i = 10
            return i + 15
        Exception.__init__(self)
        self.minSize = minSize

    def __str__(self):
        if False:
            print('Hello World!')
        large = 'x' * self.minSize
        return f'LargeError<I fail: {large}>'

class FailureInSetUpMixin:

    def setUp(self):
        if False:
            return 10
        raise FoolishError('I am a broken setUp method')

    def test_noop(self):
        if False:
            while True:
                i = 10
        pass

class SynchronousTestFailureInSetUp(FailureInSetUpMixin, unittest.SynchronousTestCase):
    pass

class AsynchronousTestFailureInSetUp(FailureInSetUpMixin, unittest.TestCase):
    pass

class FailureInTearDownMixin:

    def tearDown(self):
        if False:
            print('Hello World!')
        raise FoolishError('I am a broken tearDown method')

    def test_noop(self):
        if False:
            for i in range(10):
                print('nop')
        pass

class SynchronousTestFailureInTearDown(FailureInTearDownMixin, unittest.SynchronousTestCase):
    pass

class AsynchronousTestFailureInTearDown(FailureInTearDownMixin, unittest.TestCase):
    pass

class FailureButTearDownRunsMixin:
    """
    A test fails, but its L{tearDown} still runs.
    """
    tornDown = False

    def tearDown(self):
        if False:
            print('Hello World!')
        self.tornDown = True

    def test_fails(self):
        if False:
            i = 10
            return i + 15
        '\n        A test that fails.\n        '
        raise FoolishError('I am a broken test')

class SynchronousTestFailureButTearDownRuns(FailureButTearDownRunsMixin, unittest.SynchronousTestCase):
    pass

class AsynchronousTestFailureButTearDownRuns(FailureButTearDownRunsMixin, unittest.TestCase):
    pass

class TestRegularFail(unittest.SynchronousTestCase):

    def test_fail(self):
        if False:
            while True:
                i = 10
        self.fail('I fail')

    def test_subfail(self):
        if False:
            while True:
                i = 10
        self.subroutine()

    def subroutine(self):
        if False:
            while True:
                i = 10
        self.fail('I fail inside')

class TestAsynchronousFail(unittest.TestCase):
    """
    Test failures for L{unittest.TestCase} based classes.
    """
    text = 'I fail'

    def test_fail(self) -> defer.Deferred[None]:
        if False:
            print('Hello World!')
        '\n        A test which fails in the callback of the returned L{defer.Deferred}.\n        '
        return deferLater(reactor, 0, self.fail, 'I fail later')

    def test_failGreaterThan64k(self) -> defer.Deferred[None]:
        if False:
            i = 10
            return i + 15
        '\n        A test which fails in the callback of the returned L{defer.Deferred}\n        with a very long string.\n        '
        return deferLater(reactor, 0, self.fail, 'I fail later: ' + 'x' * 2 ** 16)

    def test_exception(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        A test which raises an exception synchronously.\n        '
        raise Exception(self.text)

    def test_exceptionGreaterThan64k(self) -> None:
        if False:
            while True:
                i = 10
        '\n        A test which raises an exception with a long string representation\n        synchronously.\n        '
        raise LargeError(2 ** 16)

    def test_exceptionGreaterThan64kEncoded(self) -> None:
        if False:
            return 10
        '\n        A test which synchronously raises an exception with a long string\n        representation including non-ascii content.\n        '
        raise Exception('☃' * 2 ** 15)

class ErrorTest(unittest.SynchronousTestCase):
    """
    A test case which has a L{test_foo} which will raise an error.

    @ivar ran: boolean indicating whether L{test_foo} has been run.
    """
    ran = False

    def test_foo(self):
        if False:
            while True:
                i = 10
        '\n        Set C{self.ran} to True and raise a C{ZeroDivisionError}\n        '
        self.ran = True
        1 / 0

@skipIf(True, 'skipping this test')
class TestSkipTestCase(unittest.SynchronousTestCase):
    pass

class DelayedCall(unittest.TestCase):
    hiddenExceptionMsg = 'something blew up'

    def go(self):
        if False:
            return 10
        raise RuntimeError(self.hiddenExceptionMsg)

    def testHiddenException(self):
        if False:
            print('Hello World!')
        '\n        What happens if an error is raised in a DelayedCall and an error is\n        also raised in the test?\n\n        L{test_reporter.ErrorReportingTests.testHiddenException} checks that\n        both errors get reported.\n\n        Note that this behaviour is deprecated. A B{real} test would return a\n        Deferred that got triggered by the callLater. This would guarantee the\n        delayed call error gets reported.\n        '
        reactor.callLater(0, self.go)
        reactor.iterate(0.01)
        self.fail('Deliberate failure to mask the hidden exception')
    testHiddenException.suppress = [util.suppress(message='reactor\\.iterate cannot be used.*', category=DeprecationWarning)]

class ReactorCleanupTests(unittest.TestCase):

    def test_leftoverPendingCalls(self):
        if False:
            i = 10
            return i + 15

        def _():
            if False:
                while True:
                    i = 10
            print('foo!')
        reactor.callLater(10000.0, _)

class SocketOpenTest(unittest.TestCase):

    def test_socketsLeftOpen(self):
        if False:
            print('Hello World!')
        f = protocol.Factory()
        f.protocol = protocol.Protocol
        reactor.listenTCP(0, f)

class TimingOutDeferred(unittest.TestCase):

    def test_alpha(self):
        if False:
            print('Hello World!')
        pass

    def test_deferredThatNeverFires(self):
        if False:
            print('Hello World!')
        self.methodCalled = True
        d = defer.Deferred()
        return d

    def test_omega(self):
        if False:
            print('Hello World!')
        pass

def unexpectedException(self):
    if False:
        return 10
    "i will raise an unexpected exception...\n    ... *CAUSE THAT'S THE KINDA GUY I AM*\n\n    >>> 1/0\n    "

class EventuallyFailingTestCase(unittest.SynchronousTestCase):
    """
    A test suite that fails after it is run a few times.
    """
    n: int = 0

    def test_it(self):
        if False:
            return 10
        '\n        Run successfully a few times and then fail forever after.\n        '
        self.n += 1
        if self.n >= 5:
            self.fail('eventually failing')