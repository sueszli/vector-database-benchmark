"""
Tests for Deferred handling by L{twisted.trial.unittest.TestCase}.
"""
from __future__ import annotations
from twisted.internet import defer, reactor, threads
from twisted.python.failure import Failure
from twisted.python.util import runWithWarningsSuppressed
from twisted.trial import unittest
from twisted.trial.util import suppress as SUPPRESS

class DeferredSetUpOK(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        d = defer.succeed('value')
        d.addCallback(self._cb_setUpCalled)
        return d

    def _cb_setUpCalled(self, ignored):
        if False:
            return 10
        self._setUpCalled = True

    def test_ok(self):
        if False:
            while True:
                i = 10
        self.assertTrue(self._setUpCalled)

class DeferredSetUpFail(unittest.TestCase):
    testCalled = False

    def setUp(self):
        if False:
            print('Hello World!')
        return defer.fail(unittest.FailTest('i fail'))

    def test_ok(self):
        if False:
            i = 10
            return i + 15
        DeferredSetUpFail.testCalled = True
        self.fail('I should not get called')

class DeferredSetUpCallbackFail(unittest.TestCase):
    testCalled = False

    def setUp(self):
        if False:
            i = 10
            return i + 15
        d = defer.succeed('value')
        d.addCallback(self._cb_setUpCalled)
        return d

    def _cb_setUpCalled(self, ignored):
        if False:
            return 10
        self.fail('deliberate failure')

    def test_ok(self):
        if False:
            return 10
        DeferredSetUpCallbackFail.testCalled = True

class DeferredSetUpError(unittest.TestCase):
    testCalled = False

    def setUp(self):
        if False:
            print('Hello World!')
        return defer.fail(RuntimeError('deliberate error'))

    def test_ok(self):
        if False:
            for i in range(10):
                print('nop')
        DeferredSetUpError.testCalled = True

class DeferredSetUpNeverFire(unittest.TestCase):
    testCalled = False

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        return defer.Deferred()

    def test_ok(self):
        if False:
            while True:
                i = 10
        DeferredSetUpNeverFire.testCalled = True

class DeferredSetUpSkip(unittest.TestCase):
    testCalled = False

    def setUp(self):
        if False:
            print('Hello World!')
        d = defer.succeed('value')
        d.addCallback(self._cb1)
        return d

    def _cb1(self, ignored):
        if False:
            print('Hello World!')
        raise unittest.SkipTest('skip me')

    def test_ok(self):
        if False:
            print('Hello World!')
        DeferredSetUpSkip.testCalled = True

class DeferredTests(unittest.TestCase):
    touched = False

    def _cb_fail(self, reason):
        if False:
            for i in range(10):
                print('nop')
        self.fail(reason)

    def _cb_error(self, reason):
        if False:
            i = 10
            return i + 15
        raise RuntimeError(reason)

    def _cb_skip(self, reason):
        if False:
            i = 10
            return i + 15
        raise unittest.SkipTest(reason)

    def _touchClass(self, ignored):
        if False:
            print('Hello World!')
        self.__class__.touched = True

    def setUp(self):
        if False:
            while True:
                i = 10
        self.__class__.touched = False

    def test_pass(self):
        if False:
            for i in range(10):
                print('nop')
        return defer.succeed('success')

    def test_passGenerated(self):
        if False:
            while True:
                i = 10
        self._touchClass(None)
        yield None
    test_passGenerated = runWithWarningsSuppressed([SUPPRESS(message='twisted.internet.defer.deferredGenerator was deprecated')], defer.deferredGenerator, test_passGenerated)

    @defer.inlineCallbacks
    def test_passInlineCallbacks(self):
        if False:
            i = 10
            return i + 15
        '\n        Test case that is decorated with L{defer.inlineCallbacks}.\n        '
        self._touchClass(None)
        yield None

    def test_fail(self):
        if False:
            while True:
                i = 10
        return defer.fail(self.failureException('I fail'))

    def test_failureInCallback(self):
        if False:
            for i in range(10):
                print('nop')
        d = defer.succeed('fail')
        d.addCallback(self._cb_fail)
        return d

    def test_errorInCallback(self):
        if False:
            print('Hello World!')
        d = defer.succeed('error')
        d.addCallback(self._cb_error)
        return d

    def test_skip(self):
        if False:
            i = 10
            return i + 15
        d = defer.succeed('skip')
        d.addCallback(self._cb_skip)
        d.addCallback(self._touchClass)
        return d

    def test_thread(self):
        if False:
            i = 10
            return i + 15
        return threads.deferToThread(lambda : None)

    def test_expectedFailure(self):
        if False:
            for i in range(10):
                print('nop')
        d = defer.succeed('todo')
        d.addCallback(self._cb_error)
        return d
    test_expectedFailure.todo = 'Expected failure'

class TimeoutTests(unittest.TestCase):
    timedOut: Failure | None = None

    def test_pass(self):
        if False:
            print('Hello World!')
        d = defer.Deferred()
        reactor.callLater(0, d.callback, 'hoorj!')
        return d
    test_pass.timeout = 2

    def test_passDefault(self):
        if False:
            i = 10
            return i + 15
        d = defer.Deferred()
        reactor.callLater(0, d.callback, 'hoorj!')
        return d

    def test_timeout(self):
        if False:
            print('Hello World!')
        return defer.Deferred()
    test_timeout.timeout = 0.1

    def test_timeoutZero(self):
        if False:
            print('Hello World!')
        return defer.Deferred()
    test_timeoutZero.timeout = 0

    def test_expectedFailure(self):
        if False:
            while True:
                i = 10
        return defer.Deferred()
    test_expectedFailure.timeout = 0.1
    test_expectedFailure.todo = 'i will get it right, eventually'

    def test_skip(self):
        if False:
            return 10
        return defer.Deferred()
    test_skip.timeout = 0.1
    test_skip.skip = 'i will get it right, eventually'

    def test_errorPropagation(self):
        if False:
            for i in range(10):
                print('nop')

        def timedOut(err):
            if False:
                for i in range(10):
                    print('nop')
            self.__class__.timedOut = err
            return err
        d = defer.Deferred()
        d.addErrback(timedOut)
        return d
    test_errorPropagation.timeout = 0.1

    def test_calledButNeverCallback(self):
        if False:
            i = 10
            return i + 15
        d = defer.Deferred()

        def neverFire(r):
            if False:
                while True:
                    i = 10
            return defer.Deferred()
        d.addCallback(neverFire)
        d.callback(1)
        return d
    test_calledButNeverCallback.timeout = 0.1

class TestClassTimeoutAttribute(unittest.TestCase):
    timeout = 0.2

    def setUp(self):
        if False:
            print('Hello World!')
        self.d = defer.Deferred()

    def testMethod(self):
        if False:
            while True:
                i = 10
        self.methodCalled = True
        return self.d