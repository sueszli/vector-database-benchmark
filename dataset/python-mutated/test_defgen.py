"""
Tests for L{twisted.internet.defer.deferredGenerator} and related APIs.
"""
import traceback
from twisted.internet import defer, reactor, task
from twisted.internet.defer import Deferred, deferredGenerator, inlineCallbacks, returnValue, waitForDeferred
from twisted.python.util import runWithWarningsSuppressed
from twisted.trial import unittest
from twisted.trial.util import suppress as SUPPRESS

def getThing():
    if False:
        while True:
            i = 10
    d = Deferred()
    reactor.callLater(0, d.callback, 'hi')
    return d

def getOwie():
    if False:
        for i in range(10):
            print('nop')
    d = Deferred()

    def CRAP():
        if False:
            for i in range(10):
                print('nop')
        d.errback(ZeroDivisionError('OMG'))
    reactor.callLater(0, CRAP)
    return d

class TerminalException(Exception):
    pass

class BaseDefgenTests:
    """
    This class sets up a bunch of test cases which will test both
    deferredGenerator and inlineCallbacks based generators. The subclasses
    DeferredGeneratorTests and InlineCallbacksTests each provide the actual
    generator implementations tested.
    """

    def testBasics(self):
        if False:
            i = 10
            return i + 15
        '\n        Test that a normal deferredGenerator works.  Tests yielding a\n        deferred which callbacks, as well as a deferred errbacks. Also\n        ensures returning a final value works.\n        '
        return self._genBasics().addCallback(self.assertEqual, 'WOOSH')

    def testBuggy(self):
        if False:
            while True:
                i = 10
        '\n        Ensure that a buggy generator properly signals a Failure\n        condition on result deferred.\n        '
        return self.assertFailure(self._genBuggy(), ZeroDivisionError)

    def testNothing(self):
        if False:
            return 10
        'Test that a generator which never yields results in None.'
        return self._genNothing().addCallback(self.assertEqual, None)

    def testHandledTerminalFailure(self):
        if False:
            return 10
        '\n        Create a Deferred Generator which yields a Deferred which fails and\n        handles the exception which results.  Assert that the Deferred\n        Generator does not errback its Deferred.\n        '
        return self._genHandledTerminalFailure().addCallback(self.assertEqual, None)

    def testHandledTerminalAsyncFailure(self):
        if False:
            return 10
        '\n        Just like testHandledTerminalFailure, only with a Deferred which fires\n        asynchronously with an error.\n        '
        d = defer.Deferred()
        deferredGeneratorResultDeferred = self._genHandledTerminalAsyncFailure(d)
        d.errback(TerminalException('Handled Terminal Failure'))
        return deferredGeneratorResultDeferred.addCallback(self.assertEqual, None)

    def testStackUsage(self):
        if False:
            i = 10
            return i + 15
        "\n        Make sure we don't blow the stack when yielding immediately\n        available deferreds.\n        "
        return self._genStackUsage().addCallback(self.assertEqual, 0)

    def testStackUsage2(self):
        if False:
            return 10
        "\n        Make sure we don't blow the stack when yielding immediately\n        available values.\n        "
        return self._genStackUsage2().addCallback(self.assertEqual, 0)

def deprecatedDeferredGenerator(f):
    if False:
        i = 10
        return i + 15
    '\n    Calls L{deferredGenerator} while suppressing the deprecation warning.\n\n    @param f: Function to call\n    @return: Return value of function.\n    '
    return runWithWarningsSuppressed([SUPPRESS(message='twisted.internet.defer.deferredGenerator was deprecated')], deferredGenerator, f)

class DeferredGeneratorTests(BaseDefgenTests, unittest.TestCase):

    @deprecatedDeferredGenerator
    def _genBasics(self):
        if False:
            return 10
        x = waitForDeferred(getThing())
        yield x
        x = x.getResult()
        self.assertEqual(x, 'hi')
        ow = waitForDeferred(getOwie())
        yield ow
        try:
            ow.getResult()
        except ZeroDivisionError as e:
            self.assertEqual(str(e), 'OMG')
        yield 'WOOSH'
        return

    @deprecatedDeferredGenerator
    def _genBuggy(self):
        if False:
            for i in range(10):
                print('nop')
        yield waitForDeferred(getThing())
        1 // 0

    @deprecatedDeferredGenerator
    def _genNothing(self):
        if False:
            return 10
        if False:
            yield 1

    @deprecatedDeferredGenerator
    def _genHandledTerminalFailure(self):
        if False:
            for i in range(10):
                print('nop')
        x = waitForDeferred(defer.fail(TerminalException('Handled Terminal Failure')))
        yield x
        try:
            x.getResult()
        except TerminalException:
            pass

    @deprecatedDeferredGenerator
    def _genHandledTerminalAsyncFailure(self, d):
        if False:
            i = 10
            return i + 15
        x = waitForDeferred(d)
        yield x
        try:
            x.getResult()
        except TerminalException:
            pass

    def _genStackUsage(self):
        if False:
            while True:
                i = 10
        for x in range(5000):
            x = waitForDeferred(defer.succeed(1))
            yield x
            x = x.getResult()
        yield 0
    _genStackUsage = deprecatedDeferredGenerator(_genStackUsage)

    def _genStackUsage2(self):
        if False:
            return 10
        for x in range(5000):
            yield 1
        yield 0
    _genStackUsage2 = deprecatedDeferredGenerator(_genStackUsage2)

    def testDeferredYielding(self):
        if False:
            while True:
                i = 10
        '\n        Ensure that yielding a Deferred directly is trapped as an\n        error.\n        '

        def _genDeferred():
            if False:
                while True:
                    i = 10
            yield getThing()
        _genDeferred = deprecatedDeferredGenerator(_genDeferred)
        return self.assertFailure(_genDeferred(), TypeError)
    suppress = [SUPPRESS(message='twisted.internet.defer.waitForDeferred was deprecated')]

class InlineCallbacksTests(BaseDefgenTests, unittest.TestCase):

    def _genBasics(self):
        if False:
            for i in range(10):
                print('nop')
        x = (yield getThing())
        self.assertEqual(x, 'hi')
        try:
            yield getOwie()
        except ZeroDivisionError as e:
            self.assertEqual(str(e), 'OMG')
        returnValue('WOOSH')
    _genBasics = inlineCallbacks(_genBasics)

    def _genBuggy(self):
        if False:
            for i in range(10):
                print('nop')
        yield getThing()
        1 / 0
    _genBuggy = inlineCallbacks(_genBuggy)

    def _genNothing(self):
        if False:
            while True:
                i = 10
        if False:
            yield 1
    _genNothing = inlineCallbacks(_genNothing)

    def _genHandledTerminalFailure(self):
        if False:
            return 10
        try:
            yield defer.fail(TerminalException('Handled Terminal Failure'))
        except TerminalException:
            pass
    _genHandledTerminalFailure = inlineCallbacks(_genHandledTerminalFailure)

    def _genHandledTerminalAsyncFailure(self, d):
        if False:
            print('Hello World!')
        try:
            yield d
        except TerminalException:
            pass
    _genHandledTerminalAsyncFailure = inlineCallbacks(_genHandledTerminalAsyncFailure)

    def _genStackUsage(self):
        if False:
            while True:
                i = 10
        for x in range(5000):
            yield defer.succeed(1)
        returnValue(0)
    _genStackUsage = inlineCallbacks(_genStackUsage)

    def _genStackUsage2(self):
        if False:
            i = 10
            return i + 15
        for x in range(5000):
            yield 1
        returnValue(0)
    _genStackUsage2 = inlineCallbacks(_genStackUsage2)

    def testYieldNonDeferred(self):
        if False:
            print('Hello World!')
        '\n        Ensure that yielding a non-deferred passes it back as the\n        result of the yield expression.\n\n        @return: A L{twisted.internet.defer.Deferred}\n        @rtype: L{twisted.internet.defer.Deferred}\n        '

        def _test():
            if False:
                for i in range(10):
                    print('nop')
            yield 5
            returnValue(5)
        _test = inlineCallbacks(_test)
        return _test().addCallback(self.assertEqual, 5)

    def testReturnNoValue(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensure a standard python return results in a None result.'

        def _noReturn():
            if False:
                while True:
                    i = 10
            yield 5
            return
        _noReturn = inlineCallbacks(_noReturn)
        return _noReturn().addCallback(self.assertEqual, None)

    def testReturnValue(self):
        if False:
            i = 10
            return i + 15
        'Ensure that returnValue works.'

        def _return():
            if False:
                while True:
                    i = 10
            yield 5
            returnValue(6)
        _return = inlineCallbacks(_return)
        return _return().addCallback(self.assertEqual, 6)

    def test_nonGeneratorReturn(self):
        if False:
            i = 10
            return i + 15
        '\n        Ensure that C{TypeError} with a message about L{inlineCallbacks} is\n        raised when a non-generator returns something other than a generator.\n        '

        def _noYield():
            if False:
                print('Hello World!')
            return 5
        _noYield = inlineCallbacks(_noYield)
        self.assertIn('inlineCallbacks', str(self.assertRaises(TypeError, _noYield)))

    def test_nonGeneratorReturnValue(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Ensure that C{TypeError} with a message about L{inlineCallbacks} is\n        raised when a non-generator calls L{returnValue}.\n        '

        def _noYield():
            if False:
                print('Hello World!')
            returnValue(5)
        _noYield = inlineCallbacks(_noYield)
        self.assertIn('inlineCallbacks', str(self.assertRaises(TypeError, _noYield)))

    def test_internalDefGenReturnValueDoesntLeak(self):
        if False:
            i = 10
            return i + 15
        "\n        When one inlineCallbacks calls another, the internal L{_DefGen_Return}\n        flow control exception raised by calling L{defer.returnValue} doesn't\n        leak into tracebacks captured in the caller.\n        "
        clock = task.Clock()

        @inlineCallbacks
        def _returns():
            if False:
                return 10
            '\n            This is the inner function using returnValue.\n            '
            yield task.deferLater(clock, 0)
            returnValue('actual-value-not-used-for-the-test')

        @inlineCallbacks
        def _raises():
            if False:
                return 10
            try:
                yield _returns()
                raise TerminalException('boom returnValue')
            except TerminalException:
                return traceback.format_exc()
        d = _raises()
        clock.advance(0)
        tb = self.successResultOf(d)
        self.assertNotIn('_DefGen_Return', tb)
        self.assertNotIn('During handling of the above exception, another exception occurred', tb)
        self.assertIn('test_defgen.TerminalException: boom returnValue', tb)

    def test_internalStopIterationDoesntLeak(self):
        if False:
            return 10
        '\n        When one inlineCallbacks calls another, the internal L{StopIteration}\n        flow control exception generated when the inner generator returns\n        doesn\'t leak into tracebacks captured in the caller.\n\n        This is similar to C{test_internalDefGenReturnValueDoesntLeak} but the\n        inner function uses the "normal" return statemement rather than the\n        C{returnValue} helper.\n        '
        clock = task.Clock()

        @inlineCallbacks
        def _returns():
            if False:
                print('Hello World!')
            yield task.deferLater(clock, 0)
            return 6

        @inlineCallbacks
        def _raises():
            if False:
                return 10
            try:
                yield _returns()
                raise TerminalException('boom normal return')
            except TerminalException:
                return traceback.format_exc()
        d = _raises()
        clock.advance(0)
        tb = self.successResultOf(d)
        self.assertNotIn('StopIteration', tb)
        self.assertNotIn('During handling of the above exception, another exception occurred', tb)
        self.assertIn('test_defgen.TerminalException: boom normal return', tb)

class DeprecateDeferredGeneratorTests(unittest.SynchronousTestCase):
    """
    Tests that L{DeferredGeneratorTests} and L{waitForDeferred} are
    deprecated.
    """

    def test_deferredGeneratorDeprecated(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        L{deferredGenerator} is deprecated.\n        '

        @deferredGenerator
        def decoratedFunction():
            if False:
                print('Hello World!')
            yield None
        warnings = self.flushWarnings([self.test_deferredGeneratorDeprecated])
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0]['category'], DeprecationWarning)
        self.assertEqual(warnings[0]['message'], 'twisted.internet.defer.deferredGenerator was deprecated in Twisted 15.0.0; please use twisted.internet.defer.inlineCallbacks instead')

    def test_waitForDeferredDeprecated(self):
        if False:
            print('Hello World!')
        '\n        L{waitForDeferred} is deprecated.\n        '
        d = Deferred()
        waitForDeferred(d)
        warnings = self.flushWarnings([self.test_waitForDeferredDeprecated])
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0]['category'], DeprecationWarning)
        self.assertEqual(warnings[0]['message'], 'twisted.internet.defer.waitForDeferred was deprecated in Twisted 15.0.0; please use twisted.internet.defer.inlineCallbacks instead')