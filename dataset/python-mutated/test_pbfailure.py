"""
Tests for error handling in PB.
"""
from io import StringIO
from twisted.internet import defer
from twisted.python import log
from twisted.python.reflect import qual
from twisted.spread import flavors, jelly, pb
from twisted.test.iosim import connectedServerAndClient
from twisted.trial import unittest

class AsynchronousException(Exception):
    """
    Helper used to test remote methods which return Deferreds which fail with
    exceptions which are not L{pb.Error} subclasses.
    """

class SynchronousException(Exception):
    """
    Helper used to test remote methods which raise exceptions which are not
    L{pb.Error} subclasses.
    """

class AsynchronousError(pb.Error):
    """
    Helper used to test remote methods which return Deferreds which fail with
    exceptions which are L{pb.Error} subclasses.
    """

class SynchronousError(pb.Error):
    """
    Helper used to test remote methods which raise exceptions which are
    L{pb.Error} subclasses.
    """

class JellyError(flavors.Jellyable, pb.Error, pb.RemoteCopy):
    pass

class SecurityError(pb.Error, pb.RemoteCopy):
    pass
pb.setUnjellyableForClass(JellyError, JellyError)
pb.setUnjellyableForClass(SecurityError, SecurityError)
pb.globalSecurity.allowInstancesOf(SecurityError)

class SimpleRoot(pb.Root):

    def remote_asynchronousException(self):
        if False:
            while True:
                i = 10
        '\n        Fail asynchronously with a non-pb.Error exception.\n        '
        return defer.fail(AsynchronousException('remote asynchronous exception'))

    def remote_synchronousException(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Fail synchronously with a non-pb.Error exception.\n        '
        raise SynchronousException('remote synchronous exception')

    def remote_asynchronousError(self):
        if False:
            print('Hello World!')
        '\n        Fail asynchronously with a pb.Error exception.\n        '
        return defer.fail(AsynchronousError('remote asynchronous error'))

    def remote_synchronousError(self):
        if False:
            while True:
                i = 10
        '\n        Fail synchronously with a pb.Error exception.\n        '
        raise SynchronousError('remote synchronous error')

    def remote_unknownError(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Fail with error that is not known to client.\n        '

        class UnknownError(pb.Error):
            pass
        raise UnknownError("I'm not known to client!")

    def remote_jelly(self):
        if False:
            print('Hello World!')
        self.raiseJelly()

    def remote_security(self):
        if False:
            for i in range(10):
                print('nop')
        self.raiseSecurity()

    def remote_deferredJelly(self):
        if False:
            for i in range(10):
                print('nop')
        d = defer.Deferred()
        d.addCallback(self.raiseJelly)
        d.callback(None)
        return d

    def remote_deferredSecurity(self):
        if False:
            print('Hello World!')
        d = defer.Deferred()
        d.addCallback(self.raiseSecurity)
        d.callback(None)
        return d

    def raiseJelly(self, results=None):
        if False:
            while True:
                i = 10
        raise JellyError("I'm jellyable!")

    def raiseSecurity(self, results=None):
        if False:
            while True:
                i = 10
        raise SecurityError("I'm secure!")

class SaveProtocolServerFactory(pb.PBServerFactory):
    """
    A L{pb.PBServerFactory} that saves the latest connected client in
    C{protocolInstance}.
    """
    protocolInstance = None

    def clientConnectionMade(self, protocol):
        if False:
            return 10
        '\n        Keep track of the given protocol.\n        '
        self.protocolInstance = protocol

class PBConnTestCase(unittest.TestCase):
    unsafeTracebacks = 0

    def setUp(self):
        if False:
            while True:
                i = 10
        self.serverFactory = SaveProtocolServerFactory(SimpleRoot())
        self.serverFactory.unsafeTracebacks = self.unsafeTracebacks
        self.clientFactory = pb.PBClientFactory()
        (self.connectedServer, self.connectedClient, self.pump) = connectedServerAndClient(lambda : self.serverFactory.buildProtocol(None), lambda : self.clientFactory.buildProtocol(None))

class PBFailureTests(PBConnTestCase):
    compare = unittest.TestCase.assertEqual

    def _exceptionTest(self, method, exceptionType, flush):
        if False:
            i = 10
            return i + 15

        def eb(err):
            if False:
                for i in range(10):
                    print('nop')
            err.trap(exceptionType)
            self.compare(err.traceback, 'Traceback unavailable\n')
            if flush:
                errs = self.flushLoggedErrors(exceptionType)
                self.assertEqual(len(errs), 1)
            return (err.type, err.value, err.traceback)
        d = self.clientFactory.getRootObject()

        def gotRootObject(root):
            if False:
                print('Hello World!')
            d = root.callRemote(method)
            d.addErrback(eb)
            return d
        d.addCallback(gotRootObject)
        self.pump.flush()

    def test_asynchronousException(self):
        if False:
            i = 10
            return i + 15
        '\n        Test that a Deferred returned by a remote method which already has a\n        Failure correctly has that error passed back to the calling side.\n        '
        return self._exceptionTest('asynchronousException', AsynchronousException, True)

    def test_synchronousException(self):
        if False:
            while True:
                i = 10
        '\n        Like L{test_asynchronousException}, but for a method which raises an\n        exception synchronously.\n        '
        return self._exceptionTest('synchronousException', SynchronousException, True)

    def test_asynchronousError(self):
        if False:
            return 10
        '\n        Like L{test_asynchronousException}, but for a method which returns a\n        Deferred failing with an L{pb.Error} subclass.\n        '
        return self._exceptionTest('asynchronousError', AsynchronousError, False)

    def test_synchronousError(self):
        if False:
            print('Hello World!')
        '\n        Like L{test_asynchronousError}, but for a method which synchronously\n        raises a L{pb.Error} subclass.\n        '
        return self._exceptionTest('synchronousError', SynchronousError, False)

    def _success(self, result, expectedResult):
        if False:
            i = 10
            return i + 15
        self.assertEqual(result, expectedResult)
        return result

    def _addFailingCallbacks(self, remoteCall, expectedResult, eb):
        if False:
            print('Hello World!')
        remoteCall.addCallbacks(self._success, eb, callbackArgs=(expectedResult,))
        return remoteCall

    def _testImpl(self, method, expected, eb, exc=None):
        if False:
            return 10
        '\n        Call the given remote method and attach the given errback to the\n        resulting Deferred.  If C{exc} is not None, also assert that one\n        exception of that type was logged.\n        '
        rootDeferred = self.clientFactory.getRootObject()

        def gotRootObj(obj):
            if False:
                print('Hello World!')
            failureDeferred = self._addFailingCallbacks(obj.callRemote(method), expected, eb)
            if exc is not None:

                def gotFailure(err):
                    if False:
                        print('Hello World!')
                    self.assertEqual(len(self.flushLoggedErrors(exc)), 1)
                    return err
                failureDeferred.addBoth(gotFailure)
            return failureDeferred
        rootDeferred.addCallback(gotRootObj)
        self.pump.flush()

    def test_jellyFailure(self):
        if False:
            while True:
                i = 10
        '\n        Test that an exception which is a subclass of L{pb.Error} has more\n        information passed across the network to the calling side.\n        '

        def failureJelly(fail):
            if False:
                for i in range(10):
                    print('nop')
            fail.trap(JellyError)
            self.assertNotIsInstance(fail.type, str)
            self.assertIsInstance(fail.value, fail.type)
            return 43
        return self._testImpl('jelly', 43, failureJelly)

    def test_deferredJellyFailure(self):
        if False:
            return 10
        '\n        Test that a Deferred which fails with a L{pb.Error} is treated in\n        the same way as a synchronously raised L{pb.Error}.\n        '

        def failureDeferredJelly(fail):
            if False:
                while True:
                    i = 10
            fail.trap(JellyError)
            self.assertNotIsInstance(fail.type, str)
            self.assertIsInstance(fail.value, fail.type)
            return 430
        return self._testImpl('deferredJelly', 430, failureDeferredJelly)

    def test_unjellyableFailure(self):
        if False:
            i = 10
            return i + 15
        '\n        A non-jellyable L{pb.Error} subclass raised by a remote method is\n        turned into a Failure with a type set to the FQPN of the exception\n        type.\n        '

        def failureUnjellyable(fail):
            if False:
                while True:
                    i = 10
            self.assertEqual(fail.type, b'twisted.spread.test.test_pbfailure.SynchronousError')
            return 431
        return self._testImpl('synchronousError', 431, failureUnjellyable)

    def test_unknownFailure(self):
        if False:
            while True:
                i = 10
        '\n        Test that an exception which is a subclass of L{pb.Error} but not\n        known on the client side has its type set properly.\n        '

        def failureUnknown(fail):
            if False:
                print('Hello World!')
            self.assertEqual(fail.type, b'twisted.spread.test.test_pbfailure.UnknownError')
            return 4310
        return self._testImpl('unknownError', 4310, failureUnknown)

    def test_securityFailure(self):
        if False:
            i = 10
            return i + 15
        '\n        Test that even if an exception is not explicitly jellyable (by being\n        a L{pb.Jellyable} subclass), as long as it is an L{pb.Error}\n        subclass it receives the same special treatment.\n        '

        def failureSecurity(fail):
            if False:
                return 10
            fail.trap(SecurityError)
            self.assertNotIsInstance(fail.type, str)
            self.assertIsInstance(fail.value, fail.type)
            return 4300
        return self._testImpl('security', 4300, failureSecurity)

    def test_deferredSecurity(self):
        if False:
            while True:
                i = 10
        '\n        Test that a Deferred which fails with a L{pb.Error} which is not\n        also a L{pb.Jellyable} is treated in the same way as a synchronously\n        raised exception of the same type.\n        '

        def failureDeferredSecurity(fail):
            if False:
                i = 10
                return i + 15
            fail.trap(SecurityError)
            self.assertNotIsInstance(fail.type, str)
            self.assertIsInstance(fail.value, fail.type)
            return 43000
        return self._testImpl('deferredSecurity', 43000, failureDeferredSecurity)

    def test_noSuchMethodFailure(self):
        if False:
            print('Hello World!')
        '\n        Test that attempting to call a method which is not defined correctly\n        results in an AttributeError on the calling side.\n        '

        def failureNoSuch(fail):
            if False:
                return 10
            fail.trap(pb.NoSuchMethod)
            self.compare(fail.traceback, 'Traceback unavailable\n')
            return 42000
        return self._testImpl('nosuch', 42000, failureNoSuch, AttributeError)

    def test_copiedFailureLogging(self):
        if False:
            print('Hello World!')
        "\n        Test that a copied failure received from a PB call can be logged\n        locally.\n\n        Note: this test needs some serious help: all it really tests is that\n        log.err(copiedFailure) doesn't raise an exception.\n        "
        d = self.clientFactory.getRootObject()

        def connected(rootObj):
            if False:
                for i in range(10):
                    print('nop')
            return rootObj.callRemote('synchronousException')
        d.addCallback(connected)

        def exception(failure):
            if False:
                print('Hello World!')
            log.err(failure)
            errs = self.flushLoggedErrors(SynchronousException)
            self.assertEqual(len(errs), 2)
        d.addErrback(exception)
        self.pump.flush()

    def test_throwExceptionIntoGenerator(self):
        if False:
            return 10
        '\n        L{pb.CopiedFailure.throwExceptionIntoGenerator} will throw a\n        L{RemoteError} into the given paused generator at the point where it\n        last yielded.\n        '
        original = pb.CopyableFailure(AttributeError('foo'))
        copy = jelly.unjelly(jelly.jelly(original, invoker=DummyInvoker()))
        exception = []

        def generatorFunc():
            if False:
                i = 10
                return i + 15
            try:
                yield None
            except pb.RemoteError as exc:
                exception.append(exc)
            else:
                self.fail('RemoteError not raised')
        gen = generatorFunc()
        gen.send(None)
        self.assertRaises(StopIteration, copy.throwExceptionIntoGenerator, gen)
        self.assertEqual(len(exception), 1)
        exc = exception[0]
        self.assertEqual(exc.remoteType, qual(AttributeError).encode('ascii'))
        self.assertEqual(exc.args, ('foo',))
        self.assertEqual(exc.remoteTraceback, 'Traceback unavailable\n')

class PBFailureUnsafeTests(PBFailureTests):
    compare = unittest.TestCase.failIfEquals
    unsafeTracebacks = 1

class DummyInvoker:
    """
    A behaviorless object to be used as the invoker parameter to
    L{jelly.jelly}.
    """
    serializingPerspective = None

class FailureJellyingTests(unittest.TestCase):
    """
    Tests for the interaction of jelly and failures.
    """

    def test_unjelliedFailureCheck(self):
        if False:
            i = 10
            return i + 15
        "\n        An unjellied L{CopyableFailure} has a check method which behaves the\n        same way as the original L{CopyableFailure}'s check method.\n        "
        original = pb.CopyableFailure(ZeroDivisionError())
        self.assertIs(original.check(ZeroDivisionError), ZeroDivisionError)
        self.assertIs(original.check(ArithmeticError), ArithmeticError)
        copied = jelly.unjelly(jelly.jelly(original, invoker=DummyInvoker()))
        self.assertIs(copied.check(ZeroDivisionError), ZeroDivisionError)
        self.assertIs(copied.check(ArithmeticError), ArithmeticError)

    def test_twiceUnjelliedFailureCheck(self):
        if False:
            return 10
        "\n        The object which results from jellying a L{CopyableFailure}, unjellying\n        the result, creating a new L{CopyableFailure} from the result of that,\n        jellying it, and finally unjellying the result of that has a check\n        method which behaves the same way as the original L{CopyableFailure}'s\n        check method.\n        "
        original = pb.CopyableFailure(ZeroDivisionError())
        self.assertIs(original.check(ZeroDivisionError), ZeroDivisionError)
        self.assertIs(original.check(ArithmeticError), ArithmeticError)
        copiedOnce = jelly.unjelly(jelly.jelly(original, invoker=DummyInvoker()))
        derivative = pb.CopyableFailure(copiedOnce)
        copiedTwice = jelly.unjelly(jelly.jelly(derivative, invoker=DummyInvoker()))
        self.assertIs(copiedTwice.check(ZeroDivisionError), ZeroDivisionError)
        self.assertIs(copiedTwice.check(ArithmeticError), ArithmeticError)

    def test_printTracebackIncludesValue(self):
        if False:
            i = 10
            return i + 15
        '\n        When L{CopiedFailure.printTraceback} is used to print a copied failure\n        which was unjellied from a L{CopyableFailure} with C{unsafeTracebacks}\n        set to C{False}, the string representation of the exception value is\n        included in the output.\n        '
        original = pb.CopyableFailure(Exception('some reason'))
        copied = jelly.unjelly(jelly.jelly(original, invoker=DummyInvoker()))
        output = StringIO()
        copied.printTraceback(output)
        exception = qual(Exception)
        expectedOutput = 'Traceback from remote host -- {}: some reason\n'.format(exception)
        self.assertEqual(expectedOutput, output.getvalue())