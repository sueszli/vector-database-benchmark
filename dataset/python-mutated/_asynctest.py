"""
Things likely to be used by writers of unit tests.

Maintainer: Jonathan Lange
"""
import inspect
import warnings
from typing import Callable, List
from zope.interface import implementer
from typing_extensions import ParamSpec
from twisted.internet import defer, utils
from twisted.python import failure
from twisted.trial import itrial, util
from twisted.trial._synctest import FailTest, SkipTest, SynchronousTestCase
_P = ParamSpec('_P')
_wait_is_running: List[None] = []

@implementer(itrial.ITestCase)
class TestCase(SynchronousTestCase):
    """
    A unit test. The atom of the unit testing universe.

    This class extends L{SynchronousTestCase} which extends C{unittest.TestCase}
    from the standard library. The main feature is the ability to return
    C{Deferred}s from tests and fixture methods and to have the suite wait for
    those C{Deferred}s to fire.  Also provides new assertions such as
    L{assertFailure}.

    @ivar timeout: A real number of seconds. If set, the test will
    raise an error if it takes longer than C{timeout} seconds.
    If not set, util.DEFAULT_TIMEOUT_DURATION is used.
    """

    def __init__(self, methodName='runTest'):
        if False:
            print('Hello World!')
        '\n        Construct an asynchronous test case for C{methodName}.\n\n        @param methodName: The name of a method on C{self}. This method should\n        be a unit test. That is, it should be a short method that calls some of\n        the assert* methods. If C{methodName} is unspecified,\n        L{SynchronousTestCase.runTest} will be used as the test method. This is\n        mostly useful for testing Trial.\n        '
        super().__init__(methodName)

    def assertFailure(self, deferred, *expectedFailures):
        if False:
            return 10
        '\n        Fail if C{deferred} does not errback with one of C{expectedFailures}.\n        Returns the original Deferred with callbacks added. You will need\n        to return this Deferred from your test case.\n        '

        def _cb(ignore):
            if False:
                i = 10
                return i + 15
            raise self.failureException(f'did not catch an error, instead got {ignore!r}')

        def _eb(failure):
            if False:
                i = 10
                return i + 15
            if failure.check(*expectedFailures):
                return failure.value
            else:
                output = '\nExpected: {!r}\nGot:\n{}'.format(expectedFailures, str(failure))
                raise self.failureException(output)
        return deferred.addCallbacks(_cb, _eb)
    failUnlessFailure = assertFailure

    def _run(self, methodName, result):
        if False:
            return 10
        from twisted.internet import reactor
        timeout = self.getTimeout()

        def onTimeout(d):
            if False:
                return 10
            e = defer.TimeoutError(f'{self!r} ({methodName}) still running at {timeout} secs')
            f = failure.Failure(e)
            try:
                d.errback(f)
            except defer.AlreadyCalledError:
                reactor.crash()
                self._timedOut = True
                todo = self.getTodo()
                if todo is not None and todo.expected(f):
                    result.addExpectedFailure(self, f, todo)
                else:
                    result.addError(self, f)
        onTimeout = utils.suppressWarnings(onTimeout, util.suppress(category=DeprecationWarning))
        method = getattr(self, methodName)
        if inspect.isgeneratorfunction(method):
            exc = TypeError('{!r} is a generator function and therefore will never run'.format(method))
            return defer.fail(exc)
        d = defer.maybeDeferred(utils.runWithWarningsSuppressed, self._getSuppress(), method)
        call = reactor.callLater(timeout, onTimeout, d)
        d.addBoth(lambda x: call.active() and call.cancel() or x)
        return d

    def __call__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return self.run(*args, **kwargs)

    def deferSetUp(self, ignored, result):
        if False:
            i = 10
            return i + 15
        d = self._run('setUp', result)
        d.addCallbacks(self.deferTestMethod, self._ebDeferSetUp, callbackArgs=(result,), errbackArgs=(result,))
        return d

    def _ebDeferSetUp(self, failure, result):
        if False:
            while True:
                i = 10
        if failure.check(SkipTest):
            result.addSkip(self, self._getSkipReason(self.setUp, failure.value))
        else:
            result.addError(self, failure)
            if failure.check(KeyboardInterrupt):
                result.stop()
        return self.deferRunCleanups(None, result)

    def deferTestMethod(self, ignored, result):
        if False:
            while True:
                i = 10
        d = self._run(self._testMethodName, result)
        d.addCallbacks(self._cbDeferTestMethod, self._ebDeferTestMethod, callbackArgs=(result,), errbackArgs=(result,))
        d.addBoth(self.deferRunCleanups, result)
        d.addBoth(self.deferTearDown, result)
        return d

    def _cbDeferTestMethod(self, ignored, result):
        if False:
            return 10
        if self.getTodo() is not None:
            result.addUnexpectedSuccess(self, self.getTodo())
        else:
            self._passed = True
        return ignored

    def _ebDeferTestMethod(self, f, result):
        if False:
            i = 10
            return i + 15
        todo = self.getTodo()
        if todo is not None and todo.expected(f):
            result.addExpectedFailure(self, f, todo)
        elif f.check(self.failureException, FailTest):
            result.addFailure(self, f)
        elif f.check(KeyboardInterrupt):
            result.addError(self, f)
            result.stop()
        elif f.check(SkipTest):
            result.addSkip(self, self._getSkipReason(getattr(self, self._testMethodName), f.value))
        else:
            result.addError(self, f)

    def deferTearDown(self, ignored, result):
        if False:
            print('Hello World!')
        d = self._run('tearDown', result)
        d.addErrback(self._ebDeferTearDown, result)
        return d

    def _ebDeferTearDown(self, failure, result):
        if False:
            while True:
                i = 10
        result.addError(self, failure)
        if failure.check(KeyboardInterrupt):
            result.stop()
        self._passed = False

    @defer.inlineCallbacks
    def deferRunCleanups(self, ignored, result):
        if False:
            i = 10
            return i + 15
        '\n        Run any scheduled cleanups and report errors (if any) to the result.\n        object.\n        '
        failures = []
        while len(self._cleanups) > 0:
            (func, args, kwargs) = self._cleanups.pop()
            try:
                yield func(*args, **kwargs)
            except Exception:
                failures.append(failure.Failure())
        for f in failures:
            result.addError(self, f)
            self._passed = False

    def _cleanUp(self, result):
        if False:
            i = 10
            return i + 15
        try:
            clean = util._Janitor(self, result).postCaseCleanup()
            if not clean:
                self._passed = False
        except BaseException:
            result.addError(self, failure.Failure())
            self._passed = False
        for error in self._observer.getErrors():
            result.addError(self, error)
            self._passed = False
        self.flushLoggedErrors()
        self._removeObserver()
        if self._passed:
            result.addSuccess(self)

    def _classCleanUp(self, result):
        if False:
            while True:
                i = 10
        try:
            util._Janitor(self, result).postClassCleanup()
        except BaseException:
            result.addError(self, failure.Failure())

    def _makeReactorMethod(self, name):
        if False:
            i = 10
            return i + 15
        '\n        Create a method which wraps the reactor method C{name}. The new\n        method issues a deprecation warning and calls the original.\n        '

        def _(*a, **kw):
            if False:
                while True:
                    i = 10
            warnings.warn('reactor.%s cannot be used inside unit tests. In the future, using %s will fail the test and may crash or hang the test run.' % (name, name), stacklevel=2, category=DeprecationWarning)
            return self._reactorMethods[name](*a, **kw)
        return _

    def _deprecateReactor(self, reactor):
        if False:
            for i in range(10):
                print('nop')
        '\n        Deprecate C{iterate}, C{crash} and C{stop} on C{reactor}. That is,\n        each method is wrapped in a function that issues a deprecation\n        warning, then calls the original.\n\n        @param reactor: The Twisted reactor.\n        '
        self._reactorMethods = {}
        for name in ['crash', 'iterate', 'stop']:
            self._reactorMethods[name] = getattr(reactor, name)
            setattr(reactor, name, self._makeReactorMethod(name))

    def _undeprecateReactor(self, reactor):
        if False:
            i = 10
            return i + 15
        '\n        Restore the deprecated reactor methods. Undoes what\n        L{_deprecateReactor} did.\n\n        @param reactor: The Twisted reactor.\n        '
        for (name, method) in self._reactorMethods.items():
            setattr(reactor, name, method)
        self._reactorMethods = {}

    def _runFixturesAndTest(self, result):
        if False:
            while True:
                i = 10
        '\n        Really run C{setUp}, the test method, and C{tearDown}.  Any of these may\n        return L{defer.Deferred}s. After they complete, do some reactor cleanup.\n\n        @param result: A L{TestResult} object.\n        '
        from twisted.internet import reactor
        self._deprecateReactor(reactor)
        self._timedOut = False
        try:
            d = self.deferSetUp(None, result)
            try:
                self._wait(d)
            finally:
                self._cleanUp(result)
                self._classCleanUp(result)
        finally:
            self._undeprecateReactor(reactor)

    def addCleanup(self, f: Callable[_P, object], *args: _P.args, **kwargs: _P.kwargs) -> None:
        if False:
            while True:
                i = 10
        '\n        Extend the base cleanup feature with support for cleanup functions which\n        return Deferreds.\n\n        If the function C{f} returns a Deferred, C{TestCase} will wait until the\n        Deferred has fired before proceeding to the next function.\n        '
        return super().addCleanup(f, *args, **kwargs)

    def getSuppress(self):
        if False:
            return 10
        return self._getSuppress()

    def getTimeout(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns the timeout value set on this test. Checks on the instance\n        first, then the class, then the module, then packages. As soon as it\n        finds something with a C{timeout} attribute, returns that. Returns\n        L{util.DEFAULT_TIMEOUT_DURATION} if it cannot find anything. See\n        L{TestCase} docstring for more details.\n        '
        timeout = util.acquireAttribute(self._parents, 'timeout', util.DEFAULT_TIMEOUT_DURATION)
        try:
            return float(timeout)
        except (ValueError, TypeError):
            warnings.warn("'timeout' attribute needs to be a number.", category=DeprecationWarning)
            return util.DEFAULT_TIMEOUT_DURATION

    def _wait(self, d, running=_wait_is_running):
        if False:
            print('Hello World!')
        'Take a Deferred that only ever callbacks. Block until it happens.'
        if running:
            raise RuntimeError('_wait is not reentrant')
        from twisted.internet import reactor
        results = []

        def append(any):
            if False:
                print('Hello World!')
            if results is not None:
                results.append(any)

        def crash(ign):
            if False:
                return 10
            if results is not None:
                reactor.crash()
        crash = utils.suppressWarnings(crash, util.suppress(message='reactor\\.crash cannot be used.*', category=DeprecationWarning))

        def stop():
            if False:
                for i in range(10):
                    print('nop')
            reactor.crash()
        stop = utils.suppressWarnings(stop, util.suppress(message='reactor\\.crash cannot be used.*', category=DeprecationWarning))
        running.append(None)
        try:
            d.addBoth(append)
            if results:
                return
            d.addBoth(crash)
            reactor.stop = stop
            try:
                reactor.run()
            finally:
                del reactor.stop
            if results or self._timedOut:
                return
            raise KeyboardInterrupt()
        finally:
            results = None
            running.pop()