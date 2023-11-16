import asyncio
import unittest
from unittest import mock
from twisted.internet import defer
from twisted.internet.defer import Deferred, succeed, fail
from twisted.python.failure import Failure
from twisted.trial.unittest import TestCase as TwistedTestCase
from golem.core.common import install_reactor
from golem.core.deferred import chain_function, DeferredSeq, deferred_from_future
from golem.tools.testwithreactor import uninstall_reactor

@mock.patch('golem.core.deferred.deferToThread', lambda x: succeed(x()))
@mock.patch('twisted.internet.reactor', mock.Mock(), create=True)
class TestDeferredSeq(unittest.TestCase):

    def test_init_empty(self):
        if False:
            while True:
                i = 10
        assert not DeferredSeq()._seq

    def test_init_with_functions(self):
        if False:
            return 10

        def fn_1():
            if False:
                print('Hello World!')
            pass

        def fn_2():
            if False:
                print('Hello World!')
            pass
        assert DeferredSeq().push(fn_1).push(fn_2)._seq == [(fn_1, (), {}), (fn_2, (), {})]

    @mock.patch('golem.core.deferred.DeferredSeq._execute')
    def test_execute_empty(self, execute):
        if False:
            return 10
        deferred_seq = DeferredSeq()
        with mock.patch('golem.core.deferred.DeferredSeq._execute', wraps=deferred_seq._execute):
            deferred_seq.execute()
        assert execute.called

    def test_execute_functions(self):
        if False:
            while True:
                i = 10
        (fn_1, fn_2) = (mock.Mock(), mock.Mock())
        DeferredSeq().push(fn_1).push(fn_2).execute()
        assert fn_1.called
        assert fn_2.called

    def test_execute_interrupted(self):
        if False:
            i = 10
            return i + 15
        (fn_1, fn_2, fn_4) = (mock.Mock(), mock.Mock(), mock.Mock())

        def fn_3(*_):
            if False:
                i = 10
                return i + 15
            raise Exception

        def def2t(f, *args, **kwargs) -> Deferred:
            if False:
                while True:
                    i = 10
            try:
                return succeed(f(*args, **kwargs))
            except Exception as exc:
                return fail(exc)
        with mock.patch('golem.core.deferred.deferToThread', def2t):
            DeferredSeq().push(fn_1).push(fn_2).push(fn_3).push(fn_4).execute()
        assert fn_1.called
        assert fn_2.called
        assert not fn_4.called

class TestChainFunction(unittest.TestCase):

    def test_callback(self):
        if False:
            while True:
                i = 10
        deferred = succeed(True)
        result = chain_function(deferred, lambda : succeed(True))
        assert result.called
        assert result.result
        assert not isinstance(result, Failure)

    def test_main_errback(self):
        if False:
            return 10
        deferred = fail(Exception())
        result = chain_function(deferred, lambda : succeed(True))
        assert result.called
        assert result.result
        assert isinstance(result.result, Failure)

    def test_fn_errback(self):
        if False:
            while True:
                i = 10
        deferred = succeed(True)
        result = chain_function(deferred, lambda : fail(Exception()))
        assert result.called
        assert result.result
        assert isinstance(result.result, Failure)

class TestDeferredFromFuture(TwistedTestCase):

    @classmethod
    def setUpClass(cls) -> None:
        if False:
            for i in range(10):
                print('nop')
        try:
            uninstall_reactor()
        except AttributeError:
            pass
        install_reactor()

    @classmethod
    def tearDownClass(cls) -> None:
        if False:
            return 10
        uninstall_reactor()

    @defer.inlineCallbacks
    def test_result(self):
        if False:
            i = 10
            return i + 15
        future = asyncio.Future()
        future.set_result(1)
        deferred = deferred_from_future(future)
        result = (yield deferred)
        self.assertEqual(result, 1)

    @defer.inlineCallbacks
    def test_exception(self):
        if False:
            while True:
                i = 10
        future = asyncio.Future()
        future.set_exception(ValueError())
        deferred = deferred_from_future(future)
        with self.assertRaises(ValueError):
            yield deferred

    @defer.inlineCallbacks
    def test_deferred_cancelled(self):
        if False:
            i = 10
            return i + 15
        future = asyncio.Future()
        deferred = deferred_from_future(future)
        deferred.cancel()
        with self.assertRaises(defer.CancelledError):
            yield deferred

    @defer.inlineCallbacks
    def test_future_cancelled(self):
        if False:
            print('Hello World!')
        future = asyncio.Future()
        deferred = deferred_from_future(future)
        future.cancel()
        with self.assertRaises(defer.CancelledError):
            yield deferred

    @defer.inlineCallbacks
    def test_timed_out(self):
        if False:
            print('Hello World!')
        from twisted.internet import reactor
        coroutine = asyncio.sleep(3)
        future = asyncio.ensure_future(coroutine)
        deferred = deferred_from_future(future)
        deferred.addTimeout(1, reactor)
        with self.assertRaises(defer.TimeoutError):
            yield deferred

    @defer.inlineCallbacks
    def test_deferred_with_timeout_cancelled(self):
        if False:
            return 10
        from twisted.internet import reactor
        future = asyncio.Future()
        deferred = deferred_from_future(future)
        deferred.addTimeout(1, reactor)
        deferred.cancel()
        with self.assertRaises(defer.CancelledError):
            yield deferred

    @defer.inlineCallbacks
    def test_future_with_timeout_cancelled(self):
        if False:
            while True:
                i = 10
        from twisted.internet import reactor
        future = asyncio.Future()
        deferred = deferred_from_future(future)
        deferred.addTimeout(1, reactor)
        future.cancel()
        with self.assertRaises(defer.CancelledError):
            yield deferred