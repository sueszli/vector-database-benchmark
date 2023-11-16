from typing import Hashable, Tuple
from typing_extensions import Protocol
from twisted.internet import defer, reactor
from twisted.internet.base import ReactorBase
from twisted.internet.defer import CancelledError, Deferred
from synapse.logging.context import LoggingContext, current_context
from synapse.util.async_helpers import Linearizer
from tests import unittest

class UnblockFunction(Protocol):

    def __call__(self, pump_reactor: bool=True) -> None:
        if False:
            return 10
        ...

class LinearizerTestCase(unittest.TestCase):

    def _start_task(self, linearizer: Linearizer, key: Hashable) -> Tuple['Deferred[None]', 'Deferred[None]', UnblockFunction]:
        if False:
            print('Hello World!')
        'Starts a task which acquires the linearizer lock, blocks, then completes.\n\n        Args:\n            linearizer: The `Linearizer`.\n            key: The `Linearizer` key.\n\n        Returns:\n            A tuple containing:\n             * A cancellable `Deferred` for the entire task.\n             * A `Deferred` that resolves once the task acquires the lock.\n             * A function that unblocks the task. Must be called by the caller\n               to allow the task to release the lock and complete.\n        '
        acquired_d: 'Deferred[None]' = Deferred()
        unblock_d: 'Deferred[None]' = Deferred()

        async def task() -> None:
            async with linearizer.queue(key):
                acquired_d.callback(None)
                await unblock_d
        d = defer.ensureDeferred(task())

        def unblock(pump_reactor: bool=True) -> None:
            if False:
                i = 10
                return i + 15
            unblock_d.callback(None)
            if pump_reactor:
                self._pump()
        return (d, acquired_d, unblock)

    def _pump(self) -> None:
        if False:
            return 10
        'Pump the reactor to advance `Linearizer`s.'
        assert isinstance(reactor, ReactorBase)
        while reactor.getDelayedCalls():
            reactor.runUntilCurrent()

    def test_linearizer(self) -> None:
        if False:
            i = 10
            return i + 15
        'Tests that a task is queued up behind an earlier task.'
        linearizer = Linearizer()
        key = object()
        (_, acquired_d1, unblock1) = self._start_task(linearizer, key)
        self.assertTrue(acquired_d1.called)
        (_, acquired_d2, unblock2) = self._start_task(linearizer, key)
        self.assertFalse(acquired_d2.called)
        unblock1()
        self.assertTrue(acquired_d2.called)
        unblock2()

    def test_linearizer_is_queued(self) -> None:
        if False:
            i = 10
            return i + 15
        'Tests `Linearizer.is_queued`.\n\n        Runs through the same scenario as `test_linearizer`.\n        '
        linearizer = Linearizer()
        key = object()
        (_, acquired_d1, unblock1) = self._start_task(linearizer, key)
        self.assertTrue(acquired_d1.called)
        self.assertFalse(linearizer.is_queued(key))
        (_, acquired_d2, unblock2) = self._start_task(linearizer, key)
        self.assertFalse(acquired_d2.called)
        self.assertTrue(linearizer.is_queued(key))
        unblock1()
        self.assertTrue(acquired_d2.called)
        self.assertFalse(linearizer.is_queued(key))
        unblock2()
        self.assertFalse(linearizer.is_queued(key))

    def test_lots_of_queued_things(self) -> None:
        if False:
            return 10
        'Tests lots of fast things queued up behind a slow thing.\n\n        The stack should *not* explode when the slow thing completes.\n        '
        linearizer = Linearizer()
        key = ''

        async def func(i: int) -> None:
            with LoggingContext('func(%s)' % i) as lc:
                async with linearizer.queue(key):
                    self.assertEqual(current_context(), lc)
                self.assertEqual(current_context(), lc)
        (_, _, unblock) = self._start_task(linearizer, key)
        for i in range(1, 100):
            defer.ensureDeferred(func(i))
        d = defer.ensureDeferred(func(1000))
        unblock()
        self.successResultOf(d)

    def test_multiple_entries(self) -> None:
        if False:
            return 10
        'Tests a `Linearizer` with a concurrency above 1.'
        limiter = Linearizer(max_count=3)
        key = object()
        (_, acquired_d1, unblock1) = self._start_task(limiter, key)
        self.assertTrue(acquired_d1.called)
        (_, acquired_d2, unblock2) = self._start_task(limiter, key)
        self.assertTrue(acquired_d2.called)
        (_, acquired_d3, unblock3) = self._start_task(limiter, key)
        self.assertTrue(acquired_d3.called)
        (_, acquired_d4, unblock4) = self._start_task(limiter, key)
        self.assertFalse(acquired_d4.called)
        (_, acquired_d5, unblock5) = self._start_task(limiter, key)
        self.assertFalse(acquired_d5.called)
        unblock1()
        self.assertTrue(acquired_d4.called)
        self.assertFalse(acquired_d5.called)
        unblock3()
        self.assertTrue(acquired_d5.called)
        unblock2()
        unblock4()
        unblock5()
        (_, acquired_d6, unblock6) = self._start_task(limiter, key)
        self.assertTrue(acquired_d6)
        unblock6()

    def test_cancellation(self) -> None:
        if False:
            i = 10
            return i + 15
        'Tests cancellation while waiting for a `Linearizer`.'
        linearizer = Linearizer()
        key = object()
        (d1, acquired_d1, unblock1) = self._start_task(linearizer, key)
        self.assertTrue(acquired_d1.called)
        (d2, acquired_d2, _) = self._start_task(linearizer, key)
        self.assertFalse(acquired_d2.called)
        (d3, acquired_d3, unblock3) = self._start_task(linearizer, key)
        self.assertFalse(acquired_d3.called)
        d2.cancel()
        unblock1()
        self.successResultOf(d1)
        self.assertTrue(d2.called)
        self.failureResultOf(d2, CancelledError)
        self.assertTrue(acquired_d3.called, 'Third task did not get the lock after the second task was cancelled')
        unblock3()
        self.successResultOf(d3)

    def test_cancellation_during_sleep(self) -> None:
        if False:
            return 10
        'Tests cancellation during the sleep just after waiting for a `Linearizer`.'
        linearizer = Linearizer()
        key = object()
        (d1, acquired_d1, unblock1) = self._start_task(linearizer, key)
        self.assertTrue(acquired_d1.called)
        (d2, acquired_d2, _) = self._start_task(linearizer, key)
        self.assertFalse(acquired_d2.called)
        (d3, acquired_d3, unblock3) = self._start_task(linearizer, key)
        self.assertFalse(acquired_d3.called)
        unblock1(pump_reactor=False)
        self.successResultOf(d1)
        d2.cancel()
        self._pump()
        self.assertTrue(d2.called)
        self.failureResultOf(d2, CancelledError)
        self.assertTrue(acquired_d3.called, 'Third task did not get the lock after the second task was cancelled')
        unblock3()
        self.successResultOf(d3)