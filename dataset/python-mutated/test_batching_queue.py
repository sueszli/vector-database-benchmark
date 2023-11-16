from typing import List, Tuple
from prometheus_client import Gauge
from twisted.internet import defer
from synapse.logging.context import make_deferred_yieldable
from synapse.util.batching_queue import BatchingQueue, number_in_flight, number_of_keys, number_queued
from tests.server import get_clock
from tests.unittest import TestCase

class BatchingQueueTestCase(TestCase):

    def setUp(self) -> None:
        if False:
            i = 10
            return i + 15
        (self.clock, hs_clock) = get_clock()
        try:
            number_queued.remove('test_queue')
            number_of_keys.remove('test_queue')
            number_in_flight.remove('test_queue')
        except KeyError:
            pass
        self._pending_calls: List[Tuple[List[str], defer.Deferred]] = []
        self.queue: BatchingQueue[str, str] = BatchingQueue('test_queue', hs_clock, self._process_queue)

    async def _process_queue(self, values: List[str]) -> str:
        d: 'defer.Deferred[str]' = defer.Deferred()
        self._pending_calls.append((values, d))
        return await make_deferred_yieldable(d)

    def _get_sample_with_name(self, metric: Gauge, name: str) -> float:
        if False:
            print('Hello World!')
        'For a prometheus metric get the value of the sample that has a\n        matching "name" label.\n        '
        for sample in next(iter(metric.collect())).samples:
            if sample.labels.get('name') == name:
                return sample.value
        self.fail('Found no matching sample')

    def _assert_metrics(self, queued: int, keys: int, in_flight: int) -> None:
        if False:
            while True:
                i = 10
        'Assert that the metrics are correct'
        sample = self._get_sample_with_name(number_queued, self.queue._name)
        self.assertEqual(sample, queued, 'number_queued')
        sample = self._get_sample_with_name(number_of_keys, self.queue._name)
        self.assertEqual(sample, keys, 'number_of_keys')
        sample = self._get_sample_with_name(number_in_flight, self.queue._name)
        self.assertEqual(sample, in_flight, 'number_in_flight')

    def test_simple(self) -> None:
        if False:
            while True:
                i = 10
        'Tests the basic case of calling `add_to_queue` once and having\n        `_process_queue` return.\n        '
        self.assertFalse(self._pending_calls)
        queue_d = defer.ensureDeferred(self.queue.add_to_queue('foo'))
        self._assert_metrics(queued=1, keys=1, in_flight=1)
        self.assertFalse(self._pending_calls)
        self.assertFalse(queue_d.called)
        self.clock.pump([0])
        self.assertEqual(len(self._pending_calls), 1)
        self.assertEqual(self._pending_calls[0][0], ['foo'])
        self.assertFalse(queue_d.called)
        self._assert_metrics(queued=0, keys=0, in_flight=1)
        self._pending_calls.pop()[1].callback('bar')
        self.assertEqual(self.successResultOf(queue_d), 'bar')
        self._assert_metrics(queued=0, keys=0, in_flight=0)

    def test_batching(self) -> None:
        if False:
            return 10
        'Test that multiple calls at the same time get batched up into one\n        call to `_process_queue`.\n        '
        self.assertFalse(self._pending_calls)
        queue_d1 = defer.ensureDeferred(self.queue.add_to_queue('foo1'))
        queue_d2 = defer.ensureDeferred(self.queue.add_to_queue('foo2'))
        self._assert_metrics(queued=2, keys=1, in_flight=2)
        self.clock.pump([0])
        self.assertEqual(len(self._pending_calls), 1)
        self.assertEqual(self._pending_calls[0][0], ['foo1', 'foo2'])
        self.assertFalse(queue_d1.called)
        self.assertFalse(queue_d2.called)
        self._assert_metrics(queued=0, keys=0, in_flight=2)
        self._pending_calls.pop()[1].callback('bar')
        self.assertEqual(self.successResultOf(queue_d1), 'bar')
        self.assertEqual(self.successResultOf(queue_d2), 'bar')
        self._assert_metrics(queued=0, keys=0, in_flight=0)

    def test_queuing(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test that we queue up requests while a `_process_queue` is being\n        called.\n        '
        self.assertFalse(self._pending_calls)
        queue_d1 = defer.ensureDeferred(self.queue.add_to_queue('foo1'))
        self.clock.pump([0])
        self.assertEqual(len(self._pending_calls), 1)
        queue_d2 = defer.ensureDeferred(self.queue.add_to_queue('foo2'))
        queue_d3 = defer.ensureDeferred(self.queue.add_to_queue('foo3'))
        self.assertEqual(len(self._pending_calls), 1)
        self.assertEqual(self._pending_calls[0][0], ['foo1'])
        self.assertFalse(queue_d1.called)
        self.assertFalse(queue_d2.called)
        self.assertFalse(queue_d3.called)
        self._assert_metrics(queued=2, keys=1, in_flight=3)
        self._pending_calls.pop()[1].callback('bar1')
        self.assertEqual(self.successResultOf(queue_d1), 'bar1')
        self.assertFalse(queue_d2.called)
        self.assertFalse(queue_d3.called)
        self._assert_metrics(queued=2, keys=1, in_flight=2)
        self.clock.pump([0])
        self.assertEqual(len(self._pending_calls), 1)
        self.assertEqual(self._pending_calls[0][0], ['foo2', 'foo3'])
        self.assertFalse(queue_d2.called)
        self.assertFalse(queue_d3.called)
        self._assert_metrics(queued=0, keys=0, in_flight=2)
        self._pending_calls.pop()[1].callback('bar2')
        self.assertEqual(self.successResultOf(queue_d2), 'bar2')
        self.assertEqual(self.successResultOf(queue_d3), 'bar2')
        self._assert_metrics(queued=0, keys=0, in_flight=0)

    def test_different_keys(self) -> None:
        if False:
            while True:
                i = 10
        'Test that calls to different keys get processed in parallel.'
        self.assertFalse(self._pending_calls)
        queue_d1 = defer.ensureDeferred(self.queue.add_to_queue('foo1', key=1))
        self.clock.pump([0])
        queue_d2 = defer.ensureDeferred(self.queue.add_to_queue('foo2', key=2))
        self.clock.pump([0])
        queue_d3 = defer.ensureDeferred(self.queue.add_to_queue('foo3', key=2))
        self.assertEqual(len(self._pending_calls), 2)
        self.assertEqual(self._pending_calls[0][0], ['foo1'])
        self.assertEqual(self._pending_calls[1][0], ['foo2'])
        self.assertFalse(queue_d1.called)
        self.assertFalse(queue_d2.called)
        self.assertFalse(queue_d3.called)
        self._assert_metrics(queued=1, keys=1, in_flight=3)
        self._pending_calls.pop(0)[1].callback('bar1')
        self.assertEqual(self.successResultOf(queue_d1), 'bar1')
        self.assertFalse(queue_d2.called)
        self.assertFalse(queue_d3.called)
        self._assert_metrics(queued=1, keys=1, in_flight=2)
        self._pending_calls.pop()[1].callback('bar2')
        self.assertEqual(self.successResultOf(queue_d2), 'bar2')
        self.assertFalse(queue_d3.called)
        self.clock.pump([0])
        self.assertEqual(len(self._pending_calls), 1)
        self.assertEqual(self._pending_calls[0][0], ['foo3'])
        self.assertFalse(queue_d3.called)
        self._assert_metrics(queued=0, keys=0, in_flight=1)
        self._pending_calls.pop()[1].callback('bar4')
        self.assertEqual(self.successResultOf(queue_d3), 'bar4')
        self._assert_metrics(queued=0, keys=0, in_flight=0)