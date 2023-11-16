import threading
from io import BytesIO
from typing import BinaryIO, Generator, Optional, cast
from unittest.mock import NonCallableMock
from zope.interface import implementer
from twisted.internet import defer, reactor as _reactor
from twisted.internet.interfaces import IPullProducer
from synapse.types import ISynapseReactor
from synapse.util.file_consumer import BackgroundFileConsumer
from tests import unittest
reactor = cast(ISynapseReactor, _reactor)

class FileConsumerTests(unittest.TestCase):

    @defer.inlineCallbacks
    def test_pull_consumer(self) -> Generator['defer.Deferred[object]', object, None]:
        if False:
            print('Hello World!')
        string_file = BytesIO()
        consumer = BackgroundFileConsumer(string_file, reactor=reactor)
        try:
            producer = DummyPullProducer()
            yield producer.register_with_consumer(consumer)
            yield producer.write_and_wait(b'Foo')
            self.assertEqual(string_file.getvalue(), b'Foo')
            yield producer.write_and_wait(b'Bar')
            self.assertEqual(string_file.getvalue(), b'FooBar')
        finally:
            consumer.unregisterProducer()
        yield consumer.wait()
        self.assertTrue(string_file.closed)

    @defer.inlineCallbacks
    def test_push_consumer(self) -> Generator['defer.Deferred[object]', object, None]:
        if False:
            print('Hello World!')
        string_file = BlockingBytesWrite()
        consumer = BackgroundFileConsumer(cast(BinaryIO, string_file), reactor=reactor)
        try:
            producer = NonCallableMock(spec_set=[])
            consumer.registerProducer(producer, True)
            consumer.write(b'Foo')
            yield string_file.wait_for_n_writes(1)
            self.assertEqual(string_file.buffer, b'Foo')
            consumer.write(b'Bar')
            yield string_file.wait_for_n_writes(2)
            self.assertEqual(string_file.buffer, b'FooBar')
        finally:
            consumer.unregisterProducer()
        yield consumer.wait()
        self.assertTrue(string_file.closed)

    @defer.inlineCallbacks
    def test_push_producer_feedback(self) -> Generator['defer.Deferred[object]', object, None]:
        if False:
            print('Hello World!')
        string_file = BlockingBytesWrite()
        consumer = BackgroundFileConsumer(cast(BinaryIO, string_file), reactor=reactor)
        try:
            producer = NonCallableMock(spec_set=['pauseProducing', 'resumeProducing'])
            resume_deferred: defer.Deferred = defer.Deferred()
            producer.resumeProducing.side_effect = lambda : resume_deferred.callback(None)
            consumer.registerProducer(producer, True)
            number_writes = 0
            with string_file.write_lock:
                for _ in range(consumer._PAUSE_ON_QUEUE_SIZE):
                    consumer.write(b'Foo')
                    number_writes += 1
                producer.pauseProducing.assert_called_once()
            yield string_file.wait_for_n_writes(number_writes)
            yield resume_deferred
            producer.resumeProducing.assert_called_once()
        finally:
            consumer.unregisterProducer()
        yield consumer.wait()
        self.assertTrue(string_file.closed)

@implementer(IPullProducer)
class DummyPullProducer:

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.consumer: Optional[BackgroundFileConsumer] = None
        self.deferred: 'defer.Deferred[object]' = defer.Deferred()

    def resumeProducing(self) -> None:
        if False:
            while True:
                i = 10
        d = self.deferred
        self.deferred = defer.Deferred()
        d.callback(None)

    def stopProducing(self) -> None:
        if False:
            while True:
                i = 10
        raise RuntimeError('Unexpected call')

    def write_and_wait(self, write_bytes: bytes) -> 'defer.Deferred[object]':
        if False:
            for i in range(10):
                print('nop')
        assert self.consumer is not None
        d = self.deferred
        self.consumer.write(write_bytes)
        return d

    def register_with_consumer(self, consumer: BackgroundFileConsumer) -> 'defer.Deferred[object]':
        if False:
            for i in range(10):
                print('nop')
        d = self.deferred
        self.consumer = consumer
        self.consumer.registerProducer(self, False)
        return d

class BlockingBytesWrite:

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        self.buffer = b''
        self.closed = False
        self.write_lock = threading.Lock()
        self._notify_write_deferred: Optional[defer.Deferred] = None
        self._number_of_writes = 0

    def write(self, write_bytes: bytes) -> None:
        if False:
            print('Hello World!')
        with self.write_lock:
            self.buffer += write_bytes
            self._number_of_writes += 1
        reactor.callFromThread(self._notify_write)

    def close(self) -> None:
        if False:
            print('Hello World!')
        self.closed = True

    def _notify_write(self) -> None:
        if False:
            return 10
        'Called by write to indicate a write happened'
        with self.write_lock:
            if not self._notify_write_deferred:
                return
            d = self._notify_write_deferred
            self._notify_write_deferred = None
        d.callback(None)

    @defer.inlineCallbacks
    def wait_for_n_writes(self, n: int) -> Generator['defer.Deferred[object]', object, None]:
        if False:
            for i in range(10):
                print('nop')
        'Wait for n writes to have happened'
        while True:
            with self.write_lock:
                if n <= self._number_of_writes:
                    return
                if not self._notify_write_deferred:
                    self._notify_write_deferred = defer.Deferred()
                d = self._notify_write_deferred
            yield d