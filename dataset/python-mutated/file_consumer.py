import queue
from typing import Any, BinaryIO, Optional, Union, cast
from twisted.internet import threads
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IPullProducer, IPushProducer
from synapse.logging.context import make_deferred_yieldable, run_in_background
from synapse.types import ISynapseReactor

class BackgroundFileConsumer:
    """A consumer that writes to a file like object. Supports both push
    and pull producers

    Args:
        file_obj: The file like object to write to. Closed when
            finished.
        reactor: the Twisted reactor to use
    """
    _PAUSE_ON_QUEUE_SIZE = 5
    _RESUME_ON_QUEUE_SIZE = 2

    def __init__(self, file_obj: BinaryIO, reactor: ISynapseReactor) -> None:
        if False:
            while True:
                i = 10
        self._file_obj: BinaryIO = file_obj
        self._reactor: ISynapseReactor = reactor
        self._producer: Optional[Union[IPushProducer, IPullProducer]] = None
        self.streaming = False
        self._paused_producer = False
        self._bytes_queue: queue.Queue[Optional[bytes]] = queue.Queue()
        self._finished_deferred: Optional[Deferred[Any]] = None
        self._write_exception: Optional[Exception] = None

    def registerProducer(self, producer: Union[IPushProducer, IPullProducer], streaming: bool) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Part of IConsumer interface\n\n        Args:\n            producer\n            streaming: True if push based producer, False if pull\n                based.\n        '
        if self._producer:
            raise Exception('registerProducer called twice')
        self._producer = producer
        self.streaming = streaming
        self._finished_deferred = run_in_background(threads.deferToThreadPool, self._reactor, self._reactor.getThreadPool(), self._writer)
        if not streaming:
            self._producer.resumeProducing()

    def unregisterProducer(self) -> None:
        if False:
            i = 10
            return i + 15
        'Part of IProducer interface'
        self._producer = None
        assert self._finished_deferred is not None
        if not self._finished_deferred.called:
            self._bytes_queue.put_nowait(None)

    def write(self, write_bytes: bytes) -> None:
        if False:
            i = 10
            return i + 15
        'Part of IProducer interface'
        if self._write_exception:
            raise self._write_exception
        assert self._finished_deferred is not None
        if self._finished_deferred.called:
            raise Exception('consumer has closed')
        self._bytes_queue.put_nowait(write_bytes)
        if self.streaming and self._bytes_queue.qsize() >= self._PAUSE_ON_QUEUE_SIZE:
            self._paused_producer = True
            assert self._producer is not None
            cast(IPushProducer, self._producer).pauseProducing()

    def _writer(self) -> None:
        if False:
            print('Hello World!')
        'This is run in a background thread to write to the file.'
        try:
            while self._producer or not self._bytes_queue.empty():
                if self._producer and self._paused_producer:
                    if self._bytes_queue.qsize() <= self._RESUME_ON_QUEUE_SIZE:
                        self._reactor.callFromThread(self._resume_paused_producer)
                bytes = self._bytes_queue.get()
                if bytes:
                    self._file_obj.write(bytes)
                if not self.streaming and self._producer:
                    self._reactor.callFromThread(self._producer.resumeProducing)
        except Exception as e:
            self._write_exception = e
            raise
        finally:
            self._file_obj.close()

    def wait(self) -> 'Deferred[None]':
        if False:
            print('Hello World!')
        'Returns a deferred that resolves when finished writing to file'
        assert self._finished_deferred is not None
        return make_deferred_yieldable(self._finished_deferred)

    def _resume_paused_producer(self) -> None:
        if False:
            return 10
        'Gets called if we should resume producing after being paused'
        if self._paused_producer and self._producer:
            self._paused_producer = False
            self._producer.resumeProducing()