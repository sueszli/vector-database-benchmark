from __future__ import annotations
import logging
import queue
from concurrent import futures
from concurrent.futures import Future
from threading import Event, Thread, current_thread
from typing import ClassVar, Generator, Generic, Optional, Tuple, Type, TypeVar
from streamlink.buffers import RingBuffer
from streamlink.stream.segmented.concurrent import ThreadPoolExecutor
from streamlink.stream.segmented.segment import Segment
from streamlink.stream.stream import Stream, StreamIO
try:
    from typing import TypeAlias
except ImportError:
    from typing_extensions import TypeAlias
log = logging.getLogger('.'.join(__name__.split('.')[:-1]))

class AwaitableMixin:

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)
        self._wait = Event()

    def wait(self, time: float) -> bool:
        if False:
            while True:
                i = 10
        '\n        Pause the thread for a specified time.\n        Return False if interrupted by another thread and True if the time runs out normally.\n        '
        return not self._wait.wait(time)
TSegment = TypeVar('TSegment', bound=Segment)
TResult = TypeVar('TResult')
TResultFuture: TypeAlias = 'Future[Optional[TResult]]'
TQueueItem: TypeAlias = Optional[Tuple[TSegment, TResultFuture, Tuple]]

class SegmentedStreamWriter(AwaitableMixin, Thread, Generic[TSegment, TResult]):
    """
    The base writer thread.
    This thread is responsible for fetching segments, processing them and finally writing the data to the buffer.
    """
    reader: SegmentedStreamReader[TSegment, TResult]
    stream: Stream

    def __init__(self, reader: SegmentedStreamReader, size: int=20, retries: Optional[int]=None, threads: Optional[int]=None, timeout: Optional[float]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(daemon=True, name=f'Thread-{self.__class__.__name__}')
        self.closed = False
        self.reader = reader
        self.stream = reader.stream
        self.session = reader.session
        self.retries = retries or self.session.options.get('stream-segment-attempts')
        self.threads = threads or self.session.options.get('stream-segment-threads')
        self.timeout = timeout or self.session.options.get('stream-segment-timeout')
        self.executor = ThreadPoolExecutor(max_workers=self.threads)
        self._queue: queue.Queue[TQueueItem] = queue.Queue(size)

    def close(self) -> None:
        if False:
            print('Hello World!')
        '\n        Shuts down the thread, its executor and closes the reader (worker thread and buffer).\n        '
        if self.closed:
            return
        log.debug('Closing writer thread')
        self.closed = True
        self._wait.set()
        self.reader.close()
        self.executor.shutdown(wait=True, cancel_futures=True)

    def put(self, segment: Optional[TSegment]) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Adds a segment to the download pool and write queue.\n        '
        if self.closed:
            return
        future: Optional[TResultFuture]
        if segment is None:
            future = None
        else:
            future = self.executor.submit(self.fetch, segment)
        self.queue(segment, future)

    def queue(self, segment: Optional[TSegment], future: Optional[TResultFuture], *data) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Puts values into a queue but aborts if this thread is closed.\n        '
        item = None if segment is None or future is None else (segment, future, data)
        while not self.closed:
            try:
                self._queue_put(item)
                return
            except queue.Full:
                continue

    def _queue_put(self, item: TQueueItem) -> None:
        if False:
            return 10
        self._queue.put(item, block=True, timeout=1)

    def _queue_get(self) -> TQueueItem:
        if False:
            while True:
                i = 10
        return self._queue.get(block=True, timeout=0.5)

    @staticmethod
    def _future_result(future: TResultFuture) -> Optional[TResult]:
        if False:
            for i in range(10):
                print('nop')
        return future.result(timeout=0.5)

    def fetch(self, segment: TSegment) -> Optional[TResult]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Fetches a segment.\n        Should be overridden by the inheriting class.\n        '

    def write(self, segment: TSegment, result: TResult, *data) -> None:
        if False:
            while True:
                i = 10
        '\n        Writes a segment to the buffer.\n        Should be overridden by the inheriting class.\n        '

    def run(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        while not self.closed:
            try:
                item = self._queue_get()
            except queue.Empty:
                continue
            if item is None:
                break
            (segment, future, data) = item
            while not self.closed:
                try:
                    result = self._future_result(future)
                except futures.TimeoutError:
                    continue
                except futures.CancelledError:
                    break
                if result is not None:
                    self.write(segment, result, *data)
                break
        self.close()

class SegmentedStreamWorker(AwaitableMixin, Thread, Generic[TSegment, TResult]):
    """
    The base worker thread.
    This thread is responsible for queueing up segments in the writer thread.
    """
    reader: SegmentedStreamReader[TSegment, TResult]
    writer: SegmentedStreamWriter[TSegment, TResult]
    stream: Stream

    def __init__(self, reader: SegmentedStreamReader, **kwargs) -> None:
        if False:
            while True:
                i = 10
        super().__init__(daemon=True, name=f'Thread-{self.__class__.__name__}')
        self.closed = False
        self.reader = reader
        self.writer = reader.writer
        self.stream = reader.stream
        self.session = reader.session

    def close(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Shuts down the thread.\n        '
        if self.closed:
            return
        log.debug('Closing worker thread')
        self.closed = True
        self._wait.set()

    def iter_segments(self) -> Generator[TSegment, None, None]:
        if False:
            while True:
                i = 10
        '\n        The iterator that generates segments for the worker thread.\n        Should be overridden by the inheriting class.\n        '
        return
        yield

    def run(self) -> None:
        if False:
            while True:
                i = 10
        for segment in self.iter_segments():
            if self.closed:
                break
            self.writer.put(segment)
        self.writer.put(None)
        self.close()

class SegmentedStreamReader(StreamIO, Generic[TSegment, TResult]):
    __worker__: ClassVar[Type[SegmentedStreamWorker]] = SegmentedStreamWorker
    __writer__: ClassVar[Type[SegmentedStreamWriter]] = SegmentedStreamWriter
    worker: SegmentedStreamWorker[TSegment, TResult]
    writer: SegmentedStreamWriter[TSegment, TResult]
    stream: Stream

    def __init__(self, stream: Stream) -> None:
        if False:
            while True:
                i = 10
        super().__init__()
        self.stream = stream
        self.session = stream.session
        self.timeout = self.session.options.get('stream-timeout')
        buffer_size = self.session.get_option('ringbuffer-size')
        self.buffer = RingBuffer(buffer_size)
        self.writer = self.__writer__(self)
        self.worker = self.__worker__(self)

    def open(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.writer.start()
        self.worker.start()

    def close(self) -> None:
        if False:
            print('Hello World!')
        self.worker.close()
        self.writer.close()
        self.buffer.close()
        current = current_thread()
        if current is not self.worker:
            self.worker.join(timeout=self.timeout)
        if current is not self.writer:
            self.writer.join(timeout=self.timeout)
        super().close()

    def read(self, size: int) -> bytes:
        if False:
            i = 10
            return i + 15
        return self.buffer.read(size, block=self.writer.is_alive(), timeout=self.timeout)