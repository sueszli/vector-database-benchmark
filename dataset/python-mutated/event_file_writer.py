"""Writes events to disk in a logdir."""
import collections
import os.path
import sys
import threading
import time
from tensorflow.python.client import _pywrap_events_writer
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat

class EventFileWriter:
    """Writes `Event` protocol buffers to an event file.

  The `EventFileWriter` class creates an event file in the specified directory,
  and asynchronously writes Event protocol buffers to the file. The Event file
  is encoded using the tfrecord format, which is similar to RecordIO.

  This class is not thread-safe.
  """

    def __init__(self, logdir, max_queue=10, flush_secs=120, filename_suffix=None):
        if False:
            while True:
                i = 10
        "Creates a `EventFileWriter` and an event file to write to.\n\n    On construction the summary writer creates a new event file in `logdir`.\n    This event file will contain `Event` protocol buffers, which are written to\n    disk via the add_event method.\n\n    The other arguments to the constructor control the asynchronous writes to\n    the event file:\n\n    *  `flush_secs`: How often, in seconds, to flush the added summaries\n       and events to disk.\n    *  `max_queue`: Maximum number of summaries or events pending to be\n       written to disk before one of the 'add' calls block.\n\n    Args:\n      logdir: A string. Directory where event file will be written.\n      max_queue: Integer. Size of the queue for pending events and summaries.\n      flush_secs: Number. How often, in seconds, to flush the\n        pending events and summaries to disk.\n      filename_suffix: A string. Every event file's name is suffixed with\n        `filename_suffix`.\n    "
        self._logdir = str(logdir)
        gfile.MakeDirs(self._logdir)
        self._max_queue = max_queue
        self._flush_secs = flush_secs
        self._flush_complete = threading.Event()
        self._flush_sentinel = object()
        self._close_sentinel = object()
        self._ev_writer = _pywrap_events_writer.EventsWriter(compat.as_bytes(os.path.join(self._logdir, 'events')))
        if filename_suffix:
            self._ev_writer.InitWithSuffix(compat.as_bytes(filename_suffix))
        self._initialize()
        self._closed = False

    def _initialize(self):
        if False:
            for i in range(10):
                print('nop')
        'Initializes or re-initializes the queue and writer thread.\n\n    The EventsWriter itself does not need to be re-initialized explicitly,\n    because it will auto-initialize itself if used after being closed.\n    '
        self._event_queue = CloseableQueue(self._max_queue)
        self._worker = _EventLoggerThread(self._event_queue, self._ev_writer, self._flush_secs, self._flush_complete, self._flush_sentinel, self._close_sentinel)
        self._worker.start()

    def get_logdir(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns the directory where event file will be written.'
        return self._logdir

    def reopen(self):
        if False:
            i = 10
            return i + 15
        'Reopens the EventFileWriter.\n\n    Can be called after `close()` to add more events in the same directory.\n    The events will go into a new events file.\n\n    Does nothing if the EventFileWriter was not closed.\n    '
        if self._closed:
            self._initialize()
            self._closed = False

    def add_event(self, event):
        if False:
            while True:
                i = 10
        'Adds an event to the event file.\n\n    Args:\n      event: An `Event` protocol buffer.\n    '
        if not self._closed:
            self._try_put(event)

    def _try_put(self, item):
        if False:
            i = 10
            return i + 15
        'Attempts to enqueue an item to the event queue.\n\n    If the queue is closed, this will close the EventFileWriter and reraise the\n    exception that caused the queue closure, if one exists.\n\n    Args:\n      item: the item to enqueue\n    '
        try:
            self._event_queue.put(item)
        except QueueClosedError:
            self._internal_close()
            if self._worker.failure_exc_info:
                (_, exception, _) = self._worker.failure_exc_info
                raise exception from None

    def flush(self):
        if False:
            for i in range(10):
                print('nop')
        'Flushes the event file to disk.\n\n    Call this method to make sure that all pending events have been written to\n    disk.\n    '
        if not self._closed:
            self._flush_complete.clear()
            self._try_put(self._flush_sentinel)
            self._flush_complete.wait()
            if self._worker.failure_exc_info:
                self._internal_close()
                (_, exception, _) = self._worker.failure_exc_info
                raise exception

    def close(self):
        if False:
            i = 10
            return i + 15
        'Flushes the event file to disk and close the file.\n\n    Call this method when you do not need the summary writer anymore.\n    '
        if not self._closed:
            self.flush()
            self._try_put(self._close_sentinel)
            self._internal_close()

    def _internal_close(self):
        if False:
            return 10
        self._closed = True
        self._worker.join()
        self._ev_writer.Close()

class _EventLoggerThread(threading.Thread):
    """Thread that logs events."""

    def __init__(self, queue, ev_writer, flush_secs, flush_complete, flush_sentinel, close_sentinel):
        if False:
            while True:
                i = 10
        'Creates an _EventLoggerThread.\n\n    Args:\n      queue: A CloseableQueue from which to dequeue events. The queue will be\n        closed just before the thread exits, whether due to `close_sentinel` or\n        any exception raised in the writing loop.\n      ev_writer: An event writer. Used to log brain events for\n        the visualizer.\n      flush_secs: How often, in seconds, to flush the\n        pending file to disk.\n      flush_complete: A threading.Event that will be set whenever a flush\n        operation requested via `flush_sentinel` has been completed.\n      flush_sentinel: A sentinel element in queue that tells this thread to\n        flush the writer and mark the current flush operation complete.\n      close_sentinel: A sentinel element in queue that tells this thread to\n        terminate and close the queue.\n    '
        threading.Thread.__init__(self, name='EventLoggerThread')
        self.daemon = True
        self._queue = queue
        self._ev_writer = ev_writer
        self._flush_secs = flush_secs
        self._next_event_flush_time = 0
        self._flush_complete = flush_complete
        self._flush_sentinel = flush_sentinel
        self._close_sentinel = close_sentinel
        self.failure_exc_info = ()

    def run(self):
        if False:
            while True:
                i = 10
        try:
            while True:
                event = self._queue.get()
                if event is self._close_sentinel:
                    return
                elif event is self._flush_sentinel:
                    self._ev_writer.Flush()
                    self._flush_complete.set()
                else:
                    self._ev_writer.WriteEvent(event)
                    now = time.time()
                    if now > self._next_event_flush_time:
                        self._ev_writer.Flush()
                        self._next_event_flush_time = now + self._flush_secs
        except Exception as e:
            logging.error('EventFileWriter writer thread error: %s', e)
            self.failure_exc_info = sys.exc_info()
            raise
        finally:
            self._flush_complete.set()
            self._queue.close()

class CloseableQueue:
    """Stripped-down fork of the standard library Queue that is closeable."""

    def __init__(self, maxsize=0):
        if False:
            return 10
        'Create a queue object with a given maximum size.\n\n    Args:\n      maxsize: int size of queue. If <= 0, the queue size is infinite.\n    '
        self._maxsize = maxsize
        self._queue = collections.deque()
        self._closed = False
        self._mutex = threading.Lock()
        self._not_empty = threading.Condition(self._mutex)
        self._not_full = threading.Condition(self._mutex)

    def get(self):
        if False:
            i = 10
            return i + 15
        'Remove and return an item from the queue.\n\n    If the queue is empty, blocks until an item is available.\n\n    Returns:\n      an item from the queue\n    '
        with self._not_empty:
            while not self._queue:
                self._not_empty.wait()
            item = self._queue.popleft()
            self._not_full.notify()
            return item

    def put(self, item):
        if False:
            while True:
                i = 10
        'Put an item into the queue.\n\n    If the queue is closed, fails immediately.\n\n    If the queue is full, blocks until space is available or until the queue\n    is closed by a call to close(), at which point this call fails.\n\n    Args:\n      item: an item to add to the queue\n\n    Raises:\n      QueueClosedError: if insertion failed because the queue is closed\n    '
        with self._not_full:
            if self._closed:
                raise QueueClosedError()
            if self._maxsize > 0:
                while len(self._queue) == self._maxsize:
                    self._not_full.wait()
                    if self._closed:
                        raise QueueClosedError()
            self._queue.append(item)
            self._not_empty.notify()

    def close(self):
        if False:
            print('Hello World!')
        'Closes the queue, causing any pending or future `put()` calls to fail.'
        with self._not_full:
            self._closed = True
            self._not_full.notify_all()

class QueueClosedError(Exception):
    """Raised when CloseableQueue.put() fails because the queue is closed."""