"""A connection adapter that tries to use the best polling method for the
platform pika is running on.

"""
import abc
import collections
import errno
import heapq
import logging
import select
import time
import threading
import pika.compat
from pika.adapters.utils import nbio_interface
from pika.adapters.base_connection import BaseConnection
from pika.adapters.utils.selector_ioloop_adapter import SelectorIOServicesAdapter, AbstractSelectorIOLoop
LOGGER = logging.getLogger(__name__)
SELECT_TYPE = None
_SELECT_ERROR_CHECKERS = {}
if pika.compat.PY3:
    _SELECT_ERROR_CHECKERS[InterruptedError] = lambda e: True
_SELECT_ERROR_CHECKERS[select.error] = lambda e: e.args[0] == errno.EINTR
_SELECT_ERROR_CHECKERS[IOError] = lambda e: e.errno == errno.EINTR
_SELECT_ERROR_CHECKERS[OSError] = lambda e: e.errno == errno.EINTR
_SELECT_ERRORS = tuple(_SELECT_ERROR_CHECKERS.keys())

def _is_resumable(exc):
    if False:
        i = 10
        return i + 15
    'Check if caught exception represents EINTR error.\n    :param exc: exception; must be one of classes in _SELECT_ERRORS\n\n    '
    checker = _SELECT_ERROR_CHECKERS.get(exc.__class__, None)
    if checker is not None:
        return checker(exc)
    else:
        return False

class SelectConnection(BaseConnection):
    """An asynchronous connection adapter that attempts to use the fastest
    event loop adapter for the given platform.

    """

    def __init__(self, parameters=None, on_open_callback=None, on_open_error_callback=None, on_close_callback=None, custom_ioloop=None, internal_connection_workflow=True):
        if False:
            for i in range(10):
                print('nop')
        "Create a new instance of the Connection object.\n\n        :param pika.connection.Parameters parameters: Connection parameters\n        :param callable on_open_callback: Method to call on connection open\n        :param None | method on_open_error_callback: Called if the connection\n            can't be established or connection establishment is interrupted by\n            `Connection.close()`: on_open_error_callback(Connection, exception).\n        :param None | method on_close_callback: Called when a previously fully\n            open connection is closed:\n            `on_close_callback(Connection, exception)`, where `exception` is\n            either an instance of `exceptions.ConnectionClosed` if closed by\n            user or broker or exception of another type that describes the cause\n            of connection failure.\n        :param None | IOLoop | nbio_interface.AbstractIOServices custom_ioloop:\n            Provide a custom I/O Loop object.\n        :param bool internal_connection_workflow: True for autonomous connection\n            establishment which is default; False for externally-managed\n            connection workflow via the `create_connection()` factory.\n        :raises: RuntimeError\n\n        "
        if isinstance(custom_ioloop, nbio_interface.AbstractIOServices):
            nbio = custom_ioloop
        else:
            nbio = SelectorIOServicesAdapter(custom_ioloop or IOLoop())
        super().__init__(parameters, on_open_callback, on_open_error_callback, on_close_callback, nbio, internal_connection_workflow=internal_connection_workflow)

    @classmethod
    def create_connection(cls, connection_configs, on_done, custom_ioloop=None, workflow=None):
        if False:
            i = 10
            return i + 15
        'Implement\n        :py:classmethod::`pika.adapters.BaseConnection.create_connection()`.\n\n        '
        nbio = SelectorIOServicesAdapter(custom_ioloop or IOLoop())

        def connection_factory(params):
            if False:
                i = 10
                return i + 15
            'Connection factory.'
            if params is None:
                raise ValueError('Expected pika.connection.Parameters instance, but got None in params arg.')
            return cls(parameters=params, custom_ioloop=nbio, internal_connection_workflow=False)
        return cls._start_connection_workflow(connection_configs=connection_configs, connection_factory=connection_factory, nbio=nbio, workflow=workflow, on_done=on_done)

    def _get_write_buffer_size(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        :returns: Current size of output data buffered by the transport\n        :rtype: int\n        '
        return self._transport.get_write_buffer_size()

class _Timeout:
    """Represents a timeout"""
    __slots__ = ('deadline', 'callback')

    def __init__(self, deadline, callback):
        if False:
            return 10
        '\n        :param float deadline: timer expiration as non-negative epoch number\n        :param callable callback: callback to call when timeout expires\n        :raises ValueError, TypeError:\n        '
        if deadline < 0:
            raise ValueError('deadline must be non-negative epoch number, but got %r' % (deadline,))
        if not callable(callback):
            raise TypeError('callback must be a callable, but got {!r}'.format(callback))
        self.deadline = deadline
        self.callback = callback

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        'NOTE: not supporting sort stability'
        if isinstance(other, _Timeout):
            return self.deadline == other.deadline
        return NotImplemented

    def __ne__(self, other):
        if False:
            while True:
                i = 10
        'NOTE: not supporting sort stability'
        result = self.__eq__(other)
        if result is not NotImplemented:
            return not result
        return NotImplemented

    def __lt__(self, other):
        if False:
            while True:
                i = 10
        'NOTE: not supporting sort stability'
        if isinstance(other, _Timeout):
            return self.deadline < other.deadline
        return NotImplemented

    def __gt__(self, other):
        if False:
            while True:
                i = 10
        'NOTE: not supporting sort stability'
        if isinstance(other, _Timeout):
            return self.deadline > other.deadline
        return NotImplemented

    def __le__(self, other):
        if False:
            while True:
                i = 10
        'NOTE: not supporting sort stability'
        if isinstance(other, _Timeout):
            return self.deadline <= other.deadline
        return NotImplemented

    def __ge__(self, other):
        if False:
            i = 10
            return i + 15
        'NOTE: not supporting sort stability'
        if isinstance(other, _Timeout):
            return self.deadline >= other.deadline
        return NotImplemented

class _Timer:
    """Manage timeouts for use in ioloop"""
    _GC_CANCELLATION_THRESHOLD = 1024

    def __init__(self):
        if False:
            while True:
                i = 10
        self._timeout_heap = []
        self._num_cancellations = 0

    def close(self):
        if False:
            while True:
                i = 10
        "Release resources. Don't use the `_Timer` instance after closing\n        it\n        "
        if self._timeout_heap is not None:
            for timeout in self._timeout_heap:
                timeout.callback = None
            self._timeout_heap = None

    def call_later(self, delay, callback):
        if False:
            while True:
                i = 10
        'Schedule a one-shot timeout given delay seconds.\n\n        NOTE: you may cancel the timer before dispatch of the callback. Timer\n            Manager cancels the timer upon dispatch of the callback.\n\n        :param float delay: Non-negative number of seconds from now until\n            expiration\n        :param callable callback: The callback method, having the signature\n            `callback()`\n\n        :rtype: _Timeout\n        :raises ValueError, TypeError\n\n        '
        if self._timeout_heap is None:
            raise ValueError('Timeout closed before call')
        if delay < 0:
            raise ValueError('call_later: delay must be non-negative, but got {!r}'.format(delay))
        now = pika.compat.time_now()
        timeout = _Timeout(now + delay, callback)
        heapq.heappush(self._timeout_heap, timeout)
        LOGGER.debug('call_later: added timeout %r with deadline=%r and callback=%r; now=%s; delay=%s', timeout, timeout.deadline, timeout.callback, now, delay)
        return timeout

    def remove_timeout(self, timeout):
        if False:
            for i in range(10):
                print('nop')
        'Cancel the timeout\n\n        :param _Timeout timeout: The timer to cancel\n\n        '
        if timeout.callback is None:
            LOGGER.debug('remove_timeout: timeout was already removed or called %r', timeout)
        else:
            LOGGER.debug('remove_timeout: removing timeout %r with deadline=%r and callback=%r', timeout, timeout.deadline, timeout.callback)
            timeout.callback = None
            self._num_cancellations += 1

    def get_remaining_interval(self):
        if False:
            while True:
                i = 10
        'Get the interval to the next timeout expiration\n\n        :returns: non-negative number of seconds until next timer expiration;\n                  None if there are no timers\n        :rtype: float\n\n        '
        if self._timeout_heap:
            now = pika.compat.time_now()
            interval = max(0, self._timeout_heap[0].deadline - now)
        else:
            interval = None
        return interval

    def process_timeouts(self):
        if False:
            i = 10
            return i + 15
        'Process pending timeouts, invoking callbacks for those whose time has\n        come\n\n        '
        if self._timeout_heap:
            now = pika.compat.time_now()
            ready_timeouts = []
            while self._timeout_heap and self._timeout_heap[0].deadline <= now:
                timeout = heapq.heappop(self._timeout_heap)
                if timeout.callback is not None:
                    ready_timeouts.append(timeout)
                else:
                    self._num_cancellations -= 1
            for timeout in ready_timeouts:
                if timeout.callback is None:
                    self._num_cancellations -= 1
                    continue
                timeout.callback()
                timeout.callback = None
            if self._num_cancellations >= self._GC_CANCELLATION_THRESHOLD and self._num_cancellations > len(self._timeout_heap) >> 1:
                self._num_cancellations = 0
                self._timeout_heap = [t for t in self._timeout_heap if t.callback is not None]
                heapq.heapify(self._timeout_heap)

class PollEvents:
    """Event flags for I/O"""
    READ = getattr(select, 'POLLIN', 1)
    WRITE = getattr(select, 'POLLOUT', 4)
    ERROR = getattr(select, 'POLLERR', 8)

class IOLoop(AbstractSelectorIOLoop):
    """I/O loop implementation that picks a suitable poller (`select`,
     `poll`, `epoll`, `kqueue`) to use based on platform.

     Implements the
     `pika.adapters.utils.selector_ioloop_adapter.AbstractSelectorIOLoop`
     interface.

    """
    READ = PollEvents.READ
    WRITE = PollEvents.WRITE
    ERROR = PollEvents.ERROR

    def __init__(self):
        if False:
            while True:
                i = 10
        self._timer = _Timer()
        self._callbacks = collections.deque()
        self._poller = self._get_poller(self._get_remaining_interval, self.process_timeouts)

    def close(self):
        if False:
            return 10
        "Release IOLoop's resources.\n\n        `IOLoop.close` is intended to be called by the application or test code\n        only after `IOLoop.start()` returns. After calling `close()`, no other\n        interaction with the closed instance of `IOLoop` should be performed.\n\n        "
        if self._callbacks is not None:
            self._poller.close()
            self._timer.close()
            self._callbacks = []

    @staticmethod
    def _get_poller(get_wait_seconds, process_timeouts):
        if False:
            print('Hello World!')
        'Determine the best poller to use for this environment and instantiate\n        it.\n\n        :param get_wait_seconds: Function for getting the maximum number of\n                                 seconds to wait for IO for use by the poller\n        :param process_timeouts: Function for processing timeouts for use by the\n                                 poller\n\n        :returns: The instantiated poller instance supporting `_PollerBase` API\n        :rtype: object\n        '
        poller = None
        kwargs = dict(get_wait_seconds=get_wait_seconds, process_timeouts=process_timeouts)
        if hasattr(select, 'epoll'):
            if not SELECT_TYPE or SELECT_TYPE == 'epoll':
                LOGGER.debug('Using EPollPoller')
                poller = EPollPoller(**kwargs)
        if not poller and hasattr(select, 'kqueue'):
            if not SELECT_TYPE or SELECT_TYPE == 'kqueue':
                LOGGER.debug('Using KQueuePoller')
                poller = KQueuePoller(**kwargs)
        if not poller and hasattr(select, 'poll') and hasattr(select.poll(), 'modify'):
            if not SELECT_TYPE or SELECT_TYPE == 'poll':
                LOGGER.debug('Using PollPoller')
                poller = PollPoller(**kwargs)
        if not poller:
            LOGGER.debug('Using SelectPoller')
            poller = SelectPoller(**kwargs)
        return poller

    def call_later(self, delay, callback):
        if False:
            print('Hello World!')
        'Add the callback to the IOLoop timer to be called after delay seconds\n        from the time of call on best-effort basis. Returns a handle to the\n        timeout.\n\n        :param float delay: The number of seconds to wait to call callback\n        :param callable callback: The callback method\n        :returns: handle to the created timeout that may be passed to\n            `remove_timeout()`\n        :rtype: object\n\n        '
        return self._timer.call_later(delay, callback)

    def remove_timeout(self, timeout_handle):
        if False:
            return 10
        'Remove a timeout\n\n        :param timeout_handle: Handle of timeout to remove\n\n        '
        self._timer.remove_timeout(timeout_handle)

    def add_callback_threadsafe(self, callback):
        if False:
            for i in range(10):
                print('nop')
        "Requests a call to the given function as soon as possible in the\n        context of this IOLoop's thread.\n\n        NOTE: This is the only thread-safe method in IOLoop. All other\n        manipulations of IOLoop must be performed from the IOLoop's thread.\n\n        For example, a thread may request a call to the `stop` method of an\n        ioloop that is running in a different thread via\n        `ioloop.add_callback_threadsafe(ioloop.stop)`\n\n        :param callable callback: The callback method\n\n        "
        if not callable(callback):
            raise TypeError('callback must be a callable, but got {!r}'.format(callback))
        self._callbacks.append(callback)
        self._poller.wake_threadsafe()
        LOGGER.debug('add_callback_threadsafe: added callback=%r', callback)
    add_callback = add_callback_threadsafe

    def process_timeouts(self):
        if False:
            return 10
        '[Extension] Process pending callbacks and timeouts, invoking those\n        whose time has come. Internal use only.\n\n        '
        for _ in pika.compat.xrange(len(self._callbacks)):
            callback = self._callbacks.popleft()
            LOGGER.debug('process_timeouts: invoking callback=%r', callback)
            callback()
        self._timer.process_timeouts()

    def _get_remaining_interval(self):
        if False:
            while True:
                i = 10
        'Get the remaining interval to the next callback or timeout\n        expiration.\n\n        :returns: non-negative number of seconds until next callback or timer\n                  expiration; None if there are no callbacks and timers\n        :rtype: float\n\n        '
        if self._callbacks:
            return 0
        return self._timer.get_remaining_interval()

    def add_handler(self, fd, handler, events):
        if False:
            while True:
                i = 10
        'Start watching the given file descriptor for events\n\n        :param int fd: The file descriptor\n        :param callable handler: When requested event(s) occur,\n            `handler(fd, events)` will be called.\n        :param int events: The event mask using READ, WRITE, ERROR.\n\n        '
        self._poller.add_handler(fd, handler, events)

    def update_handler(self, fd, events):
        if False:
            return 10
        'Changes the events we watch for\n\n        :param int fd: The file descriptor\n        :param int events: The event mask using READ, WRITE, ERROR\n\n        '
        self._poller.update_handler(fd, events)

    def remove_handler(self, fd):
        if False:
            return 10
        'Stop watching the given file descriptor for events\n\n        :param int fd: The file descriptor\n\n        '
        self._poller.remove_handler(fd)

    def start(self):
        if False:
            while True:
                i = 10
        '[API] Start the main poller loop. It will loop until requested to\n        exit. See `IOLoop.stop`.\n\n        '
        self._poller.start()

    def stop(self):
        if False:
            return 10
        "[API] Request exit from the ioloop. The loop is NOT guaranteed to\n        stop before this method returns.\n\n        To invoke `stop()` safely from a thread other than this IOLoop's thread,\n        call it via `add_callback_threadsafe`; e.g.,\n\n            `ioloop.add_callback_threadsafe(ioloop.stop)`\n\n        "
        self._poller.stop()

    def activate_poller(self):
        if False:
            print('Hello World!')
        '[Extension] Activate the poller\n\n        '
        self._poller.activate_poller()

    def deactivate_poller(self):
        if False:
            return 10
        '[Extension] Deactivate the poller\n\n        '
        self._poller.deactivate_poller()

    def poll(self):
        if False:
            for i in range(10):
                print('nop')
        '[Extension] Wait for events of interest on registered file\n        descriptors until an event of interest occurs or next timer deadline or\n        `_PollerBase._MAX_POLL_TIMEOUT`, whichever is sooner, and dispatch the\n        corresponding event handlers.\n\n        '
        self._poller.poll()

class _PollerBase(pika.compat.AbstractBase):
    """Base class for select-based IOLoop implementations"""
    _MAX_POLL_TIMEOUT = 5
    POLL_TIMEOUT_MULT = 1

    def __init__(self, get_wait_seconds, process_timeouts):
        if False:
            i = 10
            return i + 15
        '\n        :param get_wait_seconds: Function for getting the maximum number of\n                                 seconds to wait for IO for use by the poller\n        :param process_timeouts: Function for processing timeouts for use by the\n                                 poller\n\n        '
        self._get_wait_seconds = get_wait_seconds
        self._process_timeouts = process_timeouts
        self._waking_mutex = threading.Lock()
        self._fd_handlers = dict()
        self._fd_events = {PollEvents.READ: set(), PollEvents.WRITE: set(), PollEvents.ERROR: set()}
        self._processing_fd_event_map = {}
        self._running = False
        self._stopping = False
        (self._r_interrupt, self._w_interrupt) = self._get_interrupt_pair()
        self.add_handler(self._r_interrupt.fileno(), self._read_interrupt, PollEvents.READ)

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        "Release poller's resources.\n\n        `close()` is intended to be called after the poller's `start()` method\n        returns. After calling `close()`, no other interaction with the closed\n        poller instance should be performed.\n\n        "
        assert not self._running, 'Cannot call close() before start() unwinds.'
        with self._waking_mutex:
            if self._w_interrupt is not None:
                self.remove_handler(self._r_interrupt.fileno())
                self._r_interrupt.close()
                self._r_interrupt = None
                self._w_interrupt.close()
                self._w_interrupt = None
        self.deactivate_poller()
        self._fd_handlers = None
        self._fd_events = None
        self._processing_fd_event_map = None

    def wake_threadsafe(self):
        if False:
            while True:
                i = 10
        'Wake up the poller as soon as possible. As the name indicates, this\n        method is thread-safe.\n\n        '
        with self._waking_mutex:
            if self._w_interrupt is None:
                return
            try:
                self._w_interrupt.send(b'X')
            except pika.compat.SOCKET_ERROR as err:
                if err.errno != errno.EWOULDBLOCK:
                    raise
            except Exception as err:
                LOGGER.warning('Failed to send interrupt to poller: %s', err)
                raise

    def _get_max_wait(self):
        if False:
            i = 10
            return i + 15
        'Get the interval to the next timeout event, or a default interval\n\n        :returns: maximum number of self.POLL_TIMEOUT_MULT-scaled time units\n                  to wait for IO events\n        :rtype: int\n\n        '
        delay = self._get_wait_seconds()
        if delay is None:
            delay = self._MAX_POLL_TIMEOUT
        else:
            delay = min(delay, self._MAX_POLL_TIMEOUT)
        return delay * self.POLL_TIMEOUT_MULT

    def add_handler(self, fileno, handler, events):
        if False:
            while True:
                i = 10
        'Add a new fileno to the set to be monitored\n\n        :param int fileno: The file descriptor\n        :param callable handler: What is called when an event happens\n        :param int events: The event mask using READ, WRITE, ERROR\n\n        '
        self._fd_handlers[fileno] = handler
        self._set_handler_events(fileno, events)
        self._register_fd(fileno, events)

    def update_handler(self, fileno, events):
        if False:
            i = 10
            return i + 15
        'Set the events to the current events\n\n        :param int fileno: The file descriptor\n        :param int events: The event mask using READ, WRITE, ERROR\n\n        '
        (events_cleared, events_set) = self._set_handler_events(fileno, events)
        self._modify_fd_events(fileno, events=events, events_to_clear=events_cleared, events_to_set=events_set)

    def remove_handler(self, fileno):
        if False:
            while True:
                i = 10
        'Remove a file descriptor from the set\n\n        :param int fileno: The file descriptor\n\n        '
        try:
            del self._processing_fd_event_map[fileno]
        except KeyError:
            pass
        (events_cleared, _) = self._set_handler_events(fileno, 0)
        del self._fd_handlers[fileno]
        self._unregister_fd(fileno, events_to_clear=events_cleared)

    def _set_handler_events(self, fileno, events):
        if False:
            while True:
                i = 10
        "Set the handler's events to the given events; internal to\n        `_PollerBase`.\n\n        :param int fileno: The file descriptor\n        :param int events: The event mask (READ, WRITE, ERROR)\n\n        :returns: a 2-tuple (events_cleared, events_set)\n        :rtype: tuple\n        "
        events_cleared = 0
        events_set = 0
        for evt in (PollEvents.READ, PollEvents.WRITE, PollEvents.ERROR):
            if events & evt:
                if fileno not in self._fd_events[evt]:
                    self._fd_events[evt].add(fileno)
                    events_set |= evt
            elif fileno in self._fd_events[evt]:
                self._fd_events[evt].discard(fileno)
                events_cleared |= evt
        return (events_cleared, events_set)

    def activate_poller(self):
        if False:
            print('Hello World!')
        'Activate the poller\n\n        '
        self._init_poller()
        fd_to_events = collections.defaultdict(int)
        for (event, file_descriptors) in self._fd_events.items():
            for fileno in file_descriptors:
                fd_to_events[fileno] |= event
        for (fileno, events) in fd_to_events.items():
            self._register_fd(fileno, events)

    def deactivate_poller(self):
        if False:
            for i in range(10):
                print('nop')
        'Deactivate the poller\n\n        '
        self._uninit_poller()

    def start(self):
        if False:
            for i in range(10):
                print('nop')
        'Start the main poller loop. It will loop until requested to exit.\n        This method is not reentrant and will raise an error if called\n        recursively (pika/pika#1095)\n\n        :raises: RuntimeError\n\n        '
        if self._running:
            raise RuntimeError('IOLoop is not reentrant and is already running')
        LOGGER.debug('Entering IOLoop')
        self._running = True
        self.activate_poller()
        try:
            while not self._stopping:
                self.poll()
                self._process_timeouts()
        finally:
            try:
                LOGGER.debug('Deactivating poller')
                self.deactivate_poller()
            finally:
                self._stopping = False
                self._running = False

    def stop(self):
        if False:
            for i in range(10):
                print('nop')
        'Request exit from the ioloop. The loop is NOT guaranteed to stop\n        before this method returns.\n\n        '
        LOGGER.debug('Stopping IOLoop')
        self._stopping = True
        self.wake_threadsafe()

    @abc.abstractmethod
    def poll(self):
        if False:
            i = 10
            return i + 15
        'Wait for events on interested filedescriptors.\n        '
        raise NotImplementedError

    @abc.abstractmethod
    def _init_poller(self):
        if False:
            return 10
        'Notify the implementation to allocate the poller resource'
        raise NotImplementedError

    @abc.abstractmethod
    def _uninit_poller(self):
        if False:
            for i in range(10):
                print('nop')
        'Notify the implementation to release the poller resource'
        raise NotImplementedError

    @abc.abstractmethod
    def _register_fd(self, fileno, events):
        if False:
            i = 10
            return i + 15
        'The base class invokes this method to notify the implementation to\n        register the file descriptor with the polling object. The request must\n        be ignored if the poller is not activated.\n\n        :param int fileno: The file descriptor\n        :param int events: The event mask (READ, WRITE, ERROR)\n        '
        raise NotImplementedError

    @abc.abstractmethod
    def _modify_fd_events(self, fileno, events, events_to_clear, events_to_set):
        if False:
            return 10
        'The base class invoikes this method to notify the implementation to\n        modify an already registered file descriptor. The request must be\n        ignored if the poller is not activated.\n\n        :param int fileno: The file descriptor\n        :param int events: absolute events (READ, WRITE, ERROR)\n        :param int events_to_clear: The events to clear (READ, WRITE, ERROR)\n        :param int events_to_set: The events to set (READ, WRITE, ERROR)\n        '
        raise NotImplementedError

    @abc.abstractmethod
    def _unregister_fd(self, fileno, events_to_clear):
        if False:
            for i in range(10):
                print('nop')
        'The base class invokes this method to notify the implementation to\n        unregister the file descriptor being tracked by the polling object. The\n        request must be ignored if the poller is not activated.\n\n        :param int fileno: The file descriptor\n        :param int events_to_clear: The events to clear (READ, WRITE, ERROR)\n        '
        raise NotImplementedError

    def _dispatch_fd_events(self, fd_event_map):
        if False:
            while True:
                i = 10
        ' Helper to dispatch callbacks for file descriptors that received\n        events.\n\n        Before doing so we re-calculate the event mask based on what is\n        currently set in case it has been changed under our feet by a\n        previous callback. We also take a store a refernce to the\n        fd_event_map so that we can detect removal of an\n        fileno during processing of another callback and not generate\n        spurious callbacks on it.\n\n        :param dict fd_event_map: Map of fds to events received on them.\n        '
        self._processing_fd_event_map.clear()
        self._processing_fd_event_map = fd_event_map
        for fileno in pika.compat.dictkeys(fd_event_map):
            if fileno not in fd_event_map:
                continue
            events = fd_event_map[fileno]
            for evt in [PollEvents.READ, PollEvents.WRITE, PollEvents.ERROR]:
                if fileno not in self._fd_events[evt]:
                    events &= ~evt
            if events:
                handler = self._fd_handlers[fileno]
                handler(fileno, events)

    @staticmethod
    def _get_interrupt_pair():
        if False:
            for i in range(10):
                print('nop')
        ' Use a socketpair to be able to interrupt the ioloop if called\n        from another thread. Socketpair() is not supported on some OS (Win)\n        so use a pair of simple TCP sockets instead. The sockets will be\n        closed and garbage collected by python when the ioloop itself is.\n        '
        return pika.compat._nonblocking_socketpair()

    def _read_interrupt(self, _interrupt_fd, _events):
        if False:
            print('Hello World!')
        " Read the interrupt byte(s). We ignore the event mask as we can ony\n        get here if there's data to be read on our fd.\n\n        :param int _interrupt_fd: (unused) The file descriptor to read from\n        :param int _events: (unused) The events generated for this fd\n        "
        try:
            self._r_interrupt.recv(512)
        except pika.compat.SOCKET_ERROR as err:
            if err.errno != errno.EAGAIN:
                raise

class SelectPoller(_PollerBase):
    """Default behavior is to use Select since it's the widest supported and has
    all of the methods we need for child classes as well. One should only need
    to override the update_handler and start methods for additional types.

    """
    POLL_TIMEOUT_MULT = 1

    def poll(self):
        if False:
            print('Hello World!')
        'Wait for events of interest on registered file descriptors until an\n        event of interest occurs or next timer deadline or _MAX_POLL_TIMEOUT,\n        whichever is sooner, and dispatch the corresponding event handlers.\n\n        '
        while True:
            try:
                if self._fd_events[PollEvents.READ] or self._fd_events[PollEvents.WRITE] or self._fd_events[PollEvents.ERROR]:
                    (read, write, error) = select.select(self._fd_events[PollEvents.READ], self._fd_events[PollEvents.WRITE], self._fd_events[PollEvents.ERROR], self._get_max_wait())
                else:
                    time.sleep(self._get_max_wait())
                    (read, write, error) = ([], [], [])
                break
            except _SELECT_ERRORS as error:
                if _is_resumable(error):
                    continue
                else:
                    raise
        fd_event_map = collections.defaultdict(int)
        for (fd_set, evt) in zip((read, write, error), (PollEvents.READ, PollEvents.WRITE, PollEvents.ERROR)):
            for fileno in fd_set:
                fd_event_map[fileno] |= evt
        self._dispatch_fd_events(fd_event_map)

    def _init_poller(self):
        if False:
            return 10
        'Notify the implementation to allocate the poller resource'

    def _uninit_poller(self):
        if False:
            return 10
        'Notify the implementation to release the poller resource'

    def _register_fd(self, fileno, events):
        if False:
            i = 10
            return i + 15
        'The base class invokes this method to notify the implementation to\n        register the file descriptor with the polling object. The request must\n        be ignored if the poller is not activated.\n\n        :param int fileno: The file descriptor\n        :param int events: The event mask using READ, WRITE, ERROR\n        '

    def _modify_fd_events(self, fileno, events, events_to_clear, events_to_set):
        if False:
            i = 10
            return i + 15
        'The base class invoikes this method to notify the implementation to\n        modify an already registered file descriptor. The request must be\n        ignored if the poller is not activated.\n\n        :param int fileno: The file descriptor\n        :param int events: absolute events (READ, WRITE, ERROR)\n        :param int events_to_clear: The events to clear (READ, WRITE, ERROR)\n        :param int events_to_set: The events to set (READ, WRITE, ERROR)\n        '

    def _unregister_fd(self, fileno, events_to_clear):
        if False:
            while True:
                i = 10
        'The base class invokes this method to notify the implementation to\n        unregister the file descriptor being tracked by the polling object. The\n        request must be ignored if the poller is not activated.\n\n        :param int fileno: The file descriptor\n        :param int events_to_clear: The events to clear (READ, WRITE, ERROR)\n        '

class KQueuePoller(_PollerBase):
    """KQueuePoller works on BSD based systems and is faster than select"""

    def __init__(self, get_wait_seconds, process_timeouts):
        if False:
            return 10
        'Create an instance of the KQueuePoller\n        '
        self._kqueue = None
        super().__init__(get_wait_seconds, process_timeouts)

    @staticmethod
    def _map_event(kevent):
        if False:
            return 10
        'return the event type associated with a kevent object\n\n        :param kevent kevent: a kevent object as returned by kqueue.control()\n\n        '
        mask = 0
        if kevent.filter == select.KQ_FILTER_READ:
            mask = PollEvents.READ
        elif kevent.filter == select.KQ_FILTER_WRITE:
            mask = PollEvents.WRITE
            if kevent.flags & select.KQ_EV_EOF:
                mask |= PollEvents.ERROR
        elif kevent.flags & select.KQ_EV_ERROR:
            mask = PollEvents.ERROR
        else:
            LOGGER.critical('Unexpected kevent: %s', kevent)
        return mask

    def poll(self):
        if False:
            for i in range(10):
                print('nop')
        'Wait for events of interest on registered file descriptors until an\n        event of interest occurs or next timer deadline or _MAX_POLL_TIMEOUT,\n        whichever is sooner, and dispatch the corresponding event handlers.\n\n        '
        while True:
            try:
                kevents = self._kqueue.control(None, 1000, self._get_max_wait())
                break
            except _SELECT_ERRORS as error:
                if _is_resumable(error):
                    continue
                else:
                    raise
        fd_event_map = collections.defaultdict(int)
        for event in kevents:
            fd_event_map[event.ident] |= self._map_event(event)
        self._dispatch_fd_events(fd_event_map)

    def _init_poller(self):
        if False:
            while True:
                i = 10
        'Notify the implementation to allocate the poller resource'
        assert self._kqueue is None
        self._kqueue = select.kqueue()

    def _uninit_poller(self):
        if False:
            while True:
                i = 10
        'Notify the implementation to release the poller resource'
        if self._kqueue is not None:
            self._kqueue.close()
            self._kqueue = None

    def _register_fd(self, fileno, events):
        if False:
            while True:
                i = 10
        'The base class invokes this method to notify the implementation to\n        register the file descriptor with the polling object. The request must\n        be ignored if the poller is not activated.\n\n        :param int fileno: The file descriptor\n        :param int events: The event mask using READ, WRITE, ERROR\n        '
        self._modify_fd_events(fileno, events=events, events_to_clear=0, events_to_set=events)

    def _modify_fd_events(self, fileno, events, events_to_clear, events_to_set):
        if False:
            for i in range(10):
                print('nop')
        'The base class invoikes this method to notify the implementation to\n        modify an already registered file descriptor. The request must be\n        ignored if the poller is not activated.\n\n        :param int fileno: The file descriptor\n        :param int events: absolute events (READ, WRITE, ERROR)\n        :param int events_to_clear: The events to clear (READ, WRITE, ERROR)\n        :param int events_to_set: The events to set (READ, WRITE, ERROR)\n        '
        if self._kqueue is None:
            return
        kevents = list()
        if events_to_clear & PollEvents.READ:
            kevents.append(select.kevent(fileno, filter=select.KQ_FILTER_READ, flags=select.KQ_EV_DELETE))
        if events_to_set & PollEvents.READ:
            kevents.append(select.kevent(fileno, filter=select.KQ_FILTER_READ, flags=select.KQ_EV_ADD))
        if events_to_clear & PollEvents.WRITE:
            kevents.append(select.kevent(fileno, filter=select.KQ_FILTER_WRITE, flags=select.KQ_EV_DELETE))
        if events_to_set & PollEvents.WRITE:
            kevents.append(select.kevent(fileno, filter=select.KQ_FILTER_WRITE, flags=select.KQ_EV_ADD))
        self._kqueue.control(kevents, 0)

    def _unregister_fd(self, fileno, events_to_clear):
        if False:
            while True:
                i = 10
        'The base class invokes this method to notify the implementation to\n        unregister the file descriptor being tracked by the polling object. The\n        request must be ignored if the poller is not activated.\n\n        :param int fileno: The file descriptor\n        :param int events_to_clear: The events to clear (READ, WRITE, ERROR)\n        '
        self._modify_fd_events(fileno, events=0, events_to_clear=events_to_clear, events_to_set=0)

class PollPoller(_PollerBase):
    """Poll works on Linux and can have better performance than EPoll in
    certain scenarios.  Both are faster than select.

    """
    POLL_TIMEOUT_MULT = 1000

    def __init__(self, get_wait_seconds, process_timeouts):
        if False:
            i = 10
            return i + 15
        'Create an instance of the KQueuePoller\n\n        '
        self._poll = None
        super().__init__(get_wait_seconds, process_timeouts)

    @staticmethod
    def _create_poller():
        if False:
            i = 10
            return i + 15
        '\n        :rtype: `select.poll`\n        '
        return select.poll()

    def poll(self):
        if False:
            while True:
                i = 10
        'Wait for events of interest on registered file descriptors until an\n        event of interest occurs or next timer deadline or _MAX_POLL_TIMEOUT,\n        whichever is sooner, and dispatch the corresponding event handlers.\n\n        '
        while True:
            try:
                events = self._poll.poll(self._get_max_wait())
                break
            except _SELECT_ERRORS as error:
                if _is_resumable(error):
                    continue
                else:
                    raise
        fd_event_map = collections.defaultdict(int)
        for (fileno, event) in events:
            if event & select.POLLHUP and pika.compat.ON_OSX:
                event |= select.POLLERR
            fd_event_map[fileno] |= event
        self._dispatch_fd_events(fd_event_map)

    def _init_poller(self):
        if False:
            return 10
        'Notify the implementation to allocate the poller resource'
        assert self._poll is None
        self._poll = self._create_poller()

    def _uninit_poller(self):
        if False:
            return 10
        'Notify the implementation to release the poller resource'
        if self._poll is not None:
            if hasattr(self._poll, 'close'):
                self._poll.close()
            self._poll = None

    def _register_fd(self, fileno, events):
        if False:
            while True:
                i = 10
        'The base class invokes this method to notify the implementation to\n        register the file descriptor with the polling object. The request must\n        be ignored if the poller is not activated.\n\n        :param int fileno: The file descriptor\n        :param int events: The event mask using READ, WRITE, ERROR\n        '
        if self._poll is not None:
            self._poll.register(fileno, events)

    def _modify_fd_events(self, fileno, events, events_to_clear, events_to_set):
        if False:
            print('Hello World!')
        'The base class invoikes this method to notify the implementation to\n        modify an already registered file descriptor. The request must be\n        ignored if the poller is not activated.\n\n        :param int fileno: The file descriptor\n        :param int events: absolute events (READ, WRITE, ERROR)\n        :param int events_to_clear: The events to clear (READ, WRITE, ERROR)\n        :param int events_to_set: The events to set (READ, WRITE, ERROR)\n        '
        if self._poll is not None:
            self._poll.modify(fileno, events)

    def _unregister_fd(self, fileno, events_to_clear):
        if False:
            print('Hello World!')
        'The base class invokes this method to notify the implementation to\n        unregister the file descriptor being tracked by the polling object. The\n        request must be ignored if the poller is not activated.\n\n        :param int fileno: The file descriptor\n        :param int events_to_clear: The events to clear (READ, WRITE, ERROR)\n        '
        if self._poll is not None:
            self._poll.unregister(fileno)

class EPollPoller(PollPoller):
    """EPoll works on Linux and can have better performance than Poll in
    certain scenarios. Both are faster than select.

    """
    POLL_TIMEOUT_MULT = 1

    @staticmethod
    def _create_poller():
        if False:
            i = 10
            return i + 15
        '\n        :rtype: `select.poll`\n        '
        return select.epoll()