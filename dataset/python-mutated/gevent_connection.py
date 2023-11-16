"""Use pika with the Gevent IOLoop."""
import functools
import logging
import os
import threading
import weakref
try:
    import queue
except ImportError:
    import Queue as queue
import gevent
import gevent.hub
import gevent.socket
import pika.compat
from pika.adapters.base_connection import BaseConnection
from pika.adapters.utils.io_services_utils import check_callback_arg
from pika.adapters.utils.nbio_interface import AbstractIOReference, AbstractIOServices
from pika.adapters.utils.selector_ioloop_adapter import AbstractSelectorIOLoop, SelectorIOServicesAdapter
LOGGER = logging.getLogger(__name__)

class GeventConnection(BaseConnection):
    """Implementation of pika's ``BaseConnection``.

    An async selector-based connection which integrates with Gevent.
    """

    def __init__(self, parameters=None, on_open_callback=None, on_open_error_callback=None, on_close_callback=None, custom_ioloop=None, internal_connection_workflow=True):
        if False:
            print('Hello World!')
        "Create a new GeventConnection instance and connect to RabbitMQ on\n        Gevent's event-loop.\n\n        :param pika.connection.Parameters|None parameters: The connection\n            parameters\n        :param callable|None on_open_callback: The method to call when the\n            connection is open\n        :param callable|None on_open_error_callback: Called if the connection\n            can't be established or connection establishment is interrupted by\n            `Connection.close()`:\n            on_open_error_callback(Connection, exception)\n        :param callable|None on_close_callback: Called when a previously fully\n            open connection is closed:\n            `on_close_callback(Connection, exception)`, where `exception` is\n            either an instance of `exceptions.ConnectionClosed` if closed by\n            user or broker or exception of another type that describes the\n            cause of connection failure\n        :param gevent._interfaces.ILoop|nbio_interface.AbstractIOServices|None\n            custom_ioloop: Use a custom Gevent ILoop.\n        :param bool internal_connection_workflow: True for autonomous connection\n            establishment which is default; False for externally-managed\n            connection workflow via the `create_connection()` factory\n        "
        if pika.compat.ON_WINDOWS:
            raise RuntimeError('GeventConnection is not supported on Windows.')
        custom_ioloop = custom_ioloop or _GeventSelectorIOLoop(gevent.get_hub())
        if isinstance(custom_ioloop, AbstractIOServices):
            nbio = custom_ioloop
        else:
            nbio = _GeventSelectorIOServicesAdapter(custom_ioloop)
        super().__init__(parameters, on_open_callback, on_open_error_callback, on_close_callback, nbio, internal_connection_workflow=internal_connection_workflow)

    @classmethod
    def create_connection(cls, connection_configs, on_done, custom_ioloop=None, workflow=None):
        if False:
            for i in range(10):
                print('nop')
        'Implement\n        :py:classmethod::`pika.adapters.BaseConnection.create_connection()`.\n        '
        custom_ioloop = custom_ioloop or _GeventSelectorIOLoop(gevent.get_hub())
        nbio = _GeventSelectorIOServicesAdapter(custom_ioloop)

        def connection_factory(params):
            if False:
                i = 10
                return i + 15
            'Connection factory.'
            if params is None:
                raise ValueError('Expected pika.connection.Parameters instance, but got None in params arg.')
            return cls(parameters=params, custom_ioloop=nbio, internal_connection_workflow=False)
        return cls._start_connection_workflow(connection_configs=connection_configs, connection_factory=connection_factory, nbio=nbio, workflow=workflow, on_done=on_done)

class _TSafeCallbackQueue:
    """Dispatch callbacks from any thread to be executed in the main thread
    efficiently with IO events.
    """

    def __init__(self):
        if False:
            print('Hello World!')
        '\n        :param _GeventSelectorIOLoop loop: IO loop to add callbacks to.\n        '
        self._queue = queue.Queue()
        (self._read_fd, self._write_fd) = os.pipe()
        self._write_lock = threading.RLock()

    @property
    def fd(self):
        if False:
            i = 10
            return i + 15
        'The file-descriptor to register for READ events in the IO loop.'
        return self._read_fd

    def add_callback_threadsafe(self, callback):
        if False:
            return 10
        'Add an item to the queue from any thread. The configured handler\n        will be invoked with the item in the main thread.\n\n        :param item: Object to add to the queue.\n        '
        self._queue.put(callback)
        with self._write_lock:
            os.write(self._write_fd, b'\xff')

    def run_next_callback(self):
        if False:
            print('Hello World!')
        'Invoke the next callback from the queue.\n\n        MUST run in the main thread. If no callback was added to the queue,\n        this will block the IO loop.\n\n        Performs a blocking READ on the pipe so must only be called when the\n        pipe is ready for reading.\n        '
        try:
            callback = self._queue.get_nowait()
        except queue.Empty:
            LOGGER.warning('Callback queue was empty.')
        else:
            os.read(self._read_fd, 1)
            callback()

class _GeventSelectorIOLoop(AbstractSelectorIOLoop):
    """Implementation of `AbstractSelectorIOLoop` using the Gevent event loop.

    Required by implementations of `SelectorIOServicesAdapter`.
    """
    READ = 1
    WRITE = 2
    ERROR = 0

    def __init__(self, gevent_hub=None):
        if False:
            print('Hello World!')
        '\n        :param gevent._interfaces.ILoop gevent_loop:\n        '
        self._hub = gevent_hub or gevent.get_hub()
        self._io_watchers_by_fd = {}
        self._waiter = gevent.hub.Waiter()
        self._callback_queue = _TSafeCallbackQueue()

        def run_callback_in_main_thread(fd, events):
            if False:
                return 10
            'Swallow the fd and events arguments.'
            del fd
            del events
            self._callback_queue.run_next_callback()
        self.add_handler(self._callback_queue.fd, run_callback_in_main_thread, self.READ)

    def close(self):
        if False:
            print('Hello World!')
        "Release the loop's resources."
        self._hub.loop.destroy()
        self._hub = None

    def start(self):
        if False:
            i = 10
            return i + 15
        'Run the I/O loop. It will loop until requested to exit. See `stop()`.\n        '
        LOGGER.debug("Passing control to Gevent's IOLoop")
        self._waiter.get()
        LOGGER.debug("Control was passed back from Gevent's IOLoop")
        self._waiter.clear()

    def stop(self):
        if False:
            print('Hello World!')
        "Request exit from the ioloop. The loop is NOT guaranteed to\n        stop before this method returns.\n\n        To invoke `stop()` safely from a thread other than this IOLoop's thread,\n        call it via `add_callback_threadsafe`; e.g.,\n\n            `ioloop.add_callback(ioloop.stop)`\n        "
        self._waiter.switch(None)

    def add_callback(self, callback):
        if False:
            i = 10
            return i + 15
        "Requests a call to the given function as soon as possible in the\n        context of this IOLoop's thread.\n\n        NOTE: This is the only thread-safe method in IOLoop. All other\n        manipulations of IOLoop must be performed from the IOLoop's thread.\n\n        For example, a thread may request a call to the `stop` method of an\n        ioloop that is running in a different thread via\n        `ioloop.add_callback_threadsafe(ioloop.stop)`\n\n        :param callable callback: The callback method\n        "
        if gevent.get_hub() == self._hub:
            LOGGER.debug('Adding callback from main thread')
            self._hub.loop.run_callback(callback)
        else:
            LOGGER.debug('Adding callback from another thread')
            callback = functools.partial(self._hub.loop.run_callback, callback)
            self._callback_queue.add_callback_threadsafe(callback)

    def call_later(self, delay, callback):
        if False:
            return 10
        'Add the callback to the IOLoop timer to be called after delay seconds\n        from the time of call on best-effort basis. Returns a handle to the\n        timeout.\n\n        :param float delay: The number of seconds to wait to call callback\n        :param callable callback: The callback method\n        :returns: handle to the created timeout that may be passed to\n            `remove_timeout()`\n        :rtype: object\n        '
        timer = self._hub.loop.timer(delay)
        timer.start(callback)
        return timer

    def remove_timeout(self, timeout_handle):
        if False:
            while True:
                i = 10
        'Remove a timeout\n\n        :param timeout_handle: Handle of timeout to remove\n        '
        timeout_handle.close()

    def add_handler(self, fd, handler, events):
        if False:
            return 10
        'Start watching the given file descriptor for events\n\n        :param int fd: The file descriptor\n        :param callable handler: When requested event(s) occur,\n            `handler(fd, events)` will be called.\n        :param int events: The event mask (READ|WRITE)\n        '
        io_watcher = self._hub.loop.io(fd, events)
        self._io_watchers_by_fd[fd] = io_watcher
        io_watcher.start(handler, fd, events)

    def update_handler(self, fd, events):
        if False:
            i = 10
            return i + 15
        'Change the events being watched for.\n\n        :param int fd: The file descriptor\n        :param int events: The new event mask (READ|WRITE)\n        '
        io_watcher = self._io_watchers_by_fd[fd]
        callback = io_watcher.callback
        io_watcher.close()
        del self._io_watchers_by_fd[fd]
        self.add_handler(fd, callback, events)

    def remove_handler(self, fd):
        if False:
            print('Hello World!')
        'Stop watching the given file descriptor for events\n\n        :param int fd: The file descriptor\n        '
        io_watcher = self._io_watchers_by_fd[fd]
        io_watcher.close()
        del self._io_watchers_by_fd[fd]

class _GeventSelectorIOServicesAdapter(SelectorIOServicesAdapter):
    """SelectorIOServicesAdapter implementation using Gevent's DNS resolver."""

    def getaddrinfo(self, host, port, on_done, family=0, socktype=0, proto=0, flags=0):
        if False:
            i = 10
            return i + 15
        'Implement :py:meth:`.nbio_interface.AbstractIOServices.getaddrinfo()`.\n        '
        resolver = _GeventAddressResolver(native_loop=self._loop, host=host, port=port, family=family, socktype=socktype, proto=proto, flags=flags, on_done=on_done)
        resolver.start()
        return _GeventIOLoopIOHandle(resolver)

class _GeventIOLoopIOHandle(AbstractIOReference):
    """Implement `AbstractIOReference`.

    Only used to wrap the _GeventAddressResolver.
    """

    def __init__(self, subject):
        if False:
            return 10
        '\n        :param subject: subject of the reference containing a `cancel()` method\n        '
        self._cancel = subject.cancel

    def cancel(self):
        if False:
            i = 10
            return i + 15
        'Cancel pending operation\n\n        :returns: False if was already done or cancelled; True otherwise\n        :rtype: bool\n        '
        return self._cancel()

class _GeventAddressResolver:
    """Performs getaddrinfo asynchronously Gevent's configured resolver in a
    separate greenlet and invoking the provided callback with the result.

    See: http://www.gevent.org/dns.html
    """
    __slots__ = ('_loop', '_on_done', '_greenlet', '_ga_host', '_ga_port', '_ga_family', '_ga_socktype', '_ga_proto', '_ga_flags')

    def __init__(self, native_loop, host, port, family, socktype, proto, flags, on_done):
        if False:
            while True:
                i = 10
        'Initialize the `_GeventAddressResolver`.\n\n        :param AbstractSelectorIOLoop native_loop:\n        :param host: `see socket.getaddrinfo()`\n        :param port: `see socket.getaddrinfo()`\n        :param family: `see socket.getaddrinfo()`\n        :param socktype: `see socket.getaddrinfo()`\n        :param proto: `see socket.getaddrinfo()`\n        :param flags: `see socket.getaddrinfo()`\n        :param on_done: on_done(records|BaseException) callback for reporting\n            result from the given I/O loop. The single arg will be either an\n            exception object (check for `BaseException`) in case of failure or\n            the result returned by `socket.getaddrinfo()`.\n        '
        check_callback_arg(on_done, 'on_done')
        self._loop = native_loop
        self._on_done = on_done
        self._greenlet = None
        self._ga_host = host
        self._ga_port = port
        self._ga_family = family
        self._ga_socktype = socktype
        self._ga_proto = proto
        self._ga_flags = flags

    def start(self):
        if False:
            i = 10
            return i + 15
        'Start an asynchronous getaddrinfo invocation.'
        if self._greenlet is None:
            self._greenlet = gevent.spawn_raw(self._resolve)
        else:
            LOGGER.warning('_GeventAddressResolver already started')

    def cancel(self):
        if False:
            i = 10
            return i + 15
        'Cancel the pending resolver.'
        changed = False
        if self._greenlet is not None:
            changed = True
            self._stop_greenlet()
        self._cleanup()
        return changed

    def _cleanup(self):
        if False:
            print('Hello World!')
        'Stop the resolver and release any resources.'
        self._stop_greenlet()
        self._loop = None
        self._on_done = None

    def _stop_greenlet(self):
        if False:
            while True:
                i = 10
        'Stop the greenlet performing getaddrinfo if running.\n\n        Otherwise, this is a no-op.\n        '
        if self._greenlet is not None:
            gevent.kill(self._greenlet)
            self._greenlet = None

    def _resolve(self):
        if False:
            print('Hello World!')
        "Call `getaddrinfo()` and return result via user's callback\n        function on the configured IO loop.\n        "
        try:
            result = gevent.socket.getaddrinfo(self._ga_host, self._ga_port, self._ga_family, self._ga_socktype, self._ga_proto, self._ga_flags)
        except Exception as exc:
            LOGGER.error('Address resolution failed: %r', exc)
            result = exc
        callback = functools.partial(self._dispatch_callback, result)
        self._loop.add_callback(callback)

    def _dispatch_callback(self, result):
        if False:
            for i in range(10):
                print('nop')
        'Invoke the configured completion callback and any subsequent cleanup.\n\n        :param result: result from getaddrinfo, or the exception if raised.\n        '
        try:
            LOGGER.debug('Invoking async getaddrinfo() completion callback; host=%r', self._ga_host)
            self._on_done(result)
        finally:
            self._cleanup()