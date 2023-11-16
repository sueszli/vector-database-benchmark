"""
Implementation of `nbio_interface.AbstractIOServices` on top of a
selector-based I/O loop, such as tornado's and our home-grown
select_connection's I/O loops.

"""
import abc
import logging
import socket
import threading
from pika.adapters.utils import nbio_interface, io_services_utils
from pika.adapters.utils.io_services_utils import check_callback_arg, check_fd_arg
LOGGER = logging.getLogger(__name__)

class AbstractSelectorIOLoop:
    """Selector-based I/O loop interface expected by
    `selector_ioloop_adapter.SelectorIOServicesAdapter`

    NOTE: this interface follows the corresponding methods and attributes
     of `tornado.ioloop.IOLoop` in order to avoid additional adapter layering
     when wrapping tornado's IOLoop.
    """

    @property
    @abc.abstractmethod
    def READ(self):
        if False:
            return 10
        "The value of the I/O loop's READ flag; READ/WRITE/ERROR may be used\n        with bitwise operators as expected.\n\n        Implementation note: the implementations can simply replace these\n        READ/WRITE/ERROR properties with class-level attributes\n\n        "

    @property
    @abc.abstractmethod
    def WRITE(self):
        if False:
            while True:
                i = 10
        "The value of the I/O loop's WRITE flag; READ/WRITE/ERROR may be used\n        with bitwise operators as expected\n\n        "

    @property
    @abc.abstractmethod
    def ERROR(self):
        if False:
            while True:
                i = 10
        "The value of the I/O loop's ERROR flag; READ/WRITE/ERROR may be used\n        with bitwise operators as expected\n\n        "

    @abc.abstractmethod
    def close(self):
        if False:
            i = 10
            return i + 15
        "Release IOLoop's resources.\n\n        the `close()` method is intended to be called by the application or test\n        code only after `start()` returns. After calling `close()`, no other\n        interaction with the closed instance of `IOLoop` should be performed.\n\n        "

    @abc.abstractmethod
    def start(self):
        if False:
            print('Hello World!')
        'Run the I/O loop. It will loop until requested to exit. See `stop()`.\n\n        '

    @abc.abstractmethod
    def stop(self):
        if False:
            print('Hello World!')
        "Request exit from the ioloop. The loop is NOT guaranteed to\n        stop before this method returns.\n\n        To invoke `stop()` safely from a thread other than this IOLoop's thread,\n        call it via `add_callback_threadsafe`; e.g.,\n\n            `ioloop.add_callback(ioloop.stop)`\n\n        "

    @abc.abstractmethod
    def call_later(self, delay, callback):
        if False:
            for i in range(10):
                print('nop')
        'Add the callback to the IOLoop timer to be called after delay seconds\n        from the time of call on best-effort basis. Returns a handle to the\n        timeout.\n\n        :param float delay: The number of seconds to wait to call callback\n        :param callable callback: The callback method\n        :returns: handle to the created timeout that may be passed to\n            `remove_timeout()`\n        :rtype: object\n\n        '

    @abc.abstractmethod
    def remove_timeout(self, timeout_handle):
        if False:
            while True:
                i = 10
        'Remove a timeout\n\n        :param timeout_handle: Handle of timeout to remove\n\n        '

    @abc.abstractmethod
    def add_callback(self, callback):
        if False:
            return 10
        "Requests a call to the given function as soon as possible in the\n        context of this IOLoop's thread.\n\n        NOTE: This is the only thread-safe method in IOLoop. All other\n        manipulations of IOLoop must be performed from the IOLoop's thread.\n\n        For example, a thread may request a call to the `stop` method of an\n        ioloop that is running in a different thread via\n        `ioloop.add_callback_threadsafe(ioloop.stop)`\n\n        :param callable callback: The callback method\n\n        "

    @abc.abstractmethod
    def add_handler(self, fd, handler, events):
        if False:
            i = 10
            return i + 15
        'Start watching the given file descriptor for events\n\n        :param int fd: The file descriptor\n        :param callable handler: When requested event(s) occur,\n            `handler(fd, events)` will be called.\n        :param int events: The event mask using READ, WRITE, ERROR.\n\n        '

    @abc.abstractmethod
    def update_handler(self, fd, events):
        if False:
            i = 10
            return i + 15
        'Changes the events we watch for\n\n        :param int fd: The file descriptor\n        :param int events: The event mask using READ, WRITE, ERROR\n\n        '

    @abc.abstractmethod
    def remove_handler(self, fd):
        if False:
            return 10
        'Stop watching the given file descriptor for events\n\n        :param int fd: The file descriptor\n\n        '

class SelectorIOServicesAdapter(io_services_utils.SocketConnectionMixin, io_services_utils.StreamingConnectionMixin, nbio_interface.AbstractIOServices, nbio_interface.AbstractFileDescriptorServices):
    """Implements the
    :py:class:`.nbio_interface.AbstractIOServices` interface
    on top of selector-style native loop having the
    :py:class:`AbstractSelectorIOLoop` interface, such as
    :py:class:`pika.selection_connection.IOLoop` and :py:class:`tornado.IOLoop`.

    NOTE:
    :py:class:`.nbio_interface.AbstractFileDescriptorServices`
    interface is only required by the mixins.

    """

    def __init__(self, native_loop):
        if False:
            i = 10
            return i + 15
        '\n        :param AbstractSelectorIOLoop native_loop: An instance compatible with\n            the `AbstractSelectorIOLoop` interface, but not necessarily derived\n            from it.\n        '
        self._loop = native_loop
        self._watchers = dict()
        self._readable_mask = self._loop.READ
        self._writable_mask = self._loop.WRITE | self._loop.ERROR

    def get_native_ioloop(self):
        if False:
            for i in range(10):
                print('nop')
        'Implement\n        :py:meth:`.nbio_interface.AbstractIOServices.get_native_ioloop()`.\n\n        '
        return self._loop

    def close(self):
        if False:
            print('Hello World!')
        'Implement :py:meth:`.nbio_interface.AbstractIOServices.close()`.\n\n        '
        self._loop.close()

    def run(self):
        if False:
            while True:
                i = 10
        'Implement :py:meth:`.nbio_interface.AbstractIOServices.run()`.\n\n        '
        self._loop.start()

    def stop(self):
        if False:
            while True:
                i = 10
        'Implement :py:meth:`.nbio_interface.AbstractIOServices.stop()`.\n\n        '
        self._loop.stop()

    def add_callback_threadsafe(self, callback):
        if False:
            i = 10
            return i + 15
        'Implement\n        :py:meth:`.nbio_interface.AbstractIOServices.add_callback_threadsafe()`.\n\n        '
        self._loop.add_callback(callback)

    def call_later(self, delay, callback):
        if False:
            for i in range(10):
                print('nop')
        'Implement :py:meth:`.nbio_interface.AbstractIOServices.call_later()`.\n\n        '
        return _TimerHandle(self._loop.call_later(delay, callback), self._loop)

    def getaddrinfo(self, host, port, on_done, family=0, socktype=0, proto=0, flags=0):
        if False:
            i = 10
            return i + 15
        'Implement :py:meth:`.nbio_interface.AbstractIOServices.getaddrinfo()`.\n\n        '
        return _SelectorIOLoopIOHandle(_AddressResolver(native_loop=self._loop, host=host, port=port, family=family, socktype=socktype, proto=proto, flags=flags, on_done=on_done).start())

    def set_reader(self, fd, on_readable):
        if False:
            while True:
                i = 10
        'Implement\n        :py:meth:`.nbio_interface.AbstractFileDescriptorServices.set_reader()`.\n\n        '
        LOGGER.debug('SelectorIOServicesAdapter.set_reader(%s, %r)', fd, on_readable)
        check_fd_arg(fd)
        check_callback_arg(on_readable, 'on_readable')
        try:
            callbacks = self._watchers[fd]
        except KeyError:
            self._loop.add_handler(fd, self._on_reader_writer_fd_events, self._readable_mask)
            self._watchers[fd] = _FileDescriptorCallbacks(reader=on_readable)
            LOGGER.debug('set_reader(%s, _) added handler Rd', fd)
        else:
            if callbacks.reader is None:
                assert callbacks.writer is not None
                self._loop.update_handler(fd, self._readable_mask | self._writable_mask)
                LOGGER.debug('set_reader(%s, _) updated handler RdWr', fd)
            else:
                LOGGER.debug('set_reader(%s, _) replacing reader', fd)
            callbacks.reader = on_readable

    def remove_reader(self, fd):
        if False:
            for i in range(10):
                print('nop')
        'Implement\n        :py:meth:`.nbio_interface.AbstractFileDescriptorServices.remove_reader()`.\n\n        '
        LOGGER.debug('SelectorIOServicesAdapter.remove_reader(%s)', fd)
        check_fd_arg(fd)
        try:
            callbacks = self._watchers[fd]
        except KeyError:
            LOGGER.debug('remove_reader(%s) neither was set', fd)
            return False
        if callbacks.reader is None:
            assert callbacks.writer is not None
            LOGGER.debug("remove_reader(%s) reader wasn't set Wr", fd)
            return False
        callbacks.reader = None
        if callbacks.writer is None:
            del self._watchers[fd]
            self._loop.remove_handler(fd)
            LOGGER.debug('remove_reader(%s) removed handler', fd)
        else:
            self._loop.update_handler(fd, self._writable_mask)
            LOGGER.debug('remove_reader(%s) updated handler Wr', fd)
        return True

    def set_writer(self, fd, on_writable):
        if False:
            for i in range(10):
                print('nop')
        'Implement\n        :py:meth:`.nbio_interface.AbstractFileDescriptorServices.set_writer()`.\n\n        '
        LOGGER.debug('SelectorIOServicesAdapter.set_writer(%s, %r)', fd, on_writable)
        check_fd_arg(fd)
        check_callback_arg(on_writable, 'on_writable')
        try:
            callbacks = self._watchers[fd]
        except KeyError:
            self._loop.add_handler(fd, self._on_reader_writer_fd_events, self._writable_mask)
            self._watchers[fd] = _FileDescriptorCallbacks(writer=on_writable)
            LOGGER.debug('set_writer(%s, _) added handler Wr', fd)
        else:
            if callbacks.writer is None:
                assert callbacks.reader is not None
                callbacks.writer = on_writable
                self._loop.update_handler(fd, self._readable_mask | self._writable_mask)
                LOGGER.debug('set_writer(%s, _) updated handler RdWr', fd)
            else:
                LOGGER.debug('set_writer(%s, _) replacing writer', fd)
                callbacks.writer = on_writable

    def remove_writer(self, fd):
        if False:
            for i in range(10):
                print('nop')
        'Implement\n        :py:meth:`.nbio_interface.AbstractFileDescriptorServices.remove_writer()`.\n\n        '
        LOGGER.debug('SelectorIOServicesAdapter.remove_writer(%s)', fd)
        check_fd_arg(fd)
        try:
            callbacks = self._watchers[fd]
        except KeyError:
            LOGGER.debug('remove_writer(%s) neither was set.', fd)
            return False
        if callbacks.writer is None:
            assert callbacks.reader is not None
            LOGGER.debug("remove_writer(%s) writer wasn't set Rd", fd)
            return False
        callbacks.writer = None
        if callbacks.reader is None:
            del self._watchers[fd]
            self._loop.remove_handler(fd)
            LOGGER.debug('remove_writer(%s) removed handler', fd)
        else:
            self._loop.update_handler(fd, self._readable_mask)
            LOGGER.debug('remove_writer(%s) updated handler Rd', fd)
        return True

    def _on_reader_writer_fd_events(self, fd, events):
        if False:
            print('Hello World!')
        "Handle indicated file descriptor events requested via `set_reader()`\n        and `set_writer()`.\n\n        :param fd: file descriptor\n        :param events: event mask using native loop's READ/WRITE/ERROR. NOTE:\n            depending on the underlying poller mechanism, ERROR may be indicated\n            upon certain file description state even though we don't request it.\n            We ignore ERROR here since `set_reader()`/`set_writer()` don't\n            request for it.\n        "
        callbacks = self._watchers[fd]
        if events & self._readable_mask and callbacks.reader is None:
            LOGGER.warning('READ indicated on fd=%s, but reader callback is None; events=%s', fd, bin(events))
        if events & self._writable_mask:
            if callbacks.writer is not None:
                callbacks.writer()
            else:
                LOGGER.warning('WRITE indicated on fd=%s, but writer callback is None; events=%s', fd, bin(events))
        if events & self._readable_mask:
            if callbacks.reader is not None:
                callbacks.reader()
            else:
                pass

class _FileDescriptorCallbacks:
    """Holds reader and writer callbacks for a file descriptor"""
    __slots__ = ('reader', 'writer')

    def __init__(self, reader=None, writer=None):
        if False:
            print('Hello World!')
        self.reader = reader
        self.writer = writer

class _TimerHandle(nbio_interface.AbstractTimerReference):
    """This module's adaptation of `nbio_interface.AbstractTimerReference`.

    """

    def __init__(self, handle, loop):
        if False:
            return 10
        '\n\n        :param opaque handle: timer handle from the underlying loop\n            implementation that may be passed to its `remove_timeout()` method\n        :param AbstractSelectorIOLoop loop: the I/O loop instance that created\n            the timeout.\n        '
        self._handle = handle
        self._loop = loop

    def cancel(self):
        if False:
            i = 10
            return i + 15
        if self._loop is not None:
            self._loop.remove_timeout(self._handle)
            self._handle = None
            self._loop = None

class _SelectorIOLoopIOHandle(nbio_interface.AbstractIOReference):
    """This module's adaptation of `nbio_interface.AbstractIOReference`

    """

    def __init__(self, subject):
        if False:
            print('Hello World!')
        '\n        :param subject: subject of the reference containing a `cancel()` method\n\n        '
        self._cancel = subject.cancel

    def cancel(self):
        if False:
            for i in range(10):
                print('nop')
        'Cancel pending operation\n\n        :returns: False if was already done or cancelled; True otherwise\n        :rtype: bool\n\n        '
        return self._cancel()

class _AddressResolver:
    """Performs getaddrinfo asynchronously using a thread, then reports result
    via callback from the given I/O loop.

    NOTE: at this stage, we're using a thread per request, which may prove
    inefficient and even prohibitive if the app performs many of these
    operations concurrently.
    """
    NOT_STARTED = 0
    ACTIVE = 1
    CANCELED = 2
    COMPLETED = 3

    def __init__(self, native_loop, host, port, family, socktype, proto, flags, on_done):
        if False:
            i = 10
            return i + 15
        '\n\n        :param AbstractSelectorIOLoop native_loop:\n        :param host: `see socket.getaddrinfo()`\n        :param port: `see socket.getaddrinfo()`\n        :param family: `see socket.getaddrinfo()`\n        :param socktype: `see socket.getaddrinfo()`\n        :param proto: `see socket.getaddrinfo()`\n        :param flags: `see socket.getaddrinfo()`\n        :param on_done: on_done(records|BaseException) callback for reporting\n            result from the given I/O loop. The single arg will be either an\n            exception object (check for `BaseException`) in case of failure or\n            the result returned by `socket.getaddrinfo()`.\n        '
        check_callback_arg(on_done, 'on_done')
        self._state = self.NOT_STARTED
        self._result = None
        self._loop = native_loop
        self._host = host
        self._port = port
        self._family = family
        self._socktype = socktype
        self._proto = proto
        self._flags = flags
        self._on_done = on_done
        self._mutex = threading.Lock()
        self._threading_timer = None

    def _cleanup(self):
        if False:
            for i in range(10):
                print('nop')
        'Release resources\n\n        '
        self._loop = None
        self._threading_timer = None
        self._on_done = None

    def start(self):
        if False:
            for i in range(10):
                print('nop')
        'Start asynchronous DNS lookup.\n\n        :rtype: nbio_interface.AbstractIOReference\n\n        '
        assert self._state == self.NOT_STARTED, self._state
        self._state = self.ACTIVE
        self._threading_timer = threading.Timer(0, self._resolve)
        self._threading_timer.start()
        return _SelectorIOLoopIOHandle(self)

    def cancel(self):
        if False:
            print('Hello World!')
        'Cancel the pending resolver\n\n        :returns: False if was already done or cancelled; True otherwise\n        :rtype: bool\n\n        '
        with self._mutex:
            if self._state == self.ACTIVE:
                LOGGER.debug('Canceling resolver for %s:%s', self._host, self._port)
                self._state = self.CANCELED
                self._threading_timer.cancel()
                self._cleanup()
                return True
            else:
                LOGGER.debug('Ignoring _AddressResolver cancel request when not ACTIVE; (%s:%s); state=%s', self._host, self._port, self._state)
                return False

    def _resolve(self):
        if False:
            while True:
                i = 10
        "Call `socket.getaddrinfo()` and return result via user's callback\n        function on the given I/O loop\n\n        "
        try:
            result = socket.getaddrinfo(self._host, self._port, self._family, self._socktype, self._proto, self._flags)
        except Exception as exc:
            LOGGER.error('Address resolution failed: %r', exc)
            result = exc
        self._result = result
        with self._mutex:
            if self._state == self.ACTIVE:
                self._loop.add_callback(self._dispatch_result)
            else:
                LOGGER.debug('Asynchronous getaddrinfo cancellation detected; in thread; host=%r', self._host)

    def _dispatch_result(self):
        if False:
            print('Hello World!')
        "This is called from the user's I/O loop to pass the result to the\n         user via the user's on_done callback\n\n        "
        if self._state == self.ACTIVE:
            self._state = self.COMPLETED
            try:
                LOGGER.debug('Invoking asynchronous getaddrinfo() completion callback; host=%r', self._host)
                self._on_done(self._result)
            finally:
                self._cleanup()
        else:
            LOGGER.debug('Asynchronous getaddrinfo cancellation detected; in I/O loop context; host=%r', self._host)