"""Non-blocking I/O interface for pika connection adapters.

I/O interface expected by `pika.adapters.base_connection.BaseConnection`

NOTE: This API is modeled after asyncio in python3 for a couple of reasons
    1. It's a sensible API
    2. To make it easy to implement at least on top of the built-in asyncio

Furthermore, the API caters to the needs of pika core and lack of generalization
is intentional for the sake of reducing complexity of the implementation and
testing and lessening the maintenance burden.

"""
import abc
import pika.compat

class AbstractIOServices(pika.compat.AbstractBase):
    """Interface to I/O services required by `pika.adapters.BaseConnection` and
    related utilities.

    NOTE: This is not a public API. Pika users should rely on the native I/O
    loop APIs (e.g., asyncio event loop, tornado ioloop, twisted reactor, etc.)
    that corresponds to the chosen Connection adapter.

    """

    @abc.abstractmethod
    def get_native_ioloop(self):
        if False:
            return 10
        "Returns the native I/O loop instance, such as Twisted reactor,\n        asyncio's or tornado's event loop\n\n        "
        raise NotImplementedError

    @abc.abstractmethod
    def close(self):
        if False:
            for i in range(10):
                print('nop')
        "Release IOLoop's resources.\n\n        the `close()` method is intended to be called by Pika's own test\n        code only after `start()` returns. After calling `close()`, no other\n        interaction with the closed instance of `IOLoop` should be performed.\n\n        NOTE: This method is provided for Pika's own test scripts that need to\n        be able to run I/O loops generically to test multiple Connection Adapter\n        implementations. Pika users should use the native I/O loop's API\n        instead.\n\n        "
        raise NotImplementedError

    @abc.abstractmethod
    def run(self):
        if False:
            i = 10
            return i + 15
        "Run the I/O loop. It will loop until requested to exit. See `stop()`.\n\n        NOTE: the outcome or restarting an instance that had been stopped is\n        UNDEFINED!\n\n        NOTE: This method is provided for Pika's own test scripts that need to\n        be able to run I/O loops generically to test multiple Connection Adapter\n        implementations (not all of the supported I/O Loop frameworks have\n        methods named start/stop). Pika users should use the native I/O loop's\n        API instead.\n\n        "
        raise NotImplementedError

    @abc.abstractmethod
    def stop(self):
        if False:
            print('Hello World!')
        "Request exit from the ioloop. The loop is NOT guaranteed to\n        stop before this method returns.\n\n        NOTE: The outcome of calling `stop()` on a non-running instance is\n        UNDEFINED!\n\n        NOTE: This method is provided for Pika's own test scripts that need to\n        be able to run I/O loops generically to test multiple Connection Adapter\n        implementations (not all of the supported I/O Loop frameworks have\n        methods named start/stop). Pika users should use the native I/O loop's\n        API instead.\n\n        To invoke `stop()` safely from a thread other than this IOLoop's thread,\n        call it via `add_callback_threadsafe`; e.g.,\n\n            `ioloop.add_callback_threadsafe(ioloop.stop)`\n\n        "
        raise NotImplementedError

    @abc.abstractmethod
    def add_callback_threadsafe(self, callback):
        if False:
            while True:
                i = 10
        "Requests a call to the given function as soon as possible. It will be\n        called from this IOLoop's thread.\n\n        NOTE: This is the only thread-safe method offered by the IOLoop adapter.\n              All other manipulations of the IOLoop adapter and objects governed\n              by it must be performed from the IOLoop's thread.\n\n        NOTE: if you know that the requester is running on the same thread as\n              the connection it is more efficient to use the\n              `ioloop.call_later()` method with a delay of 0.\n\n        :param callable callback: The callback method; must be callable.\n        "
        raise NotImplementedError

    @abc.abstractmethod
    def call_later(self, delay, callback):
        if False:
            print('Hello World!')
        "Add the callback to the IOLoop timer to be called after delay seconds\n        from the time of call on best-effort basis. Returns a handle to the\n        timeout.\n\n        If two are scheduled for the same time, it's undefined which one will\n        be called first.\n\n        :param float delay: The number of seconds to wait to call callback\n        :param callable callback: The callback method\n        :returns: A handle that can be used to cancel the request.\n        :rtype: AbstractTimerReference\n\n        "
        raise NotImplementedError

    @abc.abstractmethod
    def getaddrinfo(self, host, port, on_done, family=0, socktype=0, proto=0, flags=0):
        if False:
            return 10
        'Perform the equivalent of `socket.getaddrinfo()` asynchronously.\n\n        See `socket.getaddrinfo()` for the standard args.\n\n        :param callable on_done: user callback that takes the return value of\n            `socket.getaddrinfo()` upon successful completion or exception upon\n            failure (check for `BaseException`) as its only arg. It will not be\n            called if the operation was cancelled.\n        :rtype: AbstractIOReference\n        '
        raise NotImplementedError

    @abc.abstractmethod
    def connect_socket(self, sock, resolved_addr, on_done):
        if False:
            return 10
        "Perform the equivalent of `socket.connect()` on a previously-resolved\n        address asynchronously.\n\n        IMPLEMENTATION NOTE: Pika's connection logic resolves the addresses\n            prior to making socket connections, so we don't need to burden the\n            implementations of this method with the extra logic of asynchronous\n            DNS resolution. Implementations can use `socket.inet_pton()` to\n            verify the address.\n\n        :param socket.socket sock: non-blocking socket that needs to be\n            connected via `socket.socket.connect()`\n        :param tuple resolved_addr: resolved destination address/port two-tuple\n            as per `socket.socket.connect()`, except that the first element must\n            be an actual IP address that's consistent with the given socket's\n            address family.\n        :param callable on_done: user callback that takes None upon successful\n            completion or exception (check for `BaseException`) upon error as\n            its only arg. It will not be called if the operation was cancelled.\n\n        :rtype: AbstractIOReference\n        :raises ValueError: if host portion of `resolved_addr` is not an IP\n            address or is inconsistent with the socket's address family as\n            validated via `socket.inet_pton()`\n        "
        raise NotImplementedError

    @abc.abstractmethod
    def create_streaming_connection(self, protocol_factory, sock, on_done, ssl_context=None, server_hostname=None):
        if False:
            print('Hello World!')
        "Perform SSL session establishment, if requested, on the already-\n        connected socket and link the streaming transport/protocol pair.\n\n        NOTE: This method takes ownership of the socket.\n\n        :param callable protocol_factory: called without args, returns an\n            instance with the `AbstractStreamProtocol` interface. The protocol's\n            `connection_made(transport)` method will be called to link it to\n            the transport after remaining connection activity (e.g., SSL session\n            establishment), if any, is completed successfully.\n        :param socket.socket sock: Already-connected, non-blocking\n            `socket.SOCK_STREAM` socket to be used by the transport. We take\n            ownership of this socket.\n        :param callable on_done: User callback\n            `on_done(BaseException | (transport, protocol))` to be notified when\n            the asynchronous operation completes. An exception arg indicates\n            failure (check for `BaseException`); otherwise the two-tuple will\n            contain the linked transport/protocol pair having\n            AbstractStreamTransport and AbstractStreamProtocol interfaces\n            respectively.\n        :param None | ssl.SSLContext ssl_context: if None, this will proceed as\n            a plaintext connection; otherwise, if not None, SSL session\n            establishment will be performed prior to linking the transport and\n            protocol.\n        :param str | None server_hostname: For use during SSL session\n            establishment to match against the target server's certificate. The\n            value `None` disables this check (which is a huge security risk)\n        :rtype: AbstractIOReference\n        "
        raise NotImplementedError

class AbstractFileDescriptorServices(pika.compat.AbstractBase):
    """Interface definition of common non-blocking file descriptor services
    required by some utility implementations.

    NOTE: This is not a public API. Pika users should rely on the native I/O
    loop APIs (e.g., asyncio event loop, tornado ioloop, twisted reactor, etc.)
    that corresponds to the chosen Connection adapter.

    """

    @abc.abstractmethod
    def set_reader(self, fd, on_readable):
        if False:
            i = 10
            return i + 15
        'Call the given callback when the file descriptor is readable.\n        Replace prior reader, if any, for the given file descriptor.\n\n        :param fd: file descriptor\n        :param callable on_readable: a callback taking no args to be notified\n            when fd becomes readable.\n\n        '
        raise NotImplementedError

    @abc.abstractmethod
    def remove_reader(self, fd):
        if False:
            print('Hello World!')
        'Stop watching the given file descriptor for readability\n\n        :param fd: file descriptor\n        :returns: True if reader was removed; False if none was registered.\n        :rtype: bool\n\n        '
        raise NotImplementedError

    @abc.abstractmethod
    def set_writer(self, fd, on_writable):
        if False:
            for i in range(10):
                print('nop')
        'Call the given callback whenever the file descriptor is writable.\n        Replace prior writer callback, if any, for the given file descriptor.\n\n        IMPLEMENTATION NOTE: For portability, implementations of\n            `set_writable()` should also watch for indication of error on the\n            socket and treat it as equivalent to the writable indication (e.g.,\n            also adding the socket to the `exceptfds` arg of `socket.select()`\n            and calling the `on_writable` callback if `select.select()`\n            indicates that the socket is in error state). Specifically, Windows\n            (unlike POSIX) only indicates error on the socket (but not writable)\n            when connection establishment fails.\n\n        :param fd: file descriptor\n        :param callable on_writable: a callback taking no args to be notified\n            when fd becomes writable.\n\n        '
        raise NotImplementedError

    @abc.abstractmethod
    def remove_writer(self, fd):
        if False:
            while True:
                i = 10
        'Stop watching the given file descriptor for writability\n\n        :param fd: file descriptor\n        :returns: True if reader was removed; False if none was registered.\n        :rtype: bool\n\n        '
        raise NotImplementedError

class AbstractTimerReference(pika.compat.AbstractBase):
    """Reference to asynchronous operation"""

    @abc.abstractmethod
    def cancel(self):
        if False:
            for i in range(10):
                print('nop')
        'Cancel callback. If already cancelled, has no affect.\n        '
        raise NotImplementedError

class AbstractIOReference(pika.compat.AbstractBase):
    """Reference to asynchronous I/O operation"""

    @abc.abstractmethod
    def cancel(self):
        if False:
            i = 10
            return i + 15
        'Cancel pending operation\n\n        :returns: False if was already done or cancelled; True otherwise\n        :rtype: bool\n        '
        raise NotImplementedError

class AbstractStreamProtocol(pika.compat.AbstractBase):
    """Stream protocol interface. It's compatible with a subset of
    `asyncio.protocols.Protocol` for compatibility with asyncio-based
    `AbstractIOServices` implementation.

    """

    @abc.abstractmethod
    def connection_made(self, transport):
        if False:
            print('Hello World!')
        'Introduces transport to protocol after transport is connected.\n\n        :param AbstractStreamTransport transport:\n        :raises Exception: Exception-based exception on error\n        '
        raise NotImplementedError

    @abc.abstractmethod
    def connection_lost(self, error):
        if False:
            for i in range(10):
                print('nop')
        "Called upon loss or closing of connection.\n\n        NOTE: `connection_made()` and `connection_lost()` are each called just\n        once and in that order. All other callbacks are called between them.\n\n        :param BaseException | None error: An exception (check for\n            `BaseException`) indicates connection failure. None indicates that\n            connection was closed on this side, such as when it's aborted or\n            when `AbstractStreamProtocol.eof_received()` returns a result that\n            doesn't evaluate to True.\n        :raises Exception: Exception-based exception on error\n        "
        raise NotImplementedError

    @abc.abstractmethod
    def eof_received(self):
        if False:
            return 10
        "Called after the remote peer shuts its write end of the connection.\n\n        :returns: A falsy value (including None) will cause the transport to\n            close itself, resulting in an eventual `connection_lost()` call\n            from the transport. If a truthy value is returned, it will be the\n            protocol's responsibility to close/abort the transport.\n        :rtype: falsy|truthy\n        :raises Exception: Exception-based exception on error\n        "
        raise NotImplementedError

    @abc.abstractmethod
    def data_received(self, data):
        if False:
            i = 10
            return i + 15
        'Called to deliver incoming data to the protocol.\n\n        :param data: Non-empty data bytes.\n        :raises Exception: Exception-based exception on error\n        '
        raise NotImplementedError

class AbstractStreamTransport(pika.compat.AbstractBase):
    """Stream transport interface. It's compatible with a subset of
    `asyncio.transports.Transport` for compatibility with asyncio-based
    `AbstractIOServices` implementation.

    """

    @abc.abstractmethod
    def abort(self):
        if False:
            while True:
                i = 10
        "Close connection abruptly without waiting for pending I/O to\n        complete. Will invoke the corresponding protocol's `connection_lost()`\n        method asynchronously (not in context of the abort() call).\n\n        :raises Exception: Exception-based exception on error\n        "
        raise NotImplementedError

    @abc.abstractmethod
    def get_protocol(self):
        if False:
            while True:
                i = 10
        'Return the protocol linked to this transport.\n\n        :rtype: AbstractStreamProtocol\n        :raises Exception: Exception-based exception on error\n        '
        raise NotImplementedError

    @abc.abstractmethod
    def write(self, data):
        if False:
            for i in range(10):
                print('nop')
        'Buffer the given data until it can be sent asynchronously.\n\n        :param bytes data:\n        :raises ValueError: if called with empty data\n        :raises Exception: Exception-based exception on error\n        '
        raise NotImplementedError

    @abc.abstractmethod
    def get_write_buffer_size(self):
        if False:
            print('Hello World!')
        '\n        :returns: Current size of output data buffered by the transport\n        :rtype: int\n        '
        raise NotImplementedError