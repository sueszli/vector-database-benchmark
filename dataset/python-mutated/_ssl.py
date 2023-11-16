from __future__ import annotations
import contextlib
import operator as _operator
import ssl as _stdlib_ssl
from enum import Enum as _Enum
from typing import TYPE_CHECKING, Any, ClassVar, Final as TFinal, Generic, TypeVar
import trio
from . import _sync
from ._highlevel_generic import aclose_forcefully
from ._util import ConflictDetector, final
from .abc import Listener, Stream
if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
T = TypeVar('T')
STARTING_RECEIVE_SIZE: TFinal = 16384

def _is_eof(exc: BaseException | None) -> bool:
    if False:
        while True:
            i = 10
    return isinstance(exc, _stdlib_ssl.SSLEOFError) or 'UNEXPECTED_EOF_WHILE_READING' in getattr(exc, 'strerror', ())

class NeedHandshakeError(Exception):
    """Some :class:`SSLStream` methods can't return any meaningful data until
    after the handshake. If you call them before the handshake, they raise
    this error.

    """

class _Once:

    def __init__(self, afn: Callable[..., Awaitable[object]], *args: object) -> None:
        if False:
            return 10
        self._afn = afn
        self._args = args
        self.started = False
        self._done = _sync.Event()

    async def ensure(self, *, checkpoint: bool) -> None:
        if not self.started:
            self.started = True
            await self._afn(*self._args)
            self._done.set()
        elif not checkpoint and self._done.is_set():
            return
        else:
            await self._done.wait()

    @property
    def done(self) -> bool:
        if False:
            print('Hello World!')
        return bool(self._done.is_set())
_State = _Enum('_State', ['OK', 'BROKEN', 'CLOSED'])
T_Stream = TypeVar('T_Stream', bound=Stream)

@final
class SSLStream(Stream, Generic[T_Stream]):
    """Encrypted communication using SSL/TLS.

    :class:`SSLStream` wraps an arbitrary :class:`~trio.abc.Stream`, and
    allows you to perform encrypted communication over it using the usual
    :class:`~trio.abc.Stream` interface. You pass regular data to
    :meth:`send_all`, then it encrypts it and sends the encrypted data on the
    underlying :class:`~trio.abc.Stream`; :meth:`receive_some` takes encrypted
    data out of the underlying :class:`~trio.abc.Stream` and decrypts it
    before returning it.

    You should read the standard library's :mod:`ssl` documentation carefully
    before attempting to use this class, and probably other general
    documentation on SSL/TLS as well. SSL/TLS is subtle and quick to
    anger. Really. I'm not kidding.

    Args:
      transport_stream (~trio.abc.Stream): The stream used to transport
          encrypted data. Required.

      ssl_context (~ssl.SSLContext): The :class:`~ssl.SSLContext` used for
          this connection. Required. Usually created by calling
          :func:`ssl.create_default_context`.

      server_hostname (str, bytes, or None): The name of the server being
          connected to. Used for `SNI
          <https://en.wikipedia.org/wiki/Server_Name_Indication>`__ and for
          validating the server's certificate (if hostname checking is
          enabled). This is effectively mandatory for clients, and actually
          mandatory if ``ssl_context.check_hostname`` is ``True``.

      server_side (bool): Whether this stream is acting as a client or
          server. Defaults to False, i.e. client mode.

      https_compatible (bool): There are two versions of SSL/TLS commonly
          encountered in the wild: the standard version, and the version used
          for HTTPS (HTTP-over-SSL/TLS).

          Standard-compliant SSL/TLS implementations always send a
          cryptographically signed ``close_notify`` message before closing the
          connection. This is important because if the underlying transport
          were simply closed, then there wouldn't be any way for the other
          side to know whether the connection was intentionally closed by the
          peer that they negotiated a cryptographic connection to, or by some
          `man-in-the-middle
          <https://en.wikipedia.org/wiki/Man-in-the-middle_attack>`__ attacker
          who can't manipulate the cryptographic stream, but can manipulate
          the transport layer (a so-called "truncation attack").

          However, this part of the standard is widely ignored by real-world
          HTTPS implementations, which means that if you want to interoperate
          with them, then you NEED to ignore it too.

          Fortunately this isn't as bad as it sounds, because the HTTP
          protocol already includes its own equivalent of ``close_notify``, so
          doing this again at the SSL/TLS level is redundant. But not all
          protocols do! Therefore, by default Trio implements the safer
          standard-compliant version (``https_compatible=False``). But if
          you're speaking HTTPS or some other protocol where
          ``close_notify``\\s are commonly skipped, then you should set
          ``https_compatible=True``; with this setting, Trio will neither
          expect nor send ``close_notify`` messages.

          If you have code that was written to use :class:`ssl.SSLSocket` and
          now you're porting it to Trio, then it may be useful to know that a
          difference between :class:`SSLStream` and :class:`ssl.SSLSocket` is
          that :class:`~ssl.SSLSocket` implements the
          ``https_compatible=True`` behavior by default.

    Attributes:
      transport_stream (trio.abc.Stream): The underlying transport stream
          that was passed to ``__init__``. An example of when this would be
          useful is if you're using :class:`SSLStream` over a
          :class:`~trio.SocketStream` and want to call the
          :class:`~trio.SocketStream`'s :meth:`~trio.SocketStream.setsockopt`
          method.

    Internally, this class is implemented using an instance of
    :class:`ssl.SSLObject`, and all of :class:`~ssl.SSLObject`'s methods and
    attributes are re-exported as methods and attributes on this class.
    However, there is one difference: :class:`~ssl.SSLObject` has several
    methods that return information about the encrypted connection, like
    :meth:`~ssl.SSLSocket.cipher` or
    :meth:`~ssl.SSLSocket.selected_alpn_protocol`. If you call them before the
    handshake, when they can't possibly return useful data, then
    :class:`ssl.SSLObject` returns None, but :class:`trio.SSLStream`
    raises :exc:`NeedHandshakeError`.

    This also means that if you register a SNI callback using
    `~ssl.SSLContext.sni_callback`, then the first argument your callback
    receives will be a :class:`ssl.SSLObject`.

    """

    def __init__(self, transport_stream: T_Stream, ssl_context: _stdlib_ssl.SSLContext, *, server_hostname: str | bytes | None=None, server_side: bool=False, https_compatible: bool=False) -> None:
        if False:
            i = 10
            return i + 15
        self.transport_stream: T_Stream = transport_stream
        self._state = _State.OK
        self._https_compatible = https_compatible
        self._outgoing = _stdlib_ssl.MemoryBIO()
        self._delayed_outgoing: bytes | None = None
        self._incoming = _stdlib_ssl.MemoryBIO()
        self._ssl_object = ssl_context.wrap_bio(self._incoming, self._outgoing, server_side=server_side, server_hostname=server_hostname)
        self._handshook = _Once(self._do_handshake)
        self._inner_send_lock = _sync.StrictFIFOLock()
        self._inner_recv_count = 0
        self._inner_recv_lock = _sync.Lock()
        self._outer_send_conflict_detector = ConflictDetector('another task is currently sending data on this SSLStream')
        self._outer_recv_conflict_detector = ConflictDetector('another task is currently receiving data on this SSLStream')
        self._estimated_receive_size = STARTING_RECEIVE_SIZE
    _forwarded: ClassVar = {'context', 'server_side', 'server_hostname', 'session', 'session_reused', 'getpeercert', 'selected_npn_protocol', 'cipher', 'shared_ciphers', 'compression', 'pending', 'get_channel_binding', 'selected_alpn_protocol', 'version'}
    _after_handshake: ClassVar = {'session_reused', 'getpeercert', 'selected_npn_protocol', 'cipher', 'shared_ciphers', 'compression', 'get_channel_binding', 'selected_alpn_protocol', 'version'}

    def __getattr__(self, name: str) -> Any:
        if False:
            print('Hello World!')
        if name in self._forwarded:
            if name in self._after_handshake and (not self._handshook.done):
                raise NeedHandshakeError(f'call do_handshake() before calling {name!r}')
            return getattr(self._ssl_object, name)
        else:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: object) -> None:
        if False:
            while True:
                i = 10
        if name in self._forwarded:
            setattr(self._ssl_object, name, value)
        else:
            super().__setattr__(name, value)

    def __dir__(self) -> list[str]:
        if False:
            i = 10
            return i + 15
        return list(super().__dir__()) + list(self._forwarded)

    def _check_status(self) -> None:
        if False:
            while True:
                i = 10
        if self._state is _State.OK:
            return
        elif self._state is _State.BROKEN:
            raise trio.BrokenResourceError
        elif self._state is _State.CLOSED:
            raise trio.ClosedResourceError
        else:
            raise AssertionError()

    async def _retry(self, fn: Callable[..., T], *args: object, ignore_want_read: bool=False, is_handshake: bool=False) -> T | None:
        await trio.lowlevel.checkpoint_if_cancelled()
        yielded = False
        finished = False
        while not finished:
            want_read = False
            ret = None
            try:
                ret = fn(*args)
            except _stdlib_ssl.SSLWantReadError:
                want_read = True
            except (_stdlib_ssl.SSLError, _stdlib_ssl.CertificateError) as exc:
                self._state = _State.BROKEN
                raise trio.BrokenResourceError from exc
            else:
                finished = True
            if ignore_want_read:
                want_read = False
                finished = True
            to_send = self._outgoing.read()
            if is_handshake and (not want_read) and self._ssl_object.server_side and (self._ssl_object.version() == 'TLSv1.3'):
                assert self._delayed_outgoing is None
                self._delayed_outgoing = to_send
                to_send = b''
            if to_send:
                async with self._inner_send_lock:
                    yielded = True
                    try:
                        if self._delayed_outgoing is not None:
                            to_send = self._delayed_outgoing + to_send
                            self._delayed_outgoing = None
                        await self.transport_stream.send_all(to_send)
                    except:
                        self._state = _State.BROKEN
                        raise
            elif want_read:
                recv_count = self._inner_recv_count
                async with self._inner_recv_lock:
                    yielded = True
                    if recv_count == self._inner_recv_count:
                        data = await self.transport_stream.receive_some()
                        if not data:
                            self._incoming.write_eof()
                        else:
                            self._estimated_receive_size = max(self._estimated_receive_size, len(data))
                            self._incoming.write(data)
                        self._inner_recv_count += 1
        if not yielded:
            await trio.lowlevel.cancel_shielded_checkpoint()
        return ret

    async def _do_handshake(self) -> None:
        try:
            await self._retry(self._ssl_object.do_handshake, is_handshake=True)
        except:
            self._state = _State.BROKEN
            raise

    async def do_handshake(self) -> None:
        """Ensure that the initial handshake has completed.

        The SSL protocol requires an initial handshake to exchange
        certificates, select cryptographic keys, and so forth, before any
        actual data can be sent or received. You don't have to call this
        method; if you don't, then :class:`SSLStream` will automatically
        perform the handshake as needed, the first time you try to send or
        receive data. But if you want to trigger it manually – for example,
        because you want to look at the peer's certificate before you start
        talking to them – then you can call this method.

        If the initial handshake is already in progress in another task, this
        waits for it to complete and then returns.

        If the initial handshake has already completed, this returns
        immediately without doing anything (except executing a checkpoint).

        .. warning:: If this method is cancelled, then it may leave the
           :class:`SSLStream` in an unusable state. If this happens then any
           future attempt to use the object will raise
           :exc:`trio.BrokenResourceError`.

        """
        self._check_status()
        await self._handshook.ensure(checkpoint=True)

    async def receive_some(self, max_bytes: int | None=None) -> bytes | bytearray:
        """Read some data from the underlying transport, decrypt it, and
        return it.

        See :meth:`trio.abc.ReceiveStream.receive_some` for details.

        .. warning:: If this method is cancelled while the initial handshake
           or a renegotiation are in progress, then it may leave the
           :class:`SSLStream` in an unusable state. If this happens then any
           future attempt to use the object will raise
           :exc:`trio.BrokenResourceError`.

        """
        with self._outer_recv_conflict_detector:
            self._check_status()
            try:
                await self._handshook.ensure(checkpoint=False)
            except trio.BrokenResourceError as exc:
                if self._https_compatible and (isinstance(exc.__cause__, _stdlib_ssl.SSLSyscallError) or _is_eof(exc.__cause__)):
                    await trio.lowlevel.checkpoint()
                    return b''
                else:
                    raise
            if max_bytes is None:
                max_bytes = max(self._estimated_receive_size, self._incoming.pending)
            else:
                max_bytes = _operator.index(max_bytes)
                if max_bytes < 1:
                    raise ValueError('max_bytes must be >= 1')
            try:
                received = await self._retry(self._ssl_object.read, max_bytes)
                assert received is not None
                return received
            except trio.BrokenResourceError as exc:
                if self._https_compatible and _is_eof(exc.__cause__):
                    await trio.lowlevel.checkpoint()
                    return b''
                else:
                    raise

    async def send_all(self, data: bytes | bytearray | memoryview) -> None:
        """Encrypt some data and then send it on the underlying transport.

        See :meth:`trio.abc.SendStream.send_all` for details.

        .. warning:: If this method is cancelled, then it may leave the
           :class:`SSLStream` in an unusable state. If this happens then any
           attempt to use the object will raise
           :exc:`trio.BrokenResourceError`.

        """
        with self._outer_send_conflict_detector:
            self._check_status()
            await self._handshook.ensure(checkpoint=False)
            if not data:
                await trio.lowlevel.checkpoint()
                return
            await self._retry(self._ssl_object.write, data)

    async def unwrap(self) -> tuple[Stream, bytes | bytearray]:
        """Cleanly close down the SSL/TLS encryption layer, allowing the
        underlying stream to be used for unencrypted communication.

        You almost certainly don't need this.

        Returns:
          A pair ``(transport_stream, trailing_bytes)``, where
          ``transport_stream`` is the underlying transport stream, and
          ``trailing_bytes`` is a byte string. Since :class:`SSLStream`
          doesn't necessarily know where the end of the encrypted data will
          be, it can happen that it accidentally reads too much from the
          underlying stream. ``trailing_bytes`` contains this extra data; you
          should process it as if it was returned from a call to
          ``transport_stream.receive_some(...)``.

        """
        with self._outer_recv_conflict_detector, self._outer_send_conflict_detector:
            self._check_status()
            await self._handshook.ensure(checkpoint=False)
            await self._retry(self._ssl_object.unwrap)
            transport_stream = self.transport_stream
            self._state = _State.CLOSED
            self.transport_stream = None
            return (transport_stream, self._incoming.read())

    async def aclose(self) -> None:
        """Gracefully shut down this connection, and close the underlying
        transport.

        If ``https_compatible`` is False (the default), then this attempts to
        first send a ``close_notify`` and then close the underlying stream by
        calling its :meth:`~trio.abc.AsyncResource.aclose` method.

        If ``https_compatible`` is set to True, then this simply closes the
        underlying stream and marks this stream as closed.

        """
        if self._state is _State.CLOSED:
            await trio.lowlevel.checkpoint()
            return
        if self._state is _State.BROKEN or self._https_compatible:
            self._state = _State.CLOSED
            await self.transport_stream.aclose()
            return
        try:
            await self._handshook.ensure(checkpoint=False)
            with contextlib.suppress(trio.BrokenResourceError, trio.BusyResourceError):
                await self._retry(self._ssl_object.unwrap, ignore_want_read=True)
        except:
            await aclose_forcefully(self.transport_stream)
            raise
        else:
            await self.transport_stream.aclose()
        finally:
            self._state = _State.CLOSED

    async def wait_send_all_might_not_block(self) -> None:
        """See :meth:`trio.abc.SendStream.wait_send_all_might_not_block`."""
        with self._outer_send_conflict_detector:
            self._check_status()
            async with self._inner_send_lock:
                await self.transport_stream.wait_send_all_might_not_block()

@final
class SSLListener(Listener[SSLStream[T_Stream]]):
    """A :class:`~trio.abc.Listener` for SSL/TLS-encrypted servers.

    :class:`SSLListener` wraps around another Listener, and converts
    all incoming connections to encrypted connections by wrapping them
    in a :class:`SSLStream`.

    Args:
      transport_listener (~trio.abc.Listener): The listener whose incoming
          connections will be wrapped in :class:`SSLStream`.

      ssl_context (~ssl.SSLContext): The :class:`~ssl.SSLContext` that will be
          used for incoming connections.

      https_compatible (bool): Passed on to :class:`SSLStream`.

    Attributes:
      transport_listener (trio.abc.Listener): The underlying listener that was
          passed to ``__init__``.

    """

    def __init__(self, transport_listener: Listener[T_Stream], ssl_context: _stdlib_ssl.SSLContext, *, https_compatible: bool=False) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.transport_listener = transport_listener
        self._ssl_context = ssl_context
        self._https_compatible = https_compatible

    async def accept(self) -> SSLStream[T_Stream]:
        """Accept the next connection and wrap it in an :class:`SSLStream`.

        See :meth:`trio.abc.Listener.accept` for details.

        """
        transport_stream = await self.transport_listener.accept()
        return SSLStream(transport_stream, self._ssl_context, server_side=True, https_compatible=self._https_compatible)

    async def aclose(self) -> None:
        """Close the transport listener."""
        await self.transport_listener.aclose()