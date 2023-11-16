from __future__ import annotations
import operator
from typing import TYPE_CHECKING, Awaitable, Callable, TypeVar
from .. import _core, _util
from .._highlevel_generic import StapledStream
from ..abc import ReceiveStream, SendStream
if TYPE_CHECKING:
    from typing_extensions import TypeAlias
AsyncHook: TypeAlias = Callable[[], Awaitable[object]]
SyncHook: TypeAlias = Callable[[], object]
SendStreamT = TypeVar('SendStreamT', bound=SendStream)
ReceiveStreamT = TypeVar('ReceiveStreamT', bound=ReceiveStream)

class _UnboundedByteQueue:

    def __init__(self) -> None:
        if False:
            return 10
        self._data = bytearray()
        self._closed = False
        self._lot = _core.ParkingLot()
        self._fetch_lock = _util.ConflictDetector('another task is already fetching data')

    def close(self) -> None:
        if False:
            while True:
                i = 10
        self._closed = True
        self._lot.unpark_all()

    def close_and_wipe(self) -> None:
        if False:
            print('Hello World!')
        self._data = bytearray()
        self.close()

    def put(self, data: bytes | bytearray | memoryview) -> None:
        if False:
            while True:
                i = 10
        if self._closed:
            raise _core.ClosedResourceError('virtual connection closed')
        self._data += data
        self._lot.unpark_all()

    def _check_max_bytes(self, max_bytes: int | None) -> None:
        if False:
            print('Hello World!')
        if max_bytes is None:
            return
        max_bytes = operator.index(max_bytes)
        if max_bytes < 1:
            raise ValueError('max_bytes must be >= 1')

    def _get_impl(self, max_bytes: int | None) -> bytearray:
        if False:
            i = 10
            return i + 15
        assert self._closed or self._data
        if max_bytes is None:
            max_bytes = len(self._data)
        if self._data:
            chunk = self._data[:max_bytes]
            del self._data[:max_bytes]
            assert chunk
            return chunk
        else:
            return bytearray()

    def get_nowait(self, max_bytes: int | None=None) -> bytearray:
        if False:
            for i in range(10):
                print('nop')
        with self._fetch_lock:
            self._check_max_bytes(max_bytes)
            if not self._closed and (not self._data):
                raise _core.WouldBlock
            return self._get_impl(max_bytes)

    async def get(self, max_bytes: int | None=None) -> bytearray:
        with self._fetch_lock:
            self._check_max_bytes(max_bytes)
            if not self._closed and (not self._data):
                await self._lot.park()
            else:
                await _core.checkpoint()
            return self._get_impl(max_bytes)

@_util.final
class MemorySendStream(SendStream):
    """An in-memory :class:`~trio.abc.SendStream`.

    Args:
      send_all_hook: An async function, or None. Called from
          :meth:`send_all`. Can do whatever you like.
      wait_send_all_might_not_block_hook: An async function, or None. Called
          from :meth:`wait_send_all_might_not_block`. Can do whatever you
          like.
      close_hook: A synchronous function, or None. Called from :meth:`close`
          and :meth:`aclose`. Can do whatever you like.

    .. attribute:: send_all_hook
                   wait_send_all_might_not_block_hook
                   close_hook

       All of these hooks are also exposed as attributes on the object, and
       you can change them at any time.

    """

    def __init__(self, send_all_hook: AsyncHook | None=None, wait_send_all_might_not_block_hook: AsyncHook | None=None, close_hook: SyncHook | None=None):
        if False:
            return 10
        self._conflict_detector = _util.ConflictDetector('another task is using this stream')
        self._outgoing = _UnboundedByteQueue()
        self.send_all_hook = send_all_hook
        self.wait_send_all_might_not_block_hook = wait_send_all_might_not_block_hook
        self.close_hook = close_hook

    async def send_all(self, data: bytes | bytearray | memoryview) -> None:
        """Places the given data into the object's internal buffer, and then
        calls the :attr:`send_all_hook` (if any).

        """
        with self._conflict_detector:
            await _core.checkpoint()
            await _core.checkpoint()
            self._outgoing.put(data)
            if self.send_all_hook is not None:
                await self.send_all_hook()

    async def wait_send_all_might_not_block(self) -> None:
        """Calls the :attr:`wait_send_all_might_not_block_hook` (if any), and
        then returns immediately.

        """
        with self._conflict_detector:
            await _core.checkpoint()
            await _core.checkpoint()
            self._outgoing.put(b'')
            if self.wait_send_all_might_not_block_hook is not None:
                await self.wait_send_all_might_not_block_hook()

    def close(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Marks this stream as closed, and then calls the :attr:`close_hook`\n        (if any).\n\n        '
        self._outgoing.close()
        if self.close_hook is not None:
            self.close_hook()

    async def aclose(self) -> None:
        """Same as :meth:`close`, but async."""
        self.close()
        await _core.checkpoint()

    async def get_data(self, max_bytes: int | None=None) -> bytearray:
        """Retrieves data from the internal buffer, blocking if necessary.

        Args:
          max_bytes (int or None): The maximum amount of data to
              retrieve. None (the default) means to retrieve all the data
              that's present (but still blocks until at least one byte is
              available).

        Returns:
          If this stream has been closed, an empty bytearray. Otherwise, the
          requested data.

        """
        return await self._outgoing.get(max_bytes)

    def get_data_nowait(self, max_bytes: int | None=None) -> bytearray:
        if False:
            while True:
                i = 10
        "Retrieves data from the internal buffer, but doesn't block.\n\n        See :meth:`get_data` for details.\n\n        Raises:\n          trio.WouldBlock: if no data is available to retrieve.\n\n        "
        return self._outgoing.get_nowait(max_bytes)

@_util.final
class MemoryReceiveStream(ReceiveStream):
    """An in-memory :class:`~trio.abc.ReceiveStream`.

    Args:
      receive_some_hook: An async function, or None. Called from
          :meth:`receive_some`. Can do whatever you like.
      close_hook: A synchronous function, or None. Called from :meth:`close`
          and :meth:`aclose`. Can do whatever you like.

    .. attribute:: receive_some_hook
                   close_hook

       Both hooks are also exposed as attributes on the object, and you can
       change them at any time.

    """

    def __init__(self, receive_some_hook: AsyncHook | None=None, close_hook: SyncHook | None=None):
        if False:
            while True:
                i = 10
        self._conflict_detector = _util.ConflictDetector('another task is using this stream')
        self._incoming = _UnboundedByteQueue()
        self._closed = False
        self.receive_some_hook = receive_some_hook
        self.close_hook = close_hook

    async def receive_some(self, max_bytes: int | None=None) -> bytearray:
        """Calls the :attr:`receive_some_hook` (if any), and then retrieves
        data from the internal buffer, blocking if necessary.

        """
        with self._conflict_detector:
            await _core.checkpoint()
            await _core.checkpoint()
            if self._closed:
                raise _core.ClosedResourceError
            if self.receive_some_hook is not None:
                await self.receive_some_hook()
            data = await self._incoming.get(max_bytes)
            if self._closed:
                raise _core.ClosedResourceError
            return data

    def close(self) -> None:
        if False:
            while True:
                i = 10
        'Discards any pending data from the internal buffer, and marks this\n        stream as closed.\n\n        '
        self._closed = True
        self._incoming.close_and_wipe()
        if self.close_hook is not None:
            self.close_hook()

    async def aclose(self) -> None:
        """Same as :meth:`close`, but async."""
        self.close()
        await _core.checkpoint()

    def put_data(self, data: bytes | bytearray | memoryview) -> None:
        if False:
            while True:
                i = 10
        'Appends the given data to the internal buffer.'
        self._incoming.put(data)

    def put_eof(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Adds an end-of-file marker to the internal buffer.'
        self._incoming.close()

def memory_stream_pump(memory_send_stream: MemorySendStream, memory_receive_stream: MemoryReceiveStream, *, max_bytes: int | None=None) -> bool:
    if False:
        i = 10
        return i + 15
    "Take data out of the given :class:`MemorySendStream`'s internal buffer,\n    and put it into the given :class:`MemoryReceiveStream`'s internal buffer.\n\n    Args:\n      memory_send_stream (MemorySendStream): The stream to get data from.\n      memory_receive_stream (MemoryReceiveStream): The stream to put data into.\n      max_bytes (int or None): The maximum amount of data to transfer in this\n          call, or None to transfer all available data.\n\n    Returns:\n      True if it successfully transferred some data, or False if there was no\n      data to transfer.\n\n    This is used to implement :func:`memory_stream_one_way_pair` and\n    :func:`memory_stream_pair`; see the latter's docstring for an example\n    of how you might use it yourself.\n\n    "
    try:
        data = memory_send_stream.get_data_nowait(max_bytes)
    except _core.WouldBlock:
        return False
    try:
        if not data:
            memory_receive_stream.put_eof()
        else:
            memory_receive_stream.put_data(data)
    except _core.ClosedResourceError:
        raise _core.BrokenResourceError('MemoryReceiveStream was closed') from None
    return True

def memory_stream_one_way_pair() -> tuple[MemorySendStream, MemoryReceiveStream]:
    if False:
        for i in range(10):
            print('nop')
    "Create a connected, pure-Python, unidirectional stream with infinite\n    buffering and flexible configuration options.\n\n    You can think of this as being a no-operating-system-involved\n    Trio-streamsified version of :func:`os.pipe` (except that :func:`os.pipe`\n    returns the streams in the wrong order – we follow the superior convention\n    that data flows from left to right).\n\n    Returns:\n      A tuple (:class:`MemorySendStream`, :class:`MemoryReceiveStream`), where\n      the :class:`MemorySendStream` has its hooks set up so that it calls\n      :func:`memory_stream_pump` from its\n      :attr:`~MemorySendStream.send_all_hook` and\n      :attr:`~MemorySendStream.close_hook`.\n\n    The end result is that data automatically flows from the\n    :class:`MemorySendStream` to the :class:`MemoryReceiveStream`. But you're\n    also free to rearrange things however you like. For example, you can\n    temporarily set the :attr:`~MemorySendStream.send_all_hook` to None if you\n    want to simulate a stall in data transmission. Or see\n    :func:`memory_stream_pair` for a more elaborate example.\n\n    "
    send_stream = MemorySendStream()
    recv_stream = MemoryReceiveStream()

    def pump_from_send_stream_to_recv_stream() -> None:
        if False:
            return 10
        memory_stream_pump(send_stream, recv_stream)

    async def async_pump_from_send_stream_to_recv_stream() -> None:
        pump_from_send_stream_to_recv_stream()
    send_stream.send_all_hook = async_pump_from_send_stream_to_recv_stream
    send_stream.close_hook = pump_from_send_stream_to_recv_stream
    return (send_stream, recv_stream)

def _make_stapled_pair(one_way_pair: Callable[[], tuple[SendStreamT, ReceiveStreamT]]) -> tuple[StapledStream[SendStreamT, ReceiveStreamT], StapledStream[SendStreamT, ReceiveStreamT]]:
    if False:
        for i in range(10):
            print('nop')
    (pipe1_send, pipe1_recv) = one_way_pair()
    (pipe2_send, pipe2_recv) = one_way_pair()
    stream1 = StapledStream(pipe1_send, pipe2_recv)
    stream2 = StapledStream(pipe2_send, pipe1_recv)
    return (stream1, stream2)

def memory_stream_pair() -> tuple[StapledStream[MemorySendStream, MemoryReceiveStream], StapledStream[MemorySendStream, MemoryReceiveStream]]:
    if False:
        i = 10
        return i + 15
    'Create a connected, pure-Python, bidirectional stream with infinite\n    buffering and flexible configuration options.\n\n    This is a convenience function that creates two one-way streams using\n    :func:`memory_stream_one_way_pair`, and then uses\n    :class:`~trio.StapledStream` to combine them into a single bidirectional\n    stream.\n\n    This is like a no-operating-system-involved, Trio-streamsified version of\n    :func:`socket.socketpair`.\n\n    Returns:\n      A pair of :class:`~trio.StapledStream` objects that are connected so\n      that data automatically flows from one to the other in both directions.\n\n    After creating a stream pair, you can send data back and forth, which is\n    enough for simple tests::\n\n       left, right = memory_stream_pair()\n       await left.send_all(b"123")\n       assert await right.receive_some() == b"123"\n       await right.send_all(b"456")\n       assert await left.receive_some() == b"456"\n\n    But if you read the docs for :class:`~trio.StapledStream` and\n    :func:`memory_stream_one_way_pair`, you\'ll see that all the pieces\n    involved in wiring this up are public APIs, so you can adjust to suit the\n    requirements of your tests. For example, here\'s how to tweak a stream so\n    that data flowing from left to right trickles in one byte at a time (but\n    data flowing from right to left proceeds at full speed)::\n\n        left, right = memory_stream_pair()\n        async def trickle():\n            # left is a StapledStream, and left.send_stream is a MemorySendStream\n            # right is a StapledStream, and right.recv_stream is a MemoryReceiveStream\n            while memory_stream_pump(left.send_stream, right.recv_stream, max_bytes=1):\n                # Pause between each byte\n                await trio.sleep(1)\n        # Normally this send_all_hook calls memory_stream_pump directly without\n        # passing in a max_bytes. We replace it with our custom version:\n        left.send_stream.send_all_hook = trickle\n\n    And here\'s a simple test using our modified stream objects::\n\n        async def sender():\n            await left.send_all(b"12345")\n            await left.send_eof()\n\n        async def receiver():\n            async for data in right:\n                print(data)\n\n        async with trio.open_nursery() as nursery:\n            nursery.start_soon(sender)\n            nursery.start_soon(receiver)\n\n    By default, this will print ``b"12345"`` and then immediately exit; with\n    our trickle stream it instead sleeps 1 second, then prints ``b"1"``, then\n    sleeps 1 second, then prints ``b"2"``, etc.\n\n    Pro-tip: you can insert sleep calls (like in our example above) to\n    manipulate the flow of data across tasks... and then use\n    :class:`MockClock` and its :attr:`~MockClock.autojump_threshold`\n    functionality to keep your test suite running quickly.\n\n    If you want to stress test a protocol implementation, one nice trick is to\n    use the :mod:`random` module (preferably with a fixed seed) to move random\n    numbers of bytes at a time, and insert random sleeps in between them. You\n    can also set up a custom :attr:`~MemoryReceiveStream.receive_some_hook` if\n    you want to manipulate things on the receiving side, and not just the\n    sending side.\n\n    '
    return _make_stapled_pair(memory_stream_one_way_pair)

class _LockstepByteQueue:

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        self._data = bytearray()
        self._sender_closed = False
        self._receiver_closed = False
        self._receiver_waiting = False
        self._waiters = _core.ParkingLot()
        self._send_conflict_detector = _util.ConflictDetector('another task is already sending')
        self._receive_conflict_detector = _util.ConflictDetector('another task is already receiving')

    def _something_happened(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._waiters.unpark_all()

    async def _wait_for(self, fn: Callable[[], bool]) -> None:
        while True:
            if fn():
                break
            if self._sender_closed or self._receiver_closed:
                break
            await self._waiters.park()
        await _core.checkpoint()

    def close_sender(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._sender_closed = True
        self._something_happened()

    def close_receiver(self) -> None:
        if False:
            print('Hello World!')
        self._receiver_closed = True
        self._something_happened()

    async def send_all(self, data: bytes | bytearray | memoryview) -> None:
        with self._send_conflict_detector:
            if self._sender_closed:
                raise _core.ClosedResourceError
            if self._receiver_closed:
                raise _core.BrokenResourceError
            assert not self._data
            self._data += data
            self._something_happened()
            await self._wait_for(lambda : self._data == b'')
            if self._sender_closed:
                raise _core.ClosedResourceError
            if self._data and self._receiver_closed:
                raise _core.BrokenResourceError

    async def wait_send_all_might_not_block(self) -> None:
        with self._send_conflict_detector:
            if self._sender_closed:
                raise _core.ClosedResourceError
            if self._receiver_closed:
                await _core.checkpoint()
                return
            await self._wait_for(lambda : self._receiver_waiting)
            if self._sender_closed:
                raise _core.ClosedResourceError

    async def receive_some(self, max_bytes: int | None=None) -> bytes | bytearray:
        with self._receive_conflict_detector:
            if max_bytes is not None:
                max_bytes = operator.index(max_bytes)
                if max_bytes < 1:
                    raise ValueError('max_bytes must be >= 1')
            if self._receiver_closed:
                raise _core.ClosedResourceError
            self._receiver_waiting = True
            self._something_happened()
            try:
                await self._wait_for(lambda : self._data != b'')
            finally:
                self._receiver_waiting = False
            if self._receiver_closed:
                raise _core.ClosedResourceError
            if self._data:
                got = self._data[:max_bytes]
                del self._data[:max_bytes]
                self._something_happened()
                return got
            else:
                assert self._sender_closed
                return b''

class _LockstepSendStream(SendStream):

    def __init__(self, lbq: _LockstepByteQueue):
        if False:
            return 10
        self._lbq = lbq

    def close(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._lbq.close_sender()

    async def aclose(self) -> None:
        self.close()
        await _core.checkpoint()

    async def send_all(self, data: bytes | bytearray | memoryview) -> None:
        await self._lbq.send_all(data)

    async def wait_send_all_might_not_block(self) -> None:
        await self._lbq.wait_send_all_might_not_block()

class _LockstepReceiveStream(ReceiveStream):

    def __init__(self, lbq: _LockstepByteQueue):
        if False:
            for i in range(10):
                print('nop')
        self._lbq = lbq

    def close(self) -> None:
        if False:
            print('Hello World!')
        self._lbq.close_receiver()

    async def aclose(self) -> None:
        self.close()
        await _core.checkpoint()

    async def receive_some(self, max_bytes: int | None=None) -> bytes | bytearray:
        return await self._lbq.receive_some(max_bytes)

def lockstep_stream_one_way_pair() -> tuple[SendStream, ReceiveStream]:
    if False:
        for i in range(10):
            print('nop')
    'Create a connected, pure Python, unidirectional stream where data flows\n    in lockstep.\n\n    Returns:\n      A tuple\n      (:class:`~trio.abc.SendStream`, :class:`~trio.abc.ReceiveStream`).\n\n    This stream has *absolutely no* buffering. Each call to\n    :meth:`~trio.abc.SendStream.send_all` will block until all the given data\n    has been returned by a call to\n    :meth:`~trio.abc.ReceiveStream.receive_some`.\n\n    This can be useful for testing flow control mechanisms in an extreme case,\n    or for setting up "clogged" streams to use with\n    :func:`check_one_way_stream` and friends.\n\n    In addition to fulfilling the :class:`~trio.abc.SendStream` and\n    :class:`~trio.abc.ReceiveStream` interfaces, the return objects\n    also have a synchronous ``close`` method.\n\n    '
    lbq = _LockstepByteQueue()
    return (_LockstepSendStream(lbq), _LockstepReceiveStream(lbq))

def lockstep_stream_pair() -> tuple[StapledStream[SendStream, ReceiveStream], StapledStream[SendStream, ReceiveStream]]:
    if False:
        while True:
            i = 10
    'Create a connected, pure-Python, bidirectional stream where data flows\n    in lockstep.\n\n    Returns:\n      A tuple (:class:`~trio.StapledStream`, :class:`~trio.StapledStream`).\n\n    This is a convenience function that creates two one-way streams using\n    :func:`lockstep_stream_one_way_pair`, and then uses\n    :class:`~trio.StapledStream` to combine them into a single bidirectional\n    stream.\n\n    '
    return _make_stapled_pair(lockstep_stream_one_way_pair)