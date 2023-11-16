from __future__ import annotations
import random
from contextlib import contextmanager, suppress
from typing import TYPE_CHECKING, Awaitable, Callable, Generic, Tuple, TypeVar
from .. import CancelScope, _core
from .._abc import AsyncResource, HalfCloseableStream, ReceiveStream, SendStream, Stream
from .._highlevel_generic import aclose_forcefully
from ._checkpoints import assert_checkpoints
if TYPE_CHECKING:
    from collections.abc import Generator
    from types import TracebackType
    from typing_extensions import ParamSpec, TypeAlias
    ArgsT = ParamSpec('ArgsT')
Res1 = TypeVar('Res1', bound=AsyncResource)
Res2 = TypeVar('Res2', bound=AsyncResource)
StreamMaker: TypeAlias = Callable[[], Awaitable[Tuple[Res1, Res2]]]

class _ForceCloseBoth(Generic[Res1, Res2]):

    def __init__(self, both: tuple[Res1, Res2]) -> None:
        if False:
            while True:
                i = 10
        (self._first, self._second) = both

    async def __aenter__(self) -> tuple[Res1, Res2]:
        return (self._first, self._second)

    async def __aexit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None) -> None:
        try:
            await aclose_forcefully(self._first)
        finally:
            await aclose_forcefully(self._second)

@contextmanager
def _assert_raises(exc: type[BaseException]) -> Generator[None, None, None]:
    if False:
        return 10
    __tracebackhide__ = True
    try:
        yield
    except exc:
        pass
    else:
        raise AssertionError(f'expected exception: {exc}')

async def check_one_way_stream(stream_maker: StreamMaker[SendStream, ReceiveStream], clogged_stream_maker: StreamMaker[SendStream, ReceiveStream] | None) -> None:
    """Perform a number of generic tests on a custom one-way stream
    implementation.

    Args:
      stream_maker: An async (!) function which returns a connected
          (:class:`~trio.abc.SendStream`, :class:`~trio.abc.ReceiveStream`)
          pair.
      clogged_stream_maker: Either None, or an async function similar to
          stream_maker, but with the extra property that the returned stream
          is in a state where ``send_all`` and
          ``wait_send_all_might_not_block`` will block until ``receive_some``
          has been called. This allows for more thorough testing of some edge
          cases, especially around ``wait_send_all_might_not_block``.

    Raises:
      AssertionError: if a test fails.

    """
    async with _ForceCloseBoth(await stream_maker()) as (s, r):
        assert isinstance(s, SendStream)
        assert isinstance(r, ReceiveStream)

        async def do_send_all(data: bytes | bytearray | memoryview) -> None:
            with assert_checkpoints():
                assert await s.send_all(data) is None

        async def do_receive_some(max_bytes: int | None=None) -> bytes | bytearray:
            with assert_checkpoints():
                return await r.receive_some(max_bytes)

        async def checked_receive_1(expected: bytes) -> None:
            assert await do_receive_some(1) == expected

        async def do_aclose(resource: AsyncResource) -> None:
            with assert_checkpoints():
                await resource.aclose()
        async with _core.open_nursery() as nursery:
            nursery.start_soon(do_send_all, b'x')
            nursery.start_soon(checked_receive_1, b'x')

        async def send_empty_then_y() -> None:
            await do_send_all(b'')
            await do_send_all(b'y')
        async with _core.open_nursery() as nursery:
            nursery.start_soon(send_empty_then_y)
            nursery.start_soon(checked_receive_1, b'y')
        async with _core.open_nursery() as nursery:
            nursery.start_soon(do_send_all, bytearray(b'1'))
            nursery.start_soon(checked_receive_1, b'1')
        async with _core.open_nursery() as nursery:
            nursery.start_soon(do_send_all, memoryview(b'2'))
            nursery.start_soon(checked_receive_1, b'2')
        with _assert_raises(ValueError):
            await r.receive_some(-1)
        with _assert_raises(ValueError):
            await r.receive_some(0)
        with _assert_raises(TypeError):
            await r.receive_some(1.5)
        async with _core.open_nursery() as nursery:
            nursery.start_soon(do_send_all, b'x')
            assert await do_receive_some() == b'x'
        async with _core.open_nursery() as nursery:
            nursery.start_soon(do_send_all, b'x')
            assert await do_receive_some(None) == b'x'
        with _assert_raises(_core.BusyResourceError):
            async with _core.open_nursery() as nursery:
                nursery.start_soon(do_receive_some, 1)
                nursery.start_soon(do_receive_some, 1)

        async def simple_check_wait_send_all_might_not_block(scope: CancelScope) -> None:
            with assert_checkpoints():
                await s.wait_send_all_might_not_block()
            scope.cancel()
        async with _core.open_nursery() as nursery:
            nursery.start_soon(simple_check_wait_send_all_might_not_block, nursery.cancel_scope)
            nursery.start_soon(do_receive_some, 1)

        async def expect_broken_stream_on_send() -> None:
            with _assert_raises(_core.BrokenResourceError):
                while True:
                    await do_send_all(b'x' * 100)
        async with _core.open_nursery() as nursery:
            nursery.start_soon(expect_broken_stream_on_send)
            nursery.start_soon(do_aclose, r)
        with _assert_raises(_core.BrokenResourceError):
            await do_send_all(b'x' * 100)
        with _assert_raises(_core.ClosedResourceError):
            await do_receive_some(4096)
        await do_aclose(r)
        await do_aclose(r)
        await do_aclose(s)
        with _assert_raises(_core.ClosedResourceError):
            await do_send_all(b'x' * 100)
        with _assert_raises(_core.ClosedResourceError):
            await do_send_all(b'')
        with _assert_raises(_core.ClosedResourceError):
            with assert_checkpoints():
                await s.wait_send_all_might_not_block()
        await do_aclose(s)
        await do_aclose(s)
    async with _ForceCloseBoth(await stream_maker()) as (s, r):

        async def send_then_close() -> None:
            await do_send_all(b'y')
            await do_aclose(s)

        async def receive_send_then_close() -> None:
            await _core.wait_all_tasks_blocked()
            await checked_receive_1(b'y')
            await checked_receive_1(b'')
            await do_aclose(r)
        async with _core.open_nursery() as nursery:
            nursery.start_soon(send_then_close)
            nursery.start_soon(receive_send_then_close)
    async with _ForceCloseBoth(await stream_maker()) as (s, r):
        await aclose_forcefully(r)
        with _assert_raises(_core.BrokenResourceError):
            while True:
                await do_send_all(b'x' * 100)
        with _assert_raises(_core.ClosedResourceError):
            await do_receive_some(4096)
    async with _ForceCloseBoth(await stream_maker()) as (s, r):
        await aclose_forcefully(s)
        with _assert_raises(_core.ClosedResourceError):
            await do_send_all(b'123')
        with suppress(_core.BrokenResourceError):
            await checked_receive_1(b'')
    async with _ForceCloseBoth(await stream_maker()) as (s, r):
        with _core.CancelScope() as scope:
            scope.cancel()
            await r.aclose()
        with _core.CancelScope() as scope:
            scope.cancel()
            await s.aclose()
        with _assert_raises(_core.ClosedResourceError):
            await do_send_all(b'123')
        with _assert_raises(_core.ClosedResourceError):
            await do_receive_some(4096)
    async with _ForceCloseBoth(await stream_maker()) as (s, r):

        async def expect_cancelled(afn: Callable[ArgsT, Awaitable[object]], *args: ArgsT.args, **kwargs: ArgsT.kwargs) -> None:
            with _assert_raises(_core.Cancelled):
                await afn(*args, **kwargs)
        with _core.CancelScope() as scope:
            scope.cancel()
            async with _core.open_nursery() as nursery:
                nursery.start_soon(expect_cancelled, do_send_all, b'x')
                nursery.start_soon(expect_cancelled, do_receive_some, 1)
        async with _core.open_nursery() as nursery:
            nursery.start_soon(do_aclose, s)
            nursery.start_soon(do_aclose, r)
    async with _ForceCloseBoth(await stream_maker()) as (s, r):

        async def receive_expecting_closed():
            with _assert_raises(_core.ClosedResourceError):
                await r.receive_some(10)
        async with _core.open_nursery() as nursery:
            nursery.start_soon(receive_expecting_closed)
            await _core.wait_all_tasks_blocked()
            await aclose_forcefully(r)
    if clogged_stream_maker is not None:
        async with _ForceCloseBoth(await clogged_stream_maker()) as (s, r):
            record: list[str] = []

            async def waiter(cancel_scope: CancelScope) -> None:
                record.append('waiter sleeping')
                with assert_checkpoints():
                    await s.wait_send_all_might_not_block()
                record.append('waiter wokeup')
                cancel_scope.cancel()

            async def receiver() -> None:
                await _core.wait_all_tasks_blocked()
                record.append('receiver starting')
                while True:
                    await r.receive_some(16834)
            async with _core.open_nursery() as nursery:
                nursery.start_soon(waiter, nursery.cancel_scope)
                await _core.wait_all_tasks_blocked()
                nursery.start_soon(receiver)
            assert record == ['waiter sleeping', 'receiver starting', 'waiter wokeup']
        async with _ForceCloseBoth(await clogged_stream_maker()) as (s, r):
            with _assert_raises(_core.BusyResourceError):
                async with _core.open_nursery() as nursery:
                    nursery.start_soon(s.wait_send_all_might_not_block)
                    nursery.start_soon(s.wait_send_all_might_not_block)
            with _assert_raises(_core.BusyResourceError):
                async with _core.open_nursery() as nursery:
                    nursery.start_soon(s.wait_send_all_might_not_block)
                    nursery.start_soon(s.send_all, b'123')
        async with _ForceCloseBoth(await clogged_stream_maker()) as (s, r):
            with _assert_raises(_core.BusyResourceError):
                async with _core.open_nursery() as nursery:
                    nursery.start_soon(s.send_all, b'123')
                    nursery.start_soon(s.send_all, b'123')
        async with _ForceCloseBoth(await clogged_stream_maker()) as (s, r):

            async def sender() -> None:
                try:
                    with assert_checkpoints():
                        await s.wait_send_all_might_not_block()
                except _core.BrokenResourceError:
                    pass

            async def receiver() -> None:
                await _core.wait_all_tasks_blocked()
                await aclose_forcefully(r)
            async with _core.open_nursery() as nursery:
                nursery.start_soon(sender)
                nursery.start_soon(receiver)
        async with _ForceCloseBoth(await clogged_stream_maker()) as (s, r):
            await aclose_forcefully(r)
            try:
                with assert_checkpoints():
                    await s.wait_send_all_might_not_block()
            except _core.BrokenResourceError:
                pass

        async def close_soon(s: SendStream) -> None:
            await _core.wait_all_tasks_blocked()
            await aclose_forcefully(s)
        async with _ForceCloseBoth(await clogged_stream_maker()) as (s, r):
            async with _core.open_nursery() as nursery:
                nursery.start_soon(close_soon, s)
                with _assert_raises(_core.ClosedResourceError):
                    await s.send_all(b'xyzzy')
        async with _ForceCloseBoth(await clogged_stream_maker()) as (s, r):
            async with _core.open_nursery() as nursery:
                nursery.start_soon(close_soon, s)
                with _assert_raises(_core.ClosedResourceError):
                    await s.wait_send_all_might_not_block()

async def check_two_way_stream(stream_maker: StreamMaker[Stream, Stream], clogged_stream_maker: StreamMaker[Stream, Stream] | None) -> None:
    """Perform a number of generic tests on a custom two-way stream
    implementation.

    This is similar to :func:`check_one_way_stream`, except that the maker
    functions are expected to return objects implementing the
    :class:`~trio.abc.Stream` interface.

    This function tests a *superset* of what :func:`check_one_way_stream`
    checks – if you call this, then you don't need to also call
    :func:`check_one_way_stream`.

    """
    await check_one_way_stream(stream_maker, clogged_stream_maker)

    async def flipped_stream_maker() -> tuple[Stream, Stream]:
        return (await stream_maker())[::-1]
    flipped_clogged_stream_maker: Callable[[], Awaitable[tuple[Stream, Stream]]] | None
    if clogged_stream_maker is not None:

        async def flipped_clogged_stream_maker() -> tuple[Stream, Stream]:
            return (await clogged_stream_maker())[::-1]
    else:
        flipped_clogged_stream_maker = None
    await check_one_way_stream(flipped_stream_maker, flipped_clogged_stream_maker)
    async with _ForceCloseBoth(await stream_maker()) as (s1, s2):
        assert isinstance(s1, Stream)
        assert isinstance(s2, Stream)
        DUPLEX_TEST_SIZE = 2 ** 20
        CHUNK_SIZE_MAX = 2 ** 14
        r = random.Random(0)
        i = r.getrandbits(8 * DUPLEX_TEST_SIZE)
        test_data = i.to_bytes(DUPLEX_TEST_SIZE, 'little')

        async def sender(s: Stream, data: bytes | bytearray | memoryview, seed: int) -> None:
            r = random.Random(seed)
            m = memoryview(data)
            while m:
                chunk_size = r.randint(1, CHUNK_SIZE_MAX)
                await s.send_all(m[:chunk_size])
                m = m[chunk_size:]

        async def receiver(s: Stream, data: bytes | bytearray, seed: int) -> None:
            r = random.Random(seed)
            got = bytearray()
            while len(got) < len(data):
                chunk = await s.receive_some(r.randint(1, CHUNK_SIZE_MAX))
                assert chunk
                got += chunk
            assert got == data
        async with _core.open_nursery() as nursery:
            nursery.start_soon(sender, s1, test_data, 0)
            nursery.start_soon(sender, s2, test_data[::-1], 1)
            nursery.start_soon(receiver, s1, test_data[::-1], 2)
            nursery.start_soon(receiver, s2, test_data, 3)

        async def expect_receive_some_empty() -> None:
            assert await s2.receive_some(10) == b''
            await s2.aclose()
        async with _core.open_nursery() as nursery:
            nursery.start_soon(expect_receive_some_empty)
            nursery.start_soon(s1.aclose)

async def check_half_closeable_stream(stream_maker: StreamMaker[HalfCloseableStream, HalfCloseableStream], clogged_stream_maker: StreamMaker[HalfCloseableStream, HalfCloseableStream] | None) -> None:
    """Perform a number of generic tests on a custom half-closeable stream
    implementation.

    This is similar to :func:`check_two_way_stream`, except that the maker
    functions are expected to return objects that implement the
    :class:`~trio.abc.HalfCloseableStream` interface.

    This function tests a *superset* of what :func:`check_two_way_stream`
    checks – if you call this, then you don't need to also call
    :func:`check_two_way_stream`.

    """
    await check_two_way_stream(stream_maker, clogged_stream_maker)
    async with _ForceCloseBoth(await stream_maker()) as (s1, s2):
        assert isinstance(s1, HalfCloseableStream)
        assert isinstance(s2, HalfCloseableStream)

        async def send_x_then_eof(s: HalfCloseableStream) -> None:
            await s.send_all(b'x')
            with assert_checkpoints():
                await s.send_eof()

        async def expect_x_then_eof(r: HalfCloseableStream) -> None:
            await _core.wait_all_tasks_blocked()
            assert await r.receive_some(10) == b'x'
            assert await r.receive_some(10) == b''
        async with _core.open_nursery() as nursery:
            nursery.start_soon(send_x_then_eof, s1)
            nursery.start_soon(expect_x_then_eof, s2)
        with _assert_raises(_core.ClosedResourceError):
            await s1.send_all(b'y')
        with assert_checkpoints():
            await s1.send_eof()
        async with _core.open_nursery() as nursery:
            nursery.start_soon(send_x_then_eof, s2)
            nursery.start_soon(expect_x_then_eof, s1)
    if clogged_stream_maker is not None:
        async with _ForceCloseBoth(await clogged_stream_maker()) as (s1, s2):
            with _assert_raises(_core.BusyResourceError):
                async with _core.open_nursery() as nursery:
                    nursery.start_soon(s1.send_all, b'x')
                    await _core.wait_all_tasks_blocked()
                    nursery.start_soon(s1.send_eof)
        async with _ForceCloseBoth(await clogged_stream_maker()) as (s1, s2):
            with _assert_raises(_core.BusyResourceError):
                async with _core.open_nursery() as nursery:
                    nursery.start_soon(s1.wait_send_all_might_not_block)
                    await _core.wait_all_tasks_blocked()
                    nursery.start_soon(s1.send_eof)