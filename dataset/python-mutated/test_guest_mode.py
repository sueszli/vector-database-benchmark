from __future__ import annotations
import asyncio
import contextlib
import contextvars
import queue
import signal
import socket
import sys
import threading
import time
import traceback
import warnings
from functools import partial
from math import inf
from typing import TYPE_CHECKING, Any, AsyncGenerator, Awaitable, Callable, NoReturn, TypeVar
import pytest
from outcome import Outcome
from pytest import MonkeyPatch, WarningsRecorder
import trio
import trio.testing
from trio.abc import Instrument
from ..._util import signal_raise
from .tutil import buggy_pypy_asyncgens, gc_collect_harder, restore_unraisablehook
if TYPE_CHECKING:
    from typing_extensions import TypeAlias
    from trio._channel import MemorySendChannel
T = TypeVar('T')
InHost: TypeAlias = Callable[[object], None]

def trivial_guest_run(trio_fn: Callable[..., Awaitable[T]], *, in_host_after_start: Callable[[], None] | None=None, **start_guest_run_kwargs: Any) -> T:
    if False:
        return 10
    todo: queue.Queue[tuple[str, Outcome[T] | Callable[..., object]]] = queue.Queue()
    host_thread = threading.current_thread()

    def run_sync_soon_threadsafe(fn: Callable[[], object]) -> None:
        if False:
            for i in range(10):
                print('nop')
        nonlocal todo
        if host_thread is threading.current_thread():
            crash = partial(pytest.fail, 'run_sync_soon_threadsafe called from host thread')
            todo.put(('run', crash))
        todo.put(('run', fn))

    def run_sync_soon_not_threadsafe(fn: Callable[[], object]) -> None:
        if False:
            return 10
        nonlocal todo
        if host_thread is not threading.current_thread():
            crash = partial(pytest.fail, 'run_sync_soon_not_threadsafe called from worker thread')
            todo.put(('run', crash))
        todo.put(('run', fn))

    def done_callback(outcome: Outcome[T]) -> None:
        if False:
            return 10
        nonlocal todo
        todo.put(('unwrap', outcome))
    trio.lowlevel.start_guest_run(trio_fn, run_sync_soon_not_threadsafe, run_sync_soon_threadsafe=run_sync_soon_threadsafe, run_sync_soon_not_threadsafe=run_sync_soon_not_threadsafe, done_callback=done_callback, **start_guest_run_kwargs)
    if in_host_after_start is not None:
        in_host_after_start()
    try:
        while True:
            (op, obj) = todo.get()
            if op == 'run':
                assert not isinstance(obj, Outcome)
                obj()
            elif op == 'unwrap':
                assert isinstance(obj, Outcome)
                return obj.unwrap()
            else:
                raise NotImplementedError(f'{op!r} not handled')
    finally:
        del todo, run_sync_soon_threadsafe, done_callback

def test_guest_trivial() -> None:
    if False:
        print('Hello World!')

    async def trio_return(in_host: InHost) -> str:
        await trio.sleep(0)
        return 'ok'
    assert trivial_guest_run(trio_return) == 'ok'

    async def trio_fail(in_host: InHost) -> NoReturn:
        raise KeyError('whoopsiedaisy')
    with pytest.raises(KeyError, match='whoopsiedaisy'):
        trivial_guest_run(trio_fail)

def test_guest_can_do_io() -> None:
    if False:
        for i in range(10):
            print('nop')

    async def trio_main(in_host: InHost) -> None:
        record = []
        (a, b) = trio.socket.socketpair()
        with a, b:
            async with trio.open_nursery() as nursery:

                async def do_receive() -> None:
                    record.append(await a.recv(1))
                nursery.start_soon(do_receive)
                await trio.testing.wait_all_tasks_blocked()
                await b.send(b'x')
        assert record == [b'x']
    trivial_guest_run(trio_main)

def test_guest_is_initialized_when_start_returns() -> None:
    if False:
        return 10
    trio_token = None
    record = []

    async def trio_main(in_host: InHost) -> str:
        record.append('main task ran')
        await trio.sleep(0)
        assert trio.lowlevel.current_trio_token() is trio_token
        return 'ok'

    def after_start() -> None:
        if False:
            return 10
        assert record == []
        nonlocal trio_token
        trio_token = trio.lowlevel.current_trio_token()
        trio_token.run_sync_soon(record.append, 'run_sync_soon cb ran')

        @trio.lowlevel.spawn_system_task
        async def early_task() -> None:
            record.append('system task ran')
            await trio.sleep(0)
    res = trivial_guest_run(trio_main, in_host_after_start=after_start)
    assert res == 'ok'
    assert set(record) == {'system task ran', 'main task ran', 'run_sync_soon cb ran'}
    with pytest.raises(trio.TrioInternalError):

        class BadClock:

            def start_clock(self) -> NoReturn:
                if False:
                    print('Hello World!')
                raise ValueError('whoops')

        def after_start_never_runs() -> None:
            if False:
                print('Hello World!')
            pytest.fail("shouldn't get here")
        trivial_guest_run(trio_main, clock=BadClock(), in_host_after_start=after_start_never_runs)

def test_host_can_directly_wake_trio_task() -> None:
    if False:
        return 10

    async def trio_main(in_host: InHost) -> str:
        ev = trio.Event()
        in_host(ev.set)
        await ev.wait()
        return 'ok'
    assert trivial_guest_run(trio_main) == 'ok'

def test_host_altering_deadlines_wakes_trio_up() -> None:
    if False:
        print('Hello World!')

    def set_deadline(cscope: trio.CancelScope, new_deadline: float) -> None:
        if False:
            while True:
                i = 10
        cscope.deadline = new_deadline

    async def trio_main(in_host: InHost) -> str:
        with trio.CancelScope() as cscope:
            in_host(lambda : set_deadline(cscope, -inf))
            await trio.sleep_forever()
        assert cscope.cancelled_caught
        with trio.CancelScope() as cscope:
            in_host(lambda : set_deadline(cscope, 1000000.0))
            in_host(lambda : set_deadline(cscope, -inf))
            await trio.sleep(999)
        assert cscope.cancelled_caught
        return 'ok'
    assert trivial_guest_run(trio_main) == 'ok'

def test_guest_mode_sniffio_integration() -> None:
    if False:
        print('Hello World!')
    from sniffio import current_async_library, thread_local as sniffio_library

    async def trio_main(in_host: InHost) -> str:

        async def synchronize() -> None:
            """Wait for all in_host() calls issued so far to complete."""
            evt = trio.Event()
            in_host(evt.set)
            await evt.wait()
        in_host(partial(setattr, sniffio_library, 'name', 'nullio'))
        await synchronize()
        assert current_async_library() == 'trio'
        record = []
        in_host(lambda : record.append(current_async_library()))
        await synchronize()
        assert record == ['nullio']
        assert current_async_library() == 'trio'
        return 'ok'
    try:
        assert trivial_guest_run(trio_main) == 'ok'
    finally:
        sniffio_library.name = None

def test_warn_set_wakeup_fd_overwrite() -> None:
    if False:
        return 10
    assert signal.set_wakeup_fd(-1) == -1

    async def trio_main(in_host: InHost) -> str:
        return 'ok'
    (a, b) = socket.socketpair()
    with a, b:
        a.setblocking(False)
        signal.set_wakeup_fd(a.fileno())
        try:
            with pytest.warns(RuntimeWarning, match='signal handling code.*collided'):
                assert trivial_guest_run(trio_main) == 'ok'
        finally:
            assert signal.set_wakeup_fd(-1) == a.fileno()
        signal.set_wakeup_fd(a.fileno())
        try:
            with pytest.warns(RuntimeWarning, match='signal handling code.*collided'):
                assert trivial_guest_run(trio_main, host_uses_signal_set_wakeup_fd=False) == 'ok'
        finally:
            assert signal.set_wakeup_fd(-1) == a.fileno()
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            assert trivial_guest_run(trio_main) == 'ok'
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            assert trivial_guest_run(trio_main, host_uses_signal_set_wakeup_fd=True) == 'ok'
        signal.set_wakeup_fd(a.fileno())
        try:

            async def trio_check_wakeup_fd_unaltered(in_host: InHost) -> str:
                fd = signal.set_wakeup_fd(-1)
                assert fd == a.fileno()
                signal.set_wakeup_fd(fd)
                return 'ok'
            with warnings.catch_warnings():
                warnings.simplefilter('error')
                assert trivial_guest_run(trio_check_wakeup_fd_unaltered, host_uses_signal_set_wakeup_fd=True) == 'ok'
        finally:
            assert signal.set_wakeup_fd(-1) == a.fileno()

def test_host_wakeup_doesnt_trigger_wait_all_tasks_blocked() -> None:
    if False:
        return 10

    def set_deadline(cscope: trio.CancelScope, new_deadline: float) -> None:
        if False:
            return 10
        print(f'setting deadline {new_deadline}')
        cscope.deadline = new_deadline

    async def trio_main(in_host: InHost) -> str:

        async def sit_in_wait_all_tasks_blocked(watb_cscope: trio.CancelScope) -> None:
            with watb_cscope:
                await trio.testing.wait_all_tasks_blocked(cushion=9999)
                raise AssertionError('wait_all_tasks_blocked should *not* return normally, only by cancellation.')
            assert watb_cscope.cancelled_caught

        async def get_woken_by_host_deadline(watb_cscope: trio.CancelScope) -> None:
            with trio.CancelScope() as cscope:
                print('scheduling stuff to happen')

                class InstrumentHelper(Instrument):

                    def __init__(self) -> None:
                        if False:
                            return 10
                        self.primed = False

                    def before_io_wait(self, timeout: float) -> None:
                        if False:
                            i = 10
                            return i + 15
                        print(f'before_io_wait({timeout})')
                        if timeout == 9999:
                            assert not self.primed
                            in_host(lambda : set_deadline(cscope, 1000000000.0))
                            self.primed = True

                    def after_io_wait(self, timeout: float) -> None:
                        if False:
                            while True:
                                i = 10
                        if self.primed:
                            print('instrument triggered')
                            in_host(lambda : cscope.cancel())
                            trio.lowlevel.remove_instrument(self)
                trio.lowlevel.add_instrument(InstrumentHelper())
                await trio.sleep_forever()
            assert cscope.cancelled_caught
            watb_cscope.cancel()
        async with trio.open_nursery() as nursery:
            watb_cscope = trio.CancelScope()
            nursery.start_soon(sit_in_wait_all_tasks_blocked, watb_cscope)
            await trio.testing.wait_all_tasks_blocked()
            nursery.start_soon(get_woken_by_host_deadline, watb_cscope)
        return 'ok'
    assert trivial_guest_run(trio_main) == 'ok'

@restore_unraisablehook()
def test_guest_warns_if_abandoned() -> None:
    if False:
        print('Hello World!')

    def do_abandoned_guest_run() -> None:
        if False:
            print('Hello World!')

        async def abandoned_main(in_host: InHost) -> None:
            in_host(lambda : 1 / 0)
            while True:
                await trio.sleep(0)
        with pytest.raises(ZeroDivisionError):
            trivial_guest_run(abandoned_main)
    with pytest.warns(RuntimeWarning, match='Trio guest run got abandoned'):
        do_abandoned_guest_run()
        gc_collect_harder()
        with pytest.raises(RuntimeError):
            trio.current_time()

def aiotrio_run(trio_fn: Callable[..., Awaitable[T]], *, pass_not_threadsafe: bool=True, **start_guest_run_kwargs: Any) -> T:
    if False:
        return 10
    loop = asyncio.new_event_loop()

    async def aio_main() -> T:
        trio_done_fut = loop.create_future()

        def trio_done_callback(main_outcome: Outcome[object]) -> None:
            if False:
                for i in range(10):
                    print('nop')
            print(f'trio_fn finished: {main_outcome!r}')
            trio_done_fut.set_result(main_outcome)
        if pass_not_threadsafe:
            start_guest_run_kwargs['run_sync_soon_not_threadsafe'] = loop.call_soon
        trio.lowlevel.start_guest_run(trio_fn, run_sync_soon_threadsafe=loop.call_soon_threadsafe, done_callback=trio_done_callback, **start_guest_run_kwargs)
        return (await trio_done_fut).unwrap()
    try:
        return loop.run_until_complete(aio_main())
    finally:
        loop.close()

def test_guest_mode_on_asyncio() -> None:
    if False:
        return 10

    async def trio_main() -> str:
        print('trio_main!')
        (to_trio, from_aio) = trio.open_memory_channel[int](float('inf'))
        from_trio: asyncio.Queue[int] = asyncio.Queue()
        aio_task = asyncio.ensure_future(aio_pingpong(from_trio, to_trio))
        await trio.sleep(0)
        from_trio.put_nowait(0)
        async for n in from_aio:
            print(f'trio got: {n}')
            from_trio.put_nowait(n + 1)
            if n >= 10:
                aio_task.cancel()
                return 'trio-main-done'
        raise AssertionError('should never be reached')

    async def aio_pingpong(from_trio: asyncio.Queue[int], to_trio: MemorySendChannel[int]) -> None:
        print('aio_pingpong!')
        try:
            while True:
                n = await from_trio.get()
                print(f'aio got: {n}')
                to_trio.send_nowait(n + 1)
        except asyncio.CancelledError:
            raise
        except:
            traceback.print_exc()
            raise
    assert aiotrio_run(trio_main, host_uses_signal_set_wakeup_fd=True) == 'trio-main-done'
    assert aiotrio_run(trio_main, pass_not_threadsafe=False, host_uses_signal_set_wakeup_fd=True) == 'trio-main-done'

def test_guest_mode_internal_errors(monkeypatch: MonkeyPatch, recwarn: WarningsRecorder) -> None:
    if False:
        i = 10
        return i + 15
    with monkeypatch.context() as m:

        async def crash_in_run_loop(in_host: InHost) -> None:
            m.setattr('trio._core._run.GLOBAL_RUN_CONTEXT.runner.runq', 'HI')
            await trio.sleep(1)
        with pytest.raises(trio.TrioInternalError):
            trivial_guest_run(crash_in_run_loop)
    with monkeypatch.context() as m:

        async def crash_in_io(in_host: InHost) -> None:
            m.setattr('trio._core._run.TheIOManager.get_events', None)
            await trio.sleep(0)
        with pytest.raises(trio.TrioInternalError):
            trivial_guest_run(crash_in_io)
    with monkeypatch.context() as m:

        async def crash_in_worker_thread_io(in_host: InHost) -> None:
            t = threading.current_thread()
            old_get_events = trio._core._run.TheIOManager.get_events

            def bad_get_events(*args: Any) -> object:
                if False:
                    print('Hello World!')
                if threading.current_thread() is not t:
                    raise ValueError('oh no!')
                else:
                    return old_get_events(*args)
            m.setattr('trio._core._run.TheIOManager.get_events', bad_get_events)
            await trio.sleep(1)
        with pytest.raises(trio.TrioInternalError):
            trivial_guest_run(crash_in_worker_thread_io)
    gc_collect_harder()

def test_guest_mode_ki() -> None:
    if False:
        i = 10
        return i + 15
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler

    async def trio_main(in_host: InHost) -> None:
        with pytest.raises(KeyboardInterrupt):
            signal_raise(signal.SIGINT)
        in_host(partial(signal_raise, signal.SIGINT))
        await trio.sleep(10)
    with pytest.raises(KeyboardInterrupt) as excinfo:
        trivial_guest_run(trio_main)
    assert excinfo.value.__context__ is None
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler
    final_exc = KeyError('whoa')

    async def trio_main_raising(in_host: InHost) -> NoReturn:
        in_host(partial(signal_raise, signal.SIGINT))
        raise final_exc
    with pytest.raises(KeyboardInterrupt) as excinfo:
        trivial_guest_run(trio_main_raising)
    assert excinfo.value.__context__ is final_exc
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler

def test_guest_mode_autojump_clock_threshold_changing() -> None:
    if False:
        while True:
            i = 10
    clock = trio.testing.MockClock()
    DURATION = 120

    async def trio_main(in_host: InHost) -> None:
        assert trio.current_time() == 0
        in_host(lambda : setattr(clock, 'autojump_threshold', 0))
        await trio.sleep(DURATION)
        assert trio.current_time() == DURATION
    start = time.monotonic()
    trivial_guest_run(trio_main, clock=clock)
    end = time.monotonic()
    assert end - start < DURATION / 2

@pytest.mark.skipif(buggy_pypy_asyncgens, reason='PyPy 7.2 is buggy')
@restore_unraisablehook()
def test_guest_mode_asyncgens() -> None:
    if False:
        i = 10
        return i + 15
    import sniffio
    record = set()

    async def agen(label: str) -> AsyncGenerator[int, None]:
        assert sniffio.current_async_library() == label
        try:
            yield 1
        finally:
            library = sniffio.current_async_library()
            with contextlib.suppress(trio.Cancelled):
                await sys.modules[library].sleep(0)
            record.add((label, library))

    async def iterate_in_aio() -> None:
        await agen('asyncio').asend(None)

    async def trio_main() -> None:
        task = asyncio.ensure_future(iterate_in_aio())
        done_evt = trio.Event()
        task.add_done_callback(lambda _: done_evt.set())
        with trio.fail_after(1):
            await done_evt.wait()
        await agen('trio').asend(None)
        gc_collect_harder()
    context = contextvars.copy_context()
    if TYPE_CHECKING:
        aiotrio_run(trio_main, host_uses_signal_set_wakeup_fd=True)
    context.run(aiotrio_run, trio_main, host_uses_signal_set_wakeup_fd=True)
    assert record == {('asyncio', 'asyncio'), ('trio', 'trio')}