from __future__ import annotations
import contextlib
import inspect
import signal
import threading
from typing import TYPE_CHECKING, AsyncIterator, Callable, Iterator
import outcome
import pytest
try:
    from async_generator import async_generator, yield_
except ImportError:
    async_generator = yield_ = None
from ... import _core
from ..._abc import Instrument
from ..._timeouts import sleep
from ..._util import signal_raise
from ...testing import wait_all_tasks_blocked
if TYPE_CHECKING:
    from ..._core import Abort, RaiseCancelT

def ki_self() -> None:
    if False:
        while True:
            i = 10
    signal_raise(signal.SIGINT)

def test_ki_self() -> None:
    if False:
        return 10
    with pytest.raises(KeyboardInterrupt):
        ki_self()

async def test_ki_enabled() -> None:
    assert not _core.currently_ki_protected()
    token = _core.current_trio_token()
    record = []

    def check() -> None:
        if False:
            while True:
                i = 10
        record.append(_core.currently_ki_protected())
    token.run_sync_soon(check)
    await wait_all_tasks_blocked()
    assert record == [True]

    @_core.enable_ki_protection
    def protected() -> None:
        if False:
            print('Hello World!')
        assert _core.currently_ki_protected()
        unprotected()

    @_core.disable_ki_protection
    def unprotected() -> None:
        if False:
            while True:
                i = 10
        assert not _core.currently_ki_protected()
    protected()

    @_core.enable_ki_protection
    async def aprotected() -> None:
        assert _core.currently_ki_protected()
        await aunprotected()

    @_core.disable_ki_protection
    async def aunprotected() -> None:
        assert not _core.currently_ki_protected()
    await aprotected()
    async with _core.open_nursery() as nursery:
        nursery.start_soon(aprotected)
        nursery.start_soon(aunprotected)

    @_core.enable_ki_protection
    def gen_protected() -> Iterator[None]:
        if False:
            while True:
                i = 10
        assert _core.currently_ki_protected()
        yield
    for _ in gen_protected():
        pass

    @_core.disable_ki_protection
    def gen_unprotected() -> Iterator[None]:
        if False:
            while True:
                i = 10
        assert not _core.currently_ki_protected()
        yield
    for _ in gen_unprotected():
        pass

async def test_ki_enabled_after_yield_briefly() -> None:

    @_core.enable_ki_protection
    async def protected() -> None:
        await child(True)

    @_core.disable_ki_protection
    async def unprotected() -> None:
        await child(False)

    async def child(expected: bool) -> None:
        import traceback
        traceback.print_stack()
        assert _core.currently_ki_protected() == expected
        await _core.checkpoint()
        traceback.print_stack()
        assert _core.currently_ki_protected() == expected
    await protected()
    await unprotected()

async def test_generator_based_context_manager_throw() -> None:

    @contextlib.contextmanager
    @_core.enable_ki_protection
    def protected_manager() -> Iterator[None]:
        if False:
            for i in range(10):
                print('nop')
        assert _core.currently_ki_protected()
        try:
            yield
        finally:
            assert _core.currently_ki_protected()
    with protected_manager():
        assert not _core.currently_ki_protected()
    with pytest.raises(KeyError):
        with protected_manager():
            raise KeyError

@pytest.mark.skipif(async_generator is None, reason='async_generator not installed')
async def test_async_generator_agen_protection() -> None:

    @_core.enable_ki_protection
    @async_generator
    async def agen_protected1() -> None:
        assert _core.currently_ki_protected()
        try:
            await yield_()
        finally:
            assert _core.currently_ki_protected()

    @_core.disable_ki_protection
    @async_generator
    async def agen_unprotected1() -> None:
        assert not _core.currently_ki_protected()
        try:
            await yield_()
        finally:
            assert not _core.currently_ki_protected()

    @async_generator
    @_core.enable_ki_protection
    async def agen_protected2() -> None:
        assert _core.currently_ki_protected()
        try:
            await yield_()
        finally:
            assert _core.currently_ki_protected()

    @async_generator
    @_core.disable_ki_protection
    async def agen_unprotected2() -> None:
        assert not _core.currently_ki_protected()
        try:
            await yield_()
        finally:
            assert not _core.currently_ki_protected()
    await _check_agen(agen_protected1)
    await _check_agen(agen_protected2)
    await _check_agen(agen_unprotected1)
    await _check_agen(agen_unprotected2)

async def test_native_agen_protection() -> None:

    @_core.enable_ki_protection
    async def agen_protected() -> AsyncIterator[None]:
        assert _core.currently_ki_protected()
        try:
            yield
        finally:
            assert _core.currently_ki_protected()

    @_core.disable_ki_protection
    async def agen_unprotected() -> AsyncIterator[None]:
        assert not _core.currently_ki_protected()
        try:
            yield
        finally:
            assert not _core.currently_ki_protected()
    await _check_agen(agen_protected)
    await _check_agen(agen_unprotected)

async def _check_agen(agen_fn: Callable[[], AsyncIterator[None]]) -> None:
    async for _ in agen_fn():
        assert not _core.currently_ki_protected()
    if inspect.isasyncgenfunction(agen_fn):
        async with contextlib.asynccontextmanager(agen_fn)():
            assert not _core.currently_ki_protected()
        with pytest.raises(KeyError):
            async with contextlib.asynccontextmanager(agen_fn)():
                raise KeyError

def test_ki_disabled_out_of_context() -> None:
    if False:
        return 10
    assert _core.currently_ki_protected()

def test_ki_disabled_in_del() -> None:
    if False:
        return 10

    def nestedfunction() -> bool:
        if False:
            for i in range(10):
                print('nop')
        return _core.currently_ki_protected()

    def __del__() -> None:
        if False:
            for i in range(10):
                print('nop')
        assert _core.currently_ki_protected()
        assert nestedfunction()

    @_core.disable_ki_protection
    def outerfunction() -> None:
        if False:
            i = 10
            return i + 15
        assert not _core.currently_ki_protected()
        assert not nestedfunction()
        __del__()
    __del__()
    outerfunction()
    assert nestedfunction()

def test_ki_protection_works() -> None:
    if False:
        print('Hello World!')

    async def sleeper(name: str, record: set[str]) -> None:
        try:
            while True:
                await _core.checkpoint()
        except _core.Cancelled:
            record.add(name + ' ok')

    async def raiser(name: str, record: set[str]) -> None:
        try:
            print('killing, protection =', _core.currently_ki_protected())
            ki_self()
        except KeyboardInterrupt:
            print('raised!')
            await _core.checkpoint()
            record.add(name + ' raise ok')
            raise
        else:
            print("didn't raise!")
            try:
                await _core.wait_task_rescheduled(lambda _: _core.Abort.SUCCEEDED)
            except _core.Cancelled:
                record.add(name + ' cancel ok')
    print('check 1')
    record_set: set[str] = set()

    async def check_unprotected_kill() -> None:
        async with _core.open_nursery() as nursery:
            nursery.start_soon(sleeper, 's1', record_set)
            nursery.start_soon(sleeper, 's2', record_set)
            nursery.start_soon(raiser, 'r1', record_set)
    with pytest.raises(KeyboardInterrupt):
        _core.run(check_unprotected_kill)
    assert record_set == {'s1 ok', 's2 ok', 'r1 raise ok'}
    print('check 2')
    record_set = set()

    async def check_protected_kill() -> None:
        async with _core.open_nursery() as nursery:
            nursery.start_soon(sleeper, 's1', record_set)
            nursery.start_soon(sleeper, 's2', record_set)
            nursery.start_soon(_core.enable_ki_protection(raiser), 'r1', record_set)
    with pytest.raises(KeyboardInterrupt):
        _core.run(check_protected_kill)
    assert record_set == {'s1 ok', 's2 ok', 'r1 cancel ok'}
    print('check 3')

    async def check_kill_during_shutdown() -> None:
        token = _core.current_trio_token()

        def kill_during_shutdown() -> None:
            if False:
                for i in range(10):
                    print('nop')
            assert _core.currently_ki_protected()
            try:
                token.run_sync_soon(kill_during_shutdown)
            except _core.RunFinishedError:
                print('kill! kill!')
                ki_self()
        token.run_sync_soon(kill_during_shutdown)
    with pytest.raises(KeyboardInterrupt):
        _core.run(check_kill_during_shutdown)
    print('check 4')

    class InstrumentOfDeath(Instrument):

        def before_run(self) -> None:
            if False:
                for i in range(10):
                    print('nop')
            ki_self()

    async def main_1() -> None:
        await _core.checkpoint()
    with pytest.raises(KeyboardInterrupt):
        _core.run(main_1, instruments=[InstrumentOfDeath()])
    print('check 5')

    @_core.enable_ki_protection
    async def main_2() -> None:
        assert _core.currently_ki_protected()
        ki_self()
        with pytest.raises(KeyboardInterrupt):
            await _core.checkpoint_if_cancelled()
    _core.run(main_2)
    print('check 6')

    @_core.enable_ki_protection
    async def main_3() -> None:
        assert _core.currently_ki_protected()
        ki_self()
        await _core.cancel_shielded_checkpoint()
        await _core.cancel_shielded_checkpoint()
        await _core.cancel_shielded_checkpoint()
        with pytest.raises(KeyboardInterrupt):
            await _core.checkpoint()
    _core.run(main_3)
    print('check 7')

    @_core.enable_ki_protection
    async def main_4() -> None:
        assert _core.currently_ki_protected()
        ki_self()
        task = _core.current_task()

        def abort(_: RaiseCancelT) -> Abort:
            if False:
                return 10
            _core.reschedule(task, outcome.Value(1))
            return _core.Abort.FAILED
        assert await _core.wait_task_rescheduled(abort) == 1
        with pytest.raises(KeyboardInterrupt):
            await _core.checkpoint()
    _core.run(main_4)
    print('check 8')

    @_core.enable_ki_protection
    async def main_5() -> None:
        assert _core.currently_ki_protected()
        ki_self()
        task = _core.current_task()

        def abort(raise_cancel: RaiseCancelT) -> Abort:
            if False:
                return 10
            result = outcome.capture(raise_cancel)
            _core.reschedule(task, result)
            return _core.Abort.FAILED
        with pytest.raises(KeyboardInterrupt):
            assert await _core.wait_task_rescheduled(abort)
        await _core.checkpoint()
    _core.run(main_5)
    print('check 9')

    @_core.enable_ki_protection
    async def main_6() -> None:
        ki_self()
    with pytest.raises(KeyboardInterrupt):
        _core.run(main_6)
    print('check 10')
    record_list = []

    async def main_7() -> None:
        assert not _core.currently_ki_protected()
        ki_self()
        record_list.append('ok')
        with pytest.raises(KeyboardInterrupt):
            await sleep(10)
    _core.run(main_7, restrict_keyboard_interrupt_to_checkpoints=True)
    assert record_list == ['ok']
    record_list = []
    with pytest.raises(KeyboardInterrupt):
        _core.run(main_7)
    assert record_list == []
    print('check 11')

    @_core.enable_ki_protection
    async def main_8() -> None:
        assert _core.currently_ki_protected()
        with _core.CancelScope() as cancel_scope:
            cancel_scope.cancel()
            with pytest.raises(_core.Cancelled):
                await _core.checkpoint()
            ki_self()
            with pytest.raises(KeyboardInterrupt):
                await _core.checkpoint()
            with pytest.raises(_core.Cancelled):
                await _core.checkpoint()
    _core.run(main_8)

def test_ki_is_good_neighbor() -> None:
    if False:
        print('Hello World!')
    try:
        orig = signal.getsignal(signal.SIGINT)

        def my_handler(signum: object, frame: object) -> None:
            if False:
                i = 10
                return i + 15
            pass

        async def main() -> None:
            signal.signal(signal.SIGINT, my_handler)
        _core.run(main)
        assert signal.getsignal(signal.SIGINT) is my_handler
    finally:
        signal.signal(signal.SIGINT, orig)

def test_ki_with_broken_threads() -> None:
    if False:
        print('Hello World!')
    thread = threading.main_thread()
    original = threading._active[thread.ident]
    try:
        del threading._active[thread.ident]

        @_core.enable_ki_protection
        async def inner() -> None:
            assert signal.getsignal(signal.SIGINT) != signal.default_int_handler
        _core.run(inner)
    finally:
        threading._active[thread.ident] = original