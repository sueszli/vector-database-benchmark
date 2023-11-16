from __future__ import annotations
import threading
import time
from contextlib import contextmanager
from queue import Queue
from typing import TYPE_CHECKING, Iterator, NoReturn
import pytest
from pytest import MonkeyPatch
from .. import _thread_cache
from .._thread_cache import ThreadCache, start_thread_soon
from .tutil import gc_collect_harder, slow
if TYPE_CHECKING:
    from outcome import Outcome

def test_thread_cache_basics() -> None:
    if False:
        while True:
            i = 10
    q: Queue[Outcome[object]] = Queue()

    def fn() -> NoReturn:
        if False:
            i = 10
            return i + 15
        raise RuntimeError('hi')

    def deliver(outcome: Outcome[object]) -> None:
        if False:
            i = 10
            return i + 15
        q.put(outcome)
    start_thread_soon(fn, deliver)
    outcome = q.get()
    with pytest.raises(RuntimeError, match='hi'):
        outcome.unwrap()

def test_thread_cache_deref() -> None:
    if False:
        print('Hello World!')
    res = [False]

    class del_me:

        def __call__(self) -> int:
            if False:
                return 10
            return 42

        def __del__(self) -> None:
            if False:
                print('Hello World!')
            res[0] = True
    q: Queue[Outcome[int]] = Queue()

    def deliver(outcome: Outcome[int]) -> None:
        if False:
            print('Hello World!')
        q.put(outcome)
    start_thread_soon(del_me(), deliver)
    outcome = q.get()
    assert outcome.unwrap() == 42
    gc_collect_harder()
    assert res[0]

@slow
def test_spawning_new_thread_from_deliver_reuses_starting_thread() -> None:
    if False:
        while True:
            i = 10
    q: Queue[Outcome[object]] = Queue()
    COUNT = 5
    for _ in range(COUNT):
        start_thread_soon(lambda : time.sleep(1), lambda result: q.put(result))
    for _ in range(COUNT):
        q.get().unwrap()
    seen_threads = set()
    done = threading.Event()

    def deliver(n: int, _: object) -> None:
        if False:
            while True:
                i = 10
        print(n)
        seen_threads.add(threading.current_thread())
        if n == 0:
            done.set()
        else:
            start_thread_soon(lambda : None, lambda _: deliver(n - 1, _))
    start_thread_soon(lambda : None, lambda _: deliver(5, _))
    done.wait()
    assert len(seen_threads) == 1

@slow
def test_idle_threads_exit(monkeypatch: MonkeyPatch) -> None:
    if False:
        while True:
            i = 10
    monkeypatch.setattr(_thread_cache, 'IDLE_TIMEOUT', 0.0001)
    q: Queue[threading.Thread] = Queue()
    start_thread_soon(lambda : None, lambda _: q.put(threading.current_thread()))
    seen_thread = q.get()
    time.sleep(1)
    assert not seen_thread.is_alive()

@contextmanager
def _join_started_threads() -> Iterator[None]:
    if False:
        for i in range(10):
            print('nop')
    before = frozenset(threading.enumerate())
    try:
        yield
    finally:
        for thread in threading.enumerate():
            if thread not in before:
                thread.join(timeout=1.0)
                assert not thread.is_alive()

def test_race_between_idle_exit_and_job_assignment(monkeypatch: MonkeyPatch) -> None:
    if False:
        for i in range(10):
            print('nop')

    class JankyLock:

        def __init__(self) -> None:
            if False:
                for i in range(10):
                    print('nop')
            self._lock = threading.Lock()
            self._counter = 3

        def acquire(self, timeout: int=-1) -> bool:
            if False:
                print('Hello World!')
            got_it = self._lock.acquire(timeout=timeout)
            if timeout == -1:
                return True
            elif got_it:
                if self._counter > 0:
                    self._counter -= 1
                    self._lock.release()
                    return False
                return True
            else:
                return False

        def release(self) -> None:
            if False:
                for i in range(10):
                    print('nop')
            self._lock.release()
    monkeypatch.setattr(_thread_cache, 'Lock', JankyLock)
    with _join_started_threads():
        tc = ThreadCache()
        done = threading.Event()
        tc.start_thread_soon(lambda : None, lambda _: done.set())
        done.wait()
        monkeypatch.setattr(_thread_cache, 'IDLE_TIMEOUT', 0.0001)
        tc.start_thread_soon(lambda : None, lambda _: None)

def test_raise_in_deliver(capfd: pytest.CaptureFixture[str]) -> None:
    if False:
        while True:
            i = 10
    seen_threads = set()

    def track_threads() -> None:
        if False:
            print('Hello World!')
        seen_threads.add(threading.current_thread())

    def deliver(_: object) -> NoReturn:
        if False:
            for i in range(10):
                print('nop')
        done.set()
        raise RuntimeError("don't do this")
    done = threading.Event()
    start_thread_soon(track_threads, deliver)
    done.wait()
    done = threading.Event()
    start_thread_soon(track_threads, lambda _: done.set())
    done.wait()
    assert len(seen_threads) == 1
    err = capfd.readouterr().err
    assert "don't do this" in err
    assert 'delivering result' in err