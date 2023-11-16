"""Run a target function on a background thread."""
from __future__ import annotations
import threading
import time
import weakref
from typing import Any, Callable, Optional
from pymongo.lock import _create_lock

class PeriodicExecutor:

    def __init__(self, interval: float, min_interval: float, target: Callable[[], bool], name: Optional[str]=None):
        if False:
            print('Hello World!')
        ' "Run a target function periodically on a background thread.\n\n        If the target\'s return value is false, the executor stops.\n\n        :Parameters:\n          - `interval`: Seconds between calls to `target`.\n          - `min_interval`: Minimum seconds between calls if `wake` is\n            called very often.\n          - `target`: A function.\n          - `name`: A name to give the underlying thread.\n        '
        self._event = False
        self._interval = interval
        self._min_interval = min_interval
        self._target = target
        self._stopped = False
        self._thread: Optional[threading.Thread] = None
        self._name = name
        self._skip_sleep = False
        self._thread_will_exit = False
        self._lock = _create_lock()

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        return f'<{self.__class__.__name__}(name={self._name}) object at 0x{id(self):x}>'

    def open(self) -> None:
        if False:
            i = 10
            return i + 15
        'Start. Multiple calls have no effect.\n\n        Not safe to call from multiple threads at once.\n        '
        with self._lock:
            if self._thread_will_exit:
                try:
                    assert self._thread is not None
                    self._thread.join()
                except ReferenceError:
                    pass
            self._thread_will_exit = False
            self._stopped = False
        started: Any = False
        try:
            started = self._thread and self._thread.is_alive()
        except ReferenceError:
            pass
        if not started:
            thread = threading.Thread(target=self._run, name=self._name)
            thread.daemon = True
            self._thread = weakref.proxy(thread)
            _register_executor(self)
            thread.start()

    def close(self, dummy: Any=None) -> None:
        if False:
            while True:
                i = 10
        "Stop. To restart, call open().\n\n        The dummy parameter allows an executor's close method to be a weakref\n        callback; see monitor.py.\n        "
        self._stopped = True

    def join(self, timeout: Optional[int]=None) -> None:
        if False:
            while True:
                i = 10
        if self._thread is not None:
            try:
                self._thread.join(timeout)
            except (ReferenceError, RuntimeError):
                pass

    def wake(self) -> None:
        if False:
            i = 10
            return i + 15
        'Execute the target function soon.'
        self._event = True

    def update_interval(self, new_interval: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._interval = new_interval

    def skip_sleep(self) -> None:
        if False:
            while True:
                i = 10
        self._skip_sleep = True

    def __should_stop(self) -> bool:
        if False:
            print('Hello World!')
        with self._lock:
            if self._stopped:
                self._thread_will_exit = True
                return True
            return False

    def _run(self) -> None:
        if False:
            i = 10
            return i + 15
        while not self.__should_stop():
            try:
                if not self._target():
                    self._stopped = True
                    break
            except BaseException:
                with self._lock:
                    self._stopped = True
                    self._thread_will_exit = True
                raise
            if self._skip_sleep:
                self._skip_sleep = False
            else:
                deadline = time.monotonic() + self._interval
                while not self._stopped and time.monotonic() < deadline:
                    time.sleep(self._min_interval)
                    if self._event:
                        break
            self._event = False
_EXECUTORS = set()

def _register_executor(executor: PeriodicExecutor) -> None:
    if False:
        for i in range(10):
            print('nop')
    ref = weakref.ref(executor, _on_executor_deleted)
    _EXECUTORS.add(ref)

def _on_executor_deleted(ref: weakref.ReferenceType[PeriodicExecutor]) -> None:
    if False:
        print('Hello World!')
    _EXECUTORS.remove(ref)

def _shutdown_executors() -> None:
    if False:
        i = 10
        return i + 15
    if _EXECUTORS is None:
        return
    executors = list(_EXECUTORS)
    for ref in executors:
        executor = ref()
        if executor:
            executor.close()
    for ref in executors:
        executor = ref()
        if executor:
            executor.join(1)
    executor = None