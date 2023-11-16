import threading
from typing import Any, List, TypeVar
from reactivex import abc, typing
from reactivex.disposable import SerialDisposable
from .observer import Observer
_T_in = TypeVar('_T_in', contravariant=True)

class ScheduledObserver(Observer[_T_in]):

    def __init__(self, scheduler: abc.SchedulerBase, observer: abc.ObserverBase[_T_in]) -> None:
        if False:
            return 10
        super().__init__()
        self.scheduler = scheduler
        self.observer = observer
        self.lock = threading.RLock()
        self.is_acquired = False
        self.has_faulted = False
        self.queue: List[typing.Action] = []
        self.disposable = SerialDisposable()

    def _on_next_core(self, value: Any) -> None:
        if False:
            while True:
                i = 10

        def action() -> None:
            if False:
                i = 10
                return i + 15
            self.observer.on_next(value)
        self.queue.append(action)

    def _on_error_core(self, error: Exception) -> None:
        if False:
            for i in range(10):
                print('nop')

        def action() -> None:
            if False:
                for i in range(10):
                    print('nop')
            self.observer.on_error(error)
        self.queue.append(action)

    def _on_completed_core(self) -> None:
        if False:
            while True:
                i = 10

        def action() -> None:
            if False:
                for i in range(10):
                    print('nop')
            self.observer.on_completed()
        self.queue.append(action)

    def ensure_active(self) -> None:
        if False:
            i = 10
            return i + 15
        is_owner = False
        with self.lock:
            if not self.has_faulted and self.queue:
                is_owner = not self.is_acquired
                self.is_acquired = True
        if is_owner:
            self.disposable.disposable = self.scheduler.schedule(self.run)

    def run(self, scheduler: abc.SchedulerBase, state: Any) -> None:
        if False:
            i = 10
            return i + 15
        parent = self
        with self.lock:
            if parent.queue:
                work = parent.queue.pop(0)
            else:
                parent.is_acquired = False
                return
        try:
            work()
        except Exception:
            with self.lock:
                parent.queue = []
                parent.has_faulted = True
            raise
        self.scheduler.schedule(self.run)

    def dispose(self) -> None:
        if False:
            print('Hello World!')
        super().dispose()
        self.disposable.dispose()