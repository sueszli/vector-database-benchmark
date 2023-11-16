import threading
from threading import Thread, Event
from typing import Callable

class IntervalRunner:
    event: Event
    thread: Thread

    def __init__(self, target: Callable[[], None], interval_seconds: float=0.1):
        if False:
            return 10
        self.event = threading.Event()
        self.target = target
        self.interval_seconds = interval_seconds
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True

    def _run(self) -> None:
        if False:
            return 10
        while not self.event.is_set():
            self.target()
            self.event.wait(self.interval_seconds)

    def start(self) -> 'IntervalRunner':
        if False:
            print('Hello World!')
        self.thread.start()
        return self

    def is_alive(self) -> bool:
        if False:
            i = 10
            return i + 15
        return self.thread is not None and self.thread.is_alive()

    def shutdown(self):
        if False:
            return 10
        if self.is_alive():
            self.event.set()
            self.thread.join()
        self.thread = None