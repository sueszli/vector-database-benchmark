import queue
import threading
from typing import Any, Generic, Optional, TypeVar
T = TypeVar('T')

class ScaleneSigQueue(Generic[T]):

    def __init__(self, process: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.queue: queue.SimpleQueue[Optional[T]] = queue.SimpleQueue()
        self.process = process
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.RLock()

    def put(self, item: Optional[T]) -> None:
        if False:
            i = 10
            return i + 15
        'Add an item to the queue.'
        self.queue.put(item)

    def get(self) -> Optional[T]:
        if False:
            while True:
                i = 10
        'Get one item from the queue.'
        return self.queue.get()

    def start(self) -> None:
        if False:
            return 10
        'Start processing.'
        if not self.thread:
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()

    def stop(self) -> None:
        if False:
            while True:
                i = 10
        'Stop processing.'
        if self.thread:
            self.queue.put(None)
            self.thread.join()
            self.thread = None

    def run(self) -> None:
        if False:
            while True:
                i = 10
        'Run the function processing items until stop is called.\n\n        Executed in a separate thread.'
        while True:
            item = self.queue.get()
            if item is None:
                break
            with self.lock:
                self.process(*item)