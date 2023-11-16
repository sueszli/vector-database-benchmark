from threading import Thread
from typing import Optional
import psutil
import time

class MemoryCheckerThread(Thread):

    def __init__(self) -> None:
        if False:
            return 10
        super().__init__()
        self.start_mem: int = psutil.virtual_memory().used
        self.max_mem: int = 0
        self.min_mem: int = self.start_mem
        self.working: bool = False

    @property
    def estm_mem(self) -> int:
        if False:
            return 10
        if self.max_mem - self.start_mem > 0:
            return self.max_mem - self.start_mem
        else:
            return max(0, self.max_mem - self.min_mem)

    def stop(self) -> None:
        if False:
            return 10
        self.working = False

    def run(self) -> None:
        if False:
            i = 10
            return i + 15
        self.working = True
        while not self._is_stopped and self.working:
            mem = psutil.virtual_memory().used
            if mem > self.max_mem:
                self.max_mem = mem
            if mem < self.min_mem:
                self.min_mem = mem
            time.sleep(0.5)

class MemoryChecker:
    _thread: Optional[MemoryCheckerThread]

    def __init__(self, do_check: bool=True) -> None:
        if False:
            while True:
                i = 10
        self._thread = MemoryCheckerThread() if do_check else None

    def __enter__(self) -> 'MemoryChecker':
        if False:
            i = 10
            return i + 15
        if self._thread:
            self._thread.start()
        return self

    def __exit__(self, *exc) -> bool:
        if False:
            while True:
                i = 10
        if self._thread:
            self._thread.stop()
        return False

    @property
    def estm_mem(self) -> Optional[int]:
        if False:
            print('Hello World!')
        return self._thread.estm_mem if self._thread else None