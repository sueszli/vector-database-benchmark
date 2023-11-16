import sys
import threading
import time
from typing import Any
from scalene.scalene_profiler import Scalene

@Scalene.shim
def replacement_lock(scalene: Scalene) -> None:
    if False:
        while True:
            i = 10

    class ReplacementLock:
        """Replace lock with a version that periodically yields and updates sleeping status."""

        def __init__(self) -> None:
            if False:
                return 10
            self.__lock: threading.Lock = scalene.get_original_lock()

        def acquire(self, blocking: bool=True, timeout: float=-1) -> bool:
            if False:
                return 10
            tident = threading.get_ident()
            if blocking == 0:
                blocking = False
            start_time = time.perf_counter()
            if blocking:
                if timeout < 0:
                    interval = sys.getswitchinterval()
                else:
                    interval = min(timeout, sys.getswitchinterval())
            else:
                interval = -1
            while True:
                scalene.set_thread_sleeping(tident)
                acquired_lock = self.__lock.acquire(blocking, interval)
                scalene.reset_thread_sleeping(tident)
                if acquired_lock:
                    return True
                if not blocking:
                    return False
                if timeout != -1:
                    end_time = time.perf_counter()
                    if end_time - start_time >= timeout:
                        return False

        def release(self) -> None:
            if False:
                while True:
                    i = 10
            self.__lock.release()

        def locked(self) -> bool:
            if False:
                print('Hello World!')
            return self.__lock.locked()

        def _at_fork_reinit(self) -> None:
            if False:
                return 10
            try:
                self.__lock._at_fork_reinit()
            except AttributeError:
                pass

        def __enter__(self) -> None:
            if False:
                print('Hello World!')
            self.acquire()

        def __exit__(self, type: str, value: str, traceback: Any) -> None:
            if False:
                for i in range(10):
                    print('nop')
            self.release()
    threading.Lock = ReplacementLock