import multiprocessing
import random
import sys
import threading
from multiprocessing.synchronize import Lock
from scalene.scalene_profiler import Scalene
from typing import Any

def _recreate_replacement_sem_lock():
    if False:
        return 10
    return ReplacementSemLock()

class ReplacementSemLock(multiprocessing.synchronize.Lock):

    def __init__(self, ctx=None):
        if False:
            for i in range(10):
                print('nop')
        if ctx is None:
            ctx = multiprocessing.get_context()
        super().__init__(ctx=ctx)

    def __enter__(self) -> bool:
        if False:
            print('Hello World!')
        max_timeout = sys.getswitchinterval()
        tident = threading.get_ident()
        while True:
            Scalene.set_thread_sleeping(tident)
            timeout = random.random() * max_timeout
            acquired = self._semlock.acquire(timeout=timeout)
            Scalene.reset_thread_sleeping(tident)
            if acquired:
                return True
            else:
                max_timeout *= 2

    def __exit__(self, *args: Any) -> None:
        if False:
            i = 10
            return i + 15
        super().__exit__(*args)

    def __reduce__(self):
        if False:
            print('Hello World!')
        return (_recreate_replacement_sem_lock, ())