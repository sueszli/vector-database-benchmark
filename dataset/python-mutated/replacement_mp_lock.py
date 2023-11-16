import multiprocessing.synchronize
import sys
import threading
from typing import Any
from scalene.scalene_profiler import Scalene
from scalene.replacement_sem_lock import ReplacementSemLock

@Scalene.shim
def replacement_mp_semlock(scalene: Scalene) -> None:
    if False:
        return 10
    ReplacementSemLock.__qualname__ = 'replacement_semlock.ReplacementSemLock'
    multiprocessing.synchronize.Lock = ReplacementSemLock