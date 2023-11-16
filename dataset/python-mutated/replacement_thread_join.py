import sys
import threading
import time
from typing import Optional
from scalene.scalene_profiler import Scalene

@Scalene.shim
def replacement_thread_join(scalene: Scalene) -> None:
    if False:
        i = 10
        return i + 15
    orig_thread_join = threading.Thread.join

    def thread_join_replacement(self: threading.Thread, timeout: Optional[float]=None) -> None:
        if False:
            while True:
                i = 10
        'We replace threading.Thread.join with this method which always\n        periodically yields.'
        start_time = time.perf_counter()
        interval = sys.getswitchinterval()
        while self.is_alive():
            scalene.set_thread_sleeping(threading.get_ident())
            orig_thread_join(self, interval)
            scalene.reset_thread_sleeping(threading.get_ident())
            if timeout is not None:
                end_time = time.perf_counter()
                if end_time - start_time >= timeout:
                    return None
        return None
    threading.Thread.join = thread_join_replacement