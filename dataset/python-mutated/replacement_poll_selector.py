import selectors
import sys
import threading
import time
from typing import List, Optional, Tuple
from scalene.scalene_profiler import Scalene

@Scalene.shim
def replacement_poll_selector(scalene: Scalene) -> None:
    if False:
        print('Hello World!')
    '\n    A replacement for selectors.PollSelector that\n    periodically wakes up to accept signals\n    '

    class ReplacementPollSelector(selectors.PollSelector):

        def select(self, timeout: Optional[float]=-1) -> List[Tuple[selectors.SelectorKey, int]]:
            if False:
                print('Hello World!')
            tident = threading.get_ident()
            start_time = time.perf_counter()
            if not timeout or timeout < 0:
                interval = sys.getswitchinterval()
            else:
                interval = min(timeout, sys.getswitchinterval())
            while True:
                scalene.set_thread_sleeping(tident)
                selected = super().select(interval)
                scalene.reset_thread_sleeping(tident)
                if selected or timeout == 0:
                    return selected
                end_time = time.perf_counter()
                if timeout and timeout != -1:
                    if end_time - start_time >= timeout:
                        return []
    ReplacementPollSelector.__qualname__ = 'replacement_poll_selector.ReplacementPollSelector'
    selectors.PollSelector = ReplacementPollSelector