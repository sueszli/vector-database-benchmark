import multiprocessing
import os
import sys
import threading
import time
from scalene.scalene_profiler import Scalene
minor_version = sys.version_info.minor

@Scalene.shim
def replacement_pjoin(scalene: Scalene) -> None:
    if False:
        return 10

    def replacement_process_join(self, timeout: float=-1) -> None:
        if False:
            print('Hello World!')
        '\n        A drop-in replacement for multiprocessing.Process.join\n        that periodically yields to handle signals\n        '
        if minor_version >= 7:
            self._check_closed()
        assert self._parent_pid == os.getpid(), 'can only join a child process'
        assert self._popen is not None, 'can only join a started process'
        tident = threading.get_ident()
        if timeout < 0:
            interval = sys.getswitchinterval()
        else:
            interval = min(timeout, sys.getswitchinterval())
        start_time = time.perf_counter()
        while True:
            scalene.set_thread_sleeping(tident)
            res = self._popen.wait(interval)
            if res is not None:
                from multiprocessing.process import _children
                scalene.remove_child_pid(self.pid)
                _children.discard(self)
                return
            scalene.reset_thread_sleeping(tident)
            if timeout != -1:
                end_time = time.perf_counter()
                if end_time - start_time >= timeout:
                    from multiprocessing.process import _children
                    _children.discard(self)
                    return
    multiprocessing.Process.join = replacement_process_join