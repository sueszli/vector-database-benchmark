import signal
import sys
from typing import List, Tuple

class ScaleneSignals:
    """
    ScaleneSignals class to configure timer signals for CPU profiling and
    to get various types of signals.
    """

    def __init__(self) -> None:
        if False:
            return 10
        self.set_timer_signals(use_virtual_time=True)
        if sys.platform != 'win32':
            self.start_profiling_signal = signal.SIGILL
            self.stop_profiling_signal = signal.SIGBUS
            self.memcpy_signal = signal.SIGPROF
            self.malloc_signal = signal.SIGXCPU
            self.free_signal = signal.SIGXFSZ
        else:
            self.start_profiling_signal = None
            self.stop_profiling_signal = None
            self.memcpy_signal = None
            self.malloc_signal = None
            self.free_signal = None

    def set_timer_signals(self, use_virtual_time: bool=True) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Set up timer signals for CPU profiling.\n\n        use_virtual_time: bool, default True\n            If True, sets virtual timer signals, otherwise sets real timer signals.\n        '
        if sys.platform == 'win32':
            self.cpu_signal = signal.SIGBREAK
            self.cpu_timer_signal = None
            return
        if use_virtual_time:
            self.cpu_timer_signal = signal.ITIMER_VIRTUAL
            self.cpu_signal = signal.SIGVTALRM
        else:
            self.cpu_timer_signal = signal.ITIMER_REAL
            self.cpu_signal = signal.SIGALRM

    def get_timer_signals(self) -> Tuple[int, signal.Signals]:
        if False:
            i = 10
            return i + 15
        '\n        Return the signals used for CPU profiling.\n\n        Returns:\n        --------\n        Tuple[int, signal.Signals]\n            Returns 2-tuple of the integers representing the CPU timer signal and the CPU signal.\n        '
        return (self.cpu_timer_signal, self.cpu_signal)

    def get_all_signals(self) -> List[int]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Return all the signals used for controlling profiling, except the CPU timer.\n\n        Returns:\n        --------\n        List[int]\n            Returns a list of integers representing all the profiling signals except the CPU timer.\n        '
        return [self.start_profiling_signal, self.stop_profiling_signal, self.memcpy_signal, self.malloc_signal, self.free_signal, self.cpu_signal]