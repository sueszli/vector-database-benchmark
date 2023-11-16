"""Deprecated Stopwatch implementation"""

class Stopwatch:
    """Deprecated zmq.Stopwatch implementation

    You can use Python's builtin timers (time.monotonic, etc.).
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        import warnings
        warnings.warn('zmq.Stopwatch is deprecated. Use stdlib time.monotonic and friends instead', DeprecationWarning, stacklevel=2)
        self._start = 0
        import time
        try:
            self._monotonic = time.monotonic
        except AttributeError:
            self._monotonic = time.time

    def start(self):
        if False:
            return 10
        'Start the counter'
        self._start = self._monotonic()

    def stop(self):
        if False:
            print('Hello World!')
        'Return time since start in microseconds'
        stop = self._monotonic()
        return int(1000000.0 * (stop - self._start))