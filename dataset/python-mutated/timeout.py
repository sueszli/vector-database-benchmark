from __future__ import annotations
import os
import signal
from threading import Timer
from typing import ContextManager
from airflow.exceptions import AirflowTaskTimeout
from airflow.utils.log.logging_mixin import LoggingMixin
from airflow.utils.platform import IS_WINDOWS
_timeout = ContextManager[None]

class TimeoutWindows(_timeout, LoggingMixin):
    """Windows timeout version: To be used in a ``with`` block and timeout its content."""

    def __init__(self, seconds=1, error_message='Timeout'):
        if False:
            print('Hello World!')
        super().__init__()
        self._timer: Timer | None = None
        self.seconds = seconds
        self.error_message = error_message + ', PID: ' + str(os.getpid())

    def handle_timeout(self, *args):
        if False:
            i = 10
            return i + 15
        'Log information and raises AirflowTaskTimeout.'
        self.log.error('Process timed out, PID: %s', str(os.getpid()))
        raise AirflowTaskTimeout(self.error_message)

    def __enter__(self):
        if False:
            return 10
        if self._timer:
            self._timer.cancel()
        self._timer = Timer(self.seconds, self.handle_timeout)
        self._timer.start()

    def __exit__(self, type_, value, traceback):
        if False:
            print('Hello World!')
        if self._timer:
            self._timer.cancel()
            self._timer = None

class TimeoutPosix(_timeout, LoggingMixin):
    """POSIX Timeout version: To be used in a ``with`` block and timeout its content."""

    def __init__(self, seconds=1, error_message='Timeout'):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.seconds = seconds
        self.error_message = error_message + ', PID: ' + str(os.getpid())

    def handle_timeout(self, signum, frame):
        if False:
            i = 10
            return i + 15
        'Log information and raises AirflowTaskTimeout.'
        self.log.error('Process timed out, PID: %s', str(os.getpid()))
        raise AirflowTaskTimeout(self.error_message)

    def __enter__(self):
        if False:
            print('Hello World!')
        try:
            signal.signal(signal.SIGALRM, self.handle_timeout)
            signal.setitimer(signal.ITIMER_REAL, self.seconds)
        except ValueError:
            self.log.warning("timeout can't be used in the current context", exc_info=True)

    def __exit__(self, type_, value, traceback):
        if False:
            return 10
        try:
            signal.setitimer(signal.ITIMER_REAL, 0)
        except ValueError:
            self.log.warning("timeout can't be used in the current context", exc_info=True)
if IS_WINDOWS:
    timeout: type[TimeoutWindows | TimeoutPosix] = TimeoutWindows
else:
    timeout = TimeoutPosix