from __future__ import annotations
import time
from asyncio import Handle
from threading import Event, Lock, Thread
from typing import Optional
from tribler.core.utilities.slow_coro_detection import logger
from tribler.core.utilities.slow_coro_detection.utils import format_info
SLOW_CORO_DURATION_THRESHOLD = 1.0
WATCHING_THREAD_INTERVAL = 1.0

class DebugInfo:

    def __init__(self):
        if False:
            print('Hello World!')
        self.handle: Optional[Handle] = None
        self.start_time: Optional[float] = None
current = DebugInfo()
lock = Lock()
_thread: Optional[SlowCoroWatchingThread] = None

def start_watching_thread():
    if False:
        print('Hello World!')
    '\n    Starts separate thread that detects and reports slow coroutines.\n    '
    global _thread
    with lock:
        if _thread is not None:
            return
        _thread = SlowCoroWatchingThread(daemon=True)
    _thread.start()

class SlowCoroWatchingThread(Thread):
    """
    A thread that detects and reports slow coroutines.
    """

    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, *, daemon=None):
        if False:
            return 10
        super().__init__(group=group, target=target, name=name, args=args, kwargs=kwargs, daemon=daemon)
        self.stop_event = Event()

    def run(self):
        if False:
            while True:
                i = 10
        prev_reported_handle = None
        while not self.stop_event.is_set():
            time.sleep(WATCHING_THREAD_INTERVAL)
            with lock:
                (handle, start_time) = (current.handle, current.start_time)
            new_reported_handle = None
            if handle is not None:
                duration = time.time() - start_time
                if duration > SLOW_CORO_DURATION_THRESHOLD:
                    _report_freeze(current.handle, duration, first_report=prev_reported_handle is not handle)
                    new_reported_handle = handle
            prev_reported_handle = new_reported_handle

    def stop(self):
        if False:
            while True:
                i = 10
        self.stop_event.set()

def _report_freeze(handle: Handle, duration: float, first_report: bool):
    if False:
        for i in range(10):
            print('nop')
    stack_cut_duration = duration * 0.8
    if first_report:
        info_str = format_info(handle, include_stack=True, stack_cut_duration=stack_cut_duration)
        logger.error(f'Slow coroutine is occupying the loop for {duration:.3f} seconds already: {info_str}')
        return
    info_str = format_info(handle, include_stack=True, stack_cut_duration=stack_cut_duration, limit=2, enable_profiling_tip=False)
    logger.error(f'Still executing {info_str}')