import linecache
import sys
import time
from dataclasses import dataclass
from types import FrameType, FunctionType
from typing import Callable, List, Optional, Tuple
from tribler.core.utilities.slow_coro_detection import logger
from tribler.core.utilities.utilities import switch_interval
SWITCH_INTERVAL = 0.1
_main_thread_stack_tracking_is_enabled: bool = False
_main_thread_stack: List[Tuple[FrameType, float]] = []

def main_stack_tracking_is_enabled() -> bool:
    if False:
        for i in range(10):
            print('nop')
    return _main_thread_stack_tracking_is_enabled

def main_thread_profile(frame: FrameType, event: str, _, now=time.time):
    if False:
        print('Hello World!')
    '\n    A hook that calls before and after a function call in the main thread if the stack tracking is activated\n    '
    if event == 'call':
        _main_thread_stack.append((frame, now()))
    elif event == 'return' and _main_thread_stack:
        _main_thread_stack.pop()
    return main_thread_profile

def start_main_thread_stack_tracing() -> Callable:
    if False:
        for i in range(10):
            print('nop')
    '\n    Activates the profiler hook in the main thread. Note that it makes Python functions about two times slower.\n    The compiled code is run as fast, so libtorrent calls and database queries should be as efficient as before.\n\n    Returns the profiler function (for testing purpose)\n    '
    logger.info('Start tracing of coroutine stack to show stack for slow coroutines (makes code execution slower)')
    global _main_thread_stack_tracking_is_enabled
    _main_thread_stack_tracking_is_enabled = True
    sys.setprofile(main_thread_profile)
    return main_thread_profile

def stop_main_thread_stack_tracing() -> Callable:
    if False:
        while True:
            i = 10
    '\n    Deactivates the profiler hook in the main thread.\n    Returns the previous profiler function (for testing purpose)\n    '
    previous_profiler = sys.getprofile()
    sys.setprofile(None)
    global _main_thread_stack_tracking_is_enabled
    _main_thread_stack_tracking_is_enabled = False
    return previous_profiler

@dataclass
class StackFrameInfo:
    func_name: str
    file_name: str
    start_time: float
    line_number: Optional[int] = None
    is_under_profiling: float = False

def _get_main_thread_stack_info() -> List[StackFrameInfo]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Quickly copies necessary information from the main thread stack, so it is possible later to format a usual\n    traceback in a separate thread.\n\n    The function temporarily changes the interpreter’s thread switch interval to prevent thread switch during\n    the stack copying. It is a lighter analogue of holding the GIL (Global Interpreter Lock).\n    '
    with switch_interval(SWITCH_INTERVAL):
        stack_info = []
        for (frame, start_time) in _main_thread_stack:
            func_name = frame.f_code.co_name
            if func_name == 'profile_wrapper':
                original_func: FunctionType = frame.f_locals.get('func')
                if original_func is not None:
                    stack_info.append(StackFrameInfo(func_name=original_func.__name__, file_name=original_func.__code__.co_filename, start_time=start_time, is_under_profiling=True))
                    break
            stack_info.append(StackFrameInfo(func_name=func_name, file_name=frame.f_code.co_filename, line_number=frame.f_lineno, start_time=start_time))
    return stack_info

def get_main_thread_stack(stack_cut_duration: Optional[float]=None, limit: Optional[int]=None, enable_profiling_tip: bool=True) -> str:
    if False:
        i = 10
        return i + 15
    '\n    Obtains the main thread stack and format it in a usual way.\n    '
    traceback_items = []
    stack_info = _get_main_thread_stack_info()
    now = time.time()
    func_name_for_tip = None
    for frame_info in stack_info:
        duration = now - frame_info.start_time
        if frame_info.is_under_profiling:
            source_line = '<is currently under profiling>'
            enable_profiling_tip = False
        elif frame_info.line_number:
            source_line = linecache.getline(frame_info.file_name, frame_info.line_number).strip()
        else:
            source_line = ''
        traceback_item = f'''  File "{frame_info.file_name}", line {frame_info.line_number or '?'}, in {frame_info.func_name} (function started {duration:.3f} seconds ago)\n    {source_line or '?'}'''
        if stack_cut_duration is not None and duration < stack_cut_duration:
            if traceback_items:
                traceback_items.pop()
            break
        func_name_for_tip = frame_info.func_name
        traceback_items.append(traceback_item)
    if limit:
        traceback_items = traceback_items[-limit:]
    if not traceback_items:
        return '<no traceback found>'
    traceback_str = '\n'.join(traceback_items) + '\n'
    tip = ''
    if enable_profiling_tip and func_name_for_tip is not None:
        tip = f'\nTip: by applying the `@profile()` decorator to the `{func_name_for_tip}` function, you can obtain statistics for its internal calls and see the reason for slowness'
    return f'Traceback (most recent call last):\n{traceback_str}{tip}'