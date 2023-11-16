import logging
import os
import sys
import tempfile
import time
import traceback
import psutil
from tribler.core.utilities.utilities import show_system_popup
logger = logging.getLogger(__name__)

def error_and_exit(title, main_text):
    if False:
        print('Hello World!')
    '\n    Show a pop-up window and sys.exit() out of Python.\n\n    :param title: the short error description\n    :param main_text: the long error description\n    '
    show_system_popup(title, main_text)
    sys.exit(1)

def check_read_write():
    if False:
        i = 10
        return i + 15
    '\n    Check if we have access to file IO, or exit with an error.\n    '
    try:
        tempfile.gettempdir()
    except OSError:
        error_and_exit('No write access!', 'Tribler does not seem to be able to have access to your filesystem. ' + 'Please grant Tribler the proper permissions and try again.')

def check_environment():
    if False:
        return 10
    '\n    Perform all of the pre-Tribler checks to see if we can run on this platform.\n    '
    logger.info('Check environment')
    check_read_write()

def check_free_space():
    if False:
        while True:
            i = 10
    logger.info('Check free space')
    try:
        free_space = psutil.disk_usage('.').free / (1024 * 1024.0)
        if free_space < 100:
            error_and_exit('Insufficient disk space', 'You have less than 100MB of usable disk space. ' + 'Please free up some space and run Tribler again.')
    except ImportError as ie:
        logger.error(ie)
        error_and_exit('Import Error', f'Import error: {ie}')

def set_process_priority(pid=None, priority_order=1):
    if False:
        return 10
    '\n    Sets process priority based on order provided. Note order range is 0-5 and higher value indicates higher priority.\n    :param pid: Process ID or None. If None, uses current process.\n    :param priority_order: Priority order (0-5). Higher value means higher priority.\n    '
    if priority_order < 0 or priority_order > 5:
        return
    if sys.platform not in {'win32', 'darwin', 'linux'}:
        return
    if sys.platform == 'win32':
        priority_classes = [psutil.IDLE_PRIORITY_CLASS, psutil.BELOW_NORMAL_PRIORITY_CLASS, psutil.NORMAL_PRIORITY_CLASS, psutil.ABOVE_NORMAL_PRIORITY_CLASS, psutil.HIGH_PRIORITY_CLASS, psutil.REALTIME_PRIORITY_CLASS]
    else:
        priority_classes = [5, 4, 3, 2, 1, 0]
    try:
        process = psutil.Process(pid if pid else os.getpid())
        process.nice(priority_classes[priority_order])
    except psutil.Error as e:
        logger.exception(e)

def enable_fault_handler(log_dir):
    if False:
        for i in range(10):
            print('nop')
    '\n    Enables fault handler if the module is available.\n    '
    logger.info(f'Enable fault handler: "{log_dir}"')
    try:
        import faulthandler
        if not log_dir.exists():
            log_dir.mkdir(parents=True, exist_ok=True)
        crash_file = log_dir / 'crash-report.log'
        faulthandler.enable(file=open(str(crash_file), 'w'), all_threads=True)
    except ImportError:
        logger.error('Fault Handler module not found.')

def check_and_enable_code_tracing(process_name, log_dir):
    if False:
        print('Hello World!')
    '\n    Checks and enable trace logging if --trace-exception or --trace-debug system flag is present.\n    :param process_name: used as prefix for log file\n    :return: Log file handler\n    '
    logger.info(f'Check and enable code tracing. Process name: "{process_name}". Log dir: "{log_dir}"')
    trace_logger = None
    if '--trace-exception' in sys.argv[1:]:
        trace_logger = open(log_dir / f'{process_name}-exceptions.log', 'w')
        sys.settrace(lambda frame, event, args: trace_calls(trace_logger, frame, event, args, filter_exceptions_only=True))
    elif '--trace-debug' in sys.argv[1:]:
        trace_logger = open(log_dir / f'{process_name}-debug.log', 'w')
        sys.settrace(lambda frame, event, args: trace_calls(trace_logger, frame, event, args))
    return trace_logger

def trace_calls(file_handler, frame, event, args, filter_exceptions_only=False):
    if False:
        return 10
    '\n    Trace all Tribler calls as it runs. Useful for debugging.\n    Checkout: https://pymotw.com/2/sys/tracing.html\n    :param file_handler: File handler where logs will be written to.\n    :param frame: Current frame\n    :param event: Call event\n    :param args: None\n    :return: next trace handler\n    '
    if event != 'call' or file_handler.closed:
        return
    if not filter_exceptions_only:
        co = frame.f_code
        func_name = co.co_name
        if func_name == 'write':
            return
        func_line_no = frame.f_lineno
        func_filename = co.co_filename
        caller = frame.f_back
        caller_line_no = caller.f_lineno
        caller_filename = caller.f_code.co_filename
        if 'tribler' in caller_filename.lower() or 'tribler' in func_filename.lower():
            trace_line = f'[{time.time()}] {func_filename}:{func_name}, line {func_line_no} called from {caller_filename}, line {caller_line_no}\n'
            file_handler.write(trace_line)
            file_handler.flush()
    return lambda _frame, _event, _args: trace_exceptions(file_handler, _frame, _event, _args)

def trace_exceptions(file_handler, frame, event, args):
    if False:
        return 10
    '\n    Trace all Tribler exceptions as it runs. Useful for debugging.\n    Checkout: https://pymotw.com/2/sys/tracing.html\n    :param file_handler: File handler where logs will be written to.\n    :param frame: Current frame\n    :param event: Exception event\n    :param args: exc_type, exc_value, exc_traceback\n    :return: None\n    '
    if event != 'exception' or file_handler.closed:
        return
    co = frame.f_code
    func_line_no = frame.f_lineno
    func_filename = co.co_filename
    if 'tribler' in func_filename.lower():
        (exc_type, exc_value, exc_traceback) = args
        trace_line = f"[{time.time()}] Exception: {func_filename}, line {func_line_no} \n{exc_type.__name__} {exc_value} \n{''.join(traceback.format_tb(exc_traceback))}"
        file_handler.write(trace_line)
        file_handler.flush()