"""Logging utilities."""
import logging as _logging
import os as _os
import sys as _sys
import _thread
import time as _time
import traceback as _traceback
from logging import DEBUG
from logging import ERROR
from logging import FATAL
from logging import INFO
from logging import WARN
import threading
from tensorflow.python.util.tf_export import tf_export
_logger = None
_logger_lock = threading.Lock()

def error_log(error_msg, level=ERROR):
    if False:
        print('Hello World!')
    'Empty helper method.'
    del error_msg, level

def _get_caller(offset=3):
    if False:
        for i in range(10):
            print('nop')
    'Returns a code and frame object for the lowest non-logging stack frame.'
    f = _sys._getframe(offset)
    our_file = f.f_code.co_filename
    f = f.f_back
    while f:
        code = f.f_code
        if code.co_filename != our_file:
            return (code, f)
        f = f.f_back
    return (None, None)
if _sys.version_info.major >= 3 and _sys.version_info.minor >= 8:

    def _logger_find_caller(stack_info=False, stacklevel=1):
        if False:
            for i in range(10):
                print('nop')
        (code, frame) = _get_caller(4)
        sinfo = None
        if stack_info:
            sinfo = '\n'.join(_traceback.format_stack())
        if code:
            return (code.co_filename, frame.f_lineno, code.co_name, sinfo)
        else:
            return ('(unknown file)', 0, '(unknown function)', sinfo)
elif _sys.version_info.major >= 3 and _sys.version_info.minor >= 2:

    def _logger_find_caller(stack_info=False):
        if False:
            i = 10
            return i + 15
        (code, frame) = _get_caller(4)
        sinfo = None
        if stack_info:
            sinfo = '\n'.join(_traceback.format_stack())
        if code:
            return (code.co_filename, frame.f_lineno, code.co_name, sinfo)
        else:
            return ('(unknown file)', 0, '(unknown function)', sinfo)
else:

    def _logger_find_caller():
        if False:
            while True:
                i = 10
        (code, frame) = _get_caller(4)
        if code:
            return (code.co_filename, frame.f_lineno, code.co_name)
        else:
            return ('(unknown file)', 0, '(unknown function)')

@tf_export('get_logger')
def get_logger():
    if False:
        for i in range(10):
            print('nop')
    'Return TF logger instance.\n\n  Returns:\n    An instance of the Python logging library Logger.\n\n  See Python documentation (https://docs.python.org/3/library/logging.html)\n  for detailed API. Below is only a summary.\n\n  The logger has 5 levels of logging from the most serious to the least:\n\n  1. FATAL\n  2. ERROR\n  3. WARN\n  4. INFO\n  5. DEBUG\n\n  The logger has the following methods, based on these logging levels:\n\n  1. fatal(msg, *args, **kwargs)\n  2. error(msg, *args, **kwargs)\n  3. warn(msg, *args, **kwargs)\n  4. info(msg, *args, **kwargs)\n  5. debug(msg, *args, **kwargs)\n\n  The `msg` can contain string formatting.  An example of logging at the `ERROR`\n  level\n  using string formating is:\n\n  >>> tf.get_logger().error("The value %d is invalid.", 3)\n\n  You can also specify the logging verbosity.  In this case, the\n  WARN level log will not be emitted:\n\n  >>> tf.get_logger().setLevel(ERROR)\n  >>> tf.get_logger().warn("This is a warning.")\n  '
    global _logger
    if _logger:
        return _logger
    _logger_lock.acquire()
    try:
        if _logger:
            return _logger
        logger = _logging.getLogger('tensorflow')
        logger.findCaller = _logger_find_caller
        if not _logging.getLogger().handlers:
            _interactive = False
            try:
                if _sys.ps1:
                    _interactive = True
            except AttributeError:
                _interactive = _sys.flags.interactive
            if _interactive:
                logger.setLevel(INFO)
                _logging_target = _sys.stdout
            else:
                _logging_target = _sys.stderr
            _handler = _logging.StreamHandler(_logging_target)
            _handler.setFormatter(_logging.Formatter(_logging.BASIC_FORMAT, None))
            logger.addHandler(_handler)
        _logger = logger
        return _logger
    finally:
        _logger_lock.release()

@tf_export(v1=['logging.log'])
def log(level, msg, *args, **kwargs):
    if False:
        return 10
    get_logger().log(level, msg, *args, **kwargs)

@tf_export(v1=['logging.debug'])
def debug(msg, *args, **kwargs):
    if False:
        while True:
            i = 10
    get_logger().debug(msg, *args, **kwargs)

@tf_export(v1=['logging.error'])
def error(msg, *args, **kwargs):
    if False:
        while True:
            i = 10
    get_logger().error(msg, *args, **kwargs)

@tf_export(v1=['logging.fatal'])
def fatal(msg, *args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    get_logger().fatal(msg, *args, **kwargs)

@tf_export(v1=['logging.info'])
def info(msg, *args, **kwargs):
    if False:
        i = 10
        return i + 15
    get_logger().info(msg, *args, **kwargs)

@tf_export(v1=['logging.warn'])
def warn(msg, *args, **kwargs):
    if False:
        print('Hello World!')
    get_logger().warning(msg, *args, **kwargs)

@tf_export(v1=['logging.warning'])
def warning(msg, *args, **kwargs):
    if False:
        while True:
            i = 10
    get_logger().warning(msg, *args, **kwargs)
_level_names = {FATAL: 'FATAL', ERROR: 'ERROR', WARN: 'WARN', INFO: 'INFO', DEBUG: 'DEBUG'}
_THREAD_ID_MASK = 2 * _sys.maxsize + 1
_log_prefix = None
_log_counter_per_token = {}

@tf_export(v1=['logging.TaskLevelStatusMessage'])
def TaskLevelStatusMessage(msg):
    if False:
        while True:
            i = 10
    error(msg)

@tf_export(v1=['logging.flush'])
def flush():
    if False:
        for i in range(10):
            print('nop')
    raise NotImplementedError()

@tf_export(v1=['logging.vlog'])
def vlog(level, msg, *args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    get_logger().log(level, msg, *args, **kwargs)

def _GetNextLogCountPerToken(token):
    if False:
        return 10
    'Wrapper for _log_counter_per_token.\n\n  Args:\n    token: The token for which to look up the count.\n\n  Returns:\n    The number of times this function has been called with\n    *token* as an argument (starting at 0)\n  '
    global _log_counter_per_token
    _log_counter_per_token[token] = 1 + _log_counter_per_token.get(token, -1)
    return _log_counter_per_token[token]

@tf_export(v1=['logging.log_every_n'])
def log_every_n(level, msg, n, *args):
    if False:
        return 10
    "Log 'msg % args' at level 'level' once per 'n' times.\n\n  Logs the 1st call, (N+1)st call, (2N+1)st call,  etc.\n  Not threadsafe.\n\n  Args:\n    level: The level at which to log.\n    msg: The message to be logged.\n    n: The number of times this should be called before it is logged.\n    *args: The args to be substituted into the msg.\n  "
    count = _GetNextLogCountPerToken(_GetFileAndLine())
    log_if(level, msg, not count % n, *args)

@tf_export(v1=['logging.log_first_n'])
def log_first_n(level, msg, n, *args):
    if False:
        i = 10
        return i + 15
    "Log 'msg % args' at level 'level' only first 'n' times.\n\n  Not threadsafe.\n\n  Args:\n    level: The level at which to log.\n    msg: The message to be logged.\n    n: The number of times this should be called before it is logged.\n    *args: The args to be substituted into the msg.\n  "
    count = _GetNextLogCountPerToken(_GetFileAndLine())
    log_if(level, msg, count < n, *args)

@tf_export(v1=['logging.log_if'])
def log_if(level, msg, condition, *args):
    if False:
        while True:
            i = 10
    "Log 'msg % args' at level 'level' only if condition is fulfilled."
    if condition:
        vlog(level, msg, *args)

def _GetFileAndLine():
    if False:
        while True:
            i = 10
    'Returns (filename, linenumber) for the stack frame.'
    (code, f) = _get_caller()
    if not code:
        return ('<unknown>', 0)
    return (code.co_filename, f.f_lineno)

def google2_log_prefix(level, timestamp=None, file_and_line=None):
    if False:
        for i in range(10):
            print('nop')
    'Assemble a logline prefix using the google2 format.'
    global _level_names
    now = timestamp or _time.time()
    now_tuple = _time.localtime(now)
    now_microsecond = int(1000000.0 * (now % 1.0))
    (filename, line) = file_and_line or _GetFileAndLine()
    basename = _os.path.basename(filename)
    severity = 'I'
    if level in _level_names:
        severity = _level_names[level][0]
    s = '%c%02d%02d %02d:%02d:%02d.%06d %5d %s:%d] ' % (severity, now_tuple[1], now_tuple[2], now_tuple[3], now_tuple[4], now_tuple[5], now_microsecond, _get_thread_id(), basename, line)
    return s

@tf_export(v1=['logging.get_verbosity'])
def get_verbosity():
    if False:
        while True:
            i = 10
    'Return how much logging output will be produced.'
    return get_logger().getEffectiveLevel()

@tf_export(v1=['logging.set_verbosity'])
def set_verbosity(v):
    if False:
        return 10
    'Sets the threshold for what messages will be logged.'
    get_logger().setLevel(v)

def _get_thread_id():
    if False:
        while True:
            i = 10
    'Get id of current thread, suitable for logging as an unsigned quantity.'
    thread_id = _thread.get_ident()
    return thread_id & _THREAD_ID_MASK
_log_prefix = google2_log_prefix
tf_export(v1=['logging.DEBUG']).export_constant(__name__, 'DEBUG')
tf_export(v1=['logging.ERROR']).export_constant(__name__, 'ERROR')
tf_export(v1=['logging.FATAL']).export_constant(__name__, 'FATAL')
tf_export(v1=['logging.INFO']).export_constant(__name__, 'INFO')
tf_export(v1=['logging.WARN']).export_constant(__name__, 'WARN')