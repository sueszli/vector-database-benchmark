import logging as _logging
import os as _os
import sys as _sys
import threading
import time as _time
from logging import DEBUG, ERROR, FATAL, INFO, WARN
import six
from tensorlayer.decorators import deprecated
__all__ = ['DEBUG', 'debug', 'ERROR', 'error', 'FATAL', 'fatal', 'INFO', 'info', 'WARN', 'warning', 'warn', 'set_verbosity', 'get_verbosity']
_logger = None
_logger_lock = threading.Lock()
_level_names = {FATAL: 'FATAL', ERROR: 'ERROR', WARN: 'WARN', INFO: 'INFO', DEBUG: 'DEBUG'}

def _get_logger():
    if False:
        return 10
    global _logger
    if _logger is not None:
        return _logger
    _logger_lock.acquire()
    try:
        if _logger:
            return _logger
        logger = _logging.getLogger('tensorlayer')
        if not _logging.getLogger().handlers:
            if hasattr(_sys, 'ps1'):
                _interactive = True
            else:
                _interactive = _sys.flags.interactive
            if _interactive:
                logger.setLevel(INFO)
                _logging_target = _sys.stdout
            else:
                _logging_target = _sys.stderr
            _handler = _logging.StreamHandler(_logging_target)
            _handler.setFormatter(_logging.Formatter('[TL] %(message)s'))
            logger.addHandler(_handler)
        _logger = logger
        return _logger
    finally:
        _logger_lock.release()

def log(level, msg, *args, **kwargs):
    if False:
        while True:
            i = 10
    _get_logger().log(level, msg, *args, **kwargs)

def debug(msg, *args, **kwargs):
    if False:
        while True:
            i = 10
    _get_logger().debug(msg, *args, **kwargs)

def info(msg, *args, **kwargs):
    if False:
        i = 10
        return i + 15
    _get_logger().info(msg, *args, **kwargs)

def error(msg, *args, **kwargs):
    if False:
        return 10
    _get_logger().error('ERROR: %s' % msg, *args, **kwargs)

def fatal(msg, *args, **kwargs):
    if False:
        i = 10
        return i + 15
    _get_logger().fatal('FATAL: %s' % msg, *args, **kwargs)

@deprecated(date='2018-09-30', instructions='This API is deprecated. Please use as `tl.logging.warning`')
def warn(msg, *args, **kwargs):
    if False:
        print('Hello World!')
    warning(msg, *args, **kwargs)

def warning(msg, *args, **kwargs):
    if False:
        return 10
    _get_logger().warning('WARNING: %s' % msg, *args, **kwargs)
_THREAD_ID_MASK = 2 * _sys.maxsize + 1
_log_prefix = None
_log_counter_per_token = {}

def TaskLevelStatusMessage(msg):
    if False:
        print('Hello World!')
    error(msg)

def flush():
    if False:
        while True:
            i = 10
    raise NotImplementedError()

def vlog(level, msg, *args, **kwargs):
    if False:
        print('Hello World!')
    _get_logger().log(level, msg, *args, **kwargs)

def _GetNextLogCountPerToken(token):
    if False:
        for i in range(10):
            print('nop')
    'Wrapper for _log_counter_per_token.\n\n    Args:\n    token: The token for which to look up the count.\n\n    Returns:\n    The number of times this function has been called with\n    *token* as an argument (starting at 0)\n    '
    global _log_counter_per_token
    _log_counter_per_token[token] = 1 + _log_counter_per_token.get(token, -1)
    return _log_counter_per_token[token]

def log_every_n(level, msg, n, *args):
    if False:
        while True:
            i = 10
    "Log 'msg % args' at level 'level' once per 'n' times.\n\n    Logs the 1st call, (N+1)st call, (2N+1)st call,  etc.\n    Not threadsafe.\n\n    Args:\n    level: The level at which to log.\n    msg: The message to be logged.\n    n: The number of times this should be called before it is logged.\n    *args: The args to be substituted into the msg.\n    "
    count = _GetNextLogCountPerToken(_GetFileAndLine())
    log_if(level, msg, not count % n, *args)

def log_first_n(level, msg, n, *args):
    if False:
        for i in range(10):
            print('nop')
    "Log 'msg % args' at level 'level' only first 'n' times.\n\n    Not threadsafe.\n\n    Args:\n    level: The level at which to log.\n    msg: The message to be logged.\n    n: The number of times this should be called before it is logged.\n    *args: The args to be substituted into the msg.\n    "
    count = _GetNextLogCountPerToken(_GetFileAndLine())
    log_if(level, msg, count < n, *args)

def log_if(level, msg, condition, *args):
    if False:
        return 10
    "Log 'msg % args' at level 'level' only if condition is fulfilled."
    if condition:
        vlog(level, msg, *args)

def _GetFileAndLine():
    if False:
        while True:
            i = 10
    'Returns (filename, linenumber) for the stack frame.'
    f = _sys._getframe()
    our_file = f.f_code.co_filename
    f = f.f_back
    while f:
        code = f.f_code
        if code.co_filename != our_file:
            return (code.co_filename, f.f_lineno)
        f = f.f_back
    return ('<unknown>', 0)

def google2_log_prefix(level, timestamp=None, file_and_line=None):
    if False:
        i = 10
        return i + 15
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
    s = '%c%02d%02d %02d: %02d: %02d.%06d %5d %s: %d] ' % (severity, now_tuple[1], now_tuple[2], now_tuple[3], now_tuple[4], now_tuple[5], now_microsecond, _get_thread_id(), basename, line)
    return s

def get_verbosity():
    if False:
        while True:
            i = 10
    'Return how much logging output will be produced.'
    return _get_logger().getEffectiveLevel()

def set_verbosity(v):
    if False:
        for i in range(10):
            print('nop')
    'Sets the threshold for what messages will be logged.'
    _get_logger().setLevel(v)

def _get_thread_id():
    if False:
        for i in range(10):
            print('nop')
    'Get id of current thread, suitable for logging as an unsigned quantity.'
    thread_id = six.moves._thread.get_ident()
    return thread_id & _THREAD_ID_MASK
_log_prefix = google2_log_prefix