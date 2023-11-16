from __future__ import absolute_import
import logging
import time
from contextlib import contextmanager

def make_timing_logger(logger, precision=3, level=logging.DEBUG):
    if False:
        return 10
    ' Return a timing logger.\n\n    Usage::\n\n        >>> logger = logging.getLogger(\'foobar\')\n        >>> log_time = make_timing_logger(\n        ...     logger, level=logging.INFO, precision=2)\n        >>>\n        >>> with log_time("hello %s", "world"):\n        ...     time.sleep(1)\n        INFO:foobar:hello world in 1.00s\n    '

    @contextmanager
    def log_time(msg, *args):
        if False:
            print('Hello World!')
        ' Log `msg` and `*args` with (naive wallclock) timing information\n        when the context block exits.\n        '
        start_time = time.time()
        try:
            yield
        finally:
            message = '{} in %0.{}fs'.format(msg, precision)
            duration = time.time() - start_time
            args = args + (duration,)
            logger.log(level, message, *args)
    return log_time