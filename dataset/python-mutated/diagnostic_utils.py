"""
Diagnostic utilities
"""
import functools
import sys
import traceback

def create_log_exception_decorator(logger):
    if False:
        return 10
    "Create a decorator that logs and reraises any exceptions that escape\n    the decorated function\n\n    :param logging.Logger logger:\n    :returns: the decorator\n    :rtype: callable\n\n    Usage example\n\n    import logging\n\n    from pika.diagnostics_utils import create_log_exception_decorator\n\n    _log_exception = create_log_exception_decorator(logging.getLogger(__name__))\n\n    @_log_exception\n    def my_func_or_method():\n        raise Exception('Oops!')\n\n    "

    def log_exception(func):
        if False:
            print('Hello World!')
        'The decorator returned by the parent function\n\n        :param func: function to be wrapped\n        :returns: the function wrapper\n        :rtype: callable\n        '

        @functools.wraps(func)
        def log_exception_func_wrap(*args, **kwargs):
            if False:
                while True:
                    i = 10
            "The wrapper function returned by the decorator. Invokes the\n            function with the given args/kwargs and returns the function's\n            return value. If the function exits with an exception, logs the\n            exception traceback and re-raises the\n\n            :param args: positional args passed to wrapped function\n            :param kwargs: keyword args passed to wrapped function\n            :returns: whatever the wrapped function returns\n            :rtype: object\n            "
            try:
                return func(*args, **kwargs)
            except:
                logger.exception("Wrapped func exited with exception. Caller's stack:\n%s", ''.join(traceback.format_exception(*sys.exc_info())))
                raise
        return log_exception_func_wrap
    return log_exception