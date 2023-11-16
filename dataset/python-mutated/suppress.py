"""
Module for suppress errors in Amazon Provider.

.. warning::
    Only for internal usage, this module might be changed or removed in the future
    without any further notice.

:meta: private
"""
from __future__ import annotations
import logging
from functools import wraps
from typing import Callable, TypeVar
from airflow.typing_compat import ParamSpec
PS = ParamSpec('PS')
RT = TypeVar('RT')
log = logging.getLogger(__name__)

def return_on_error(return_value: RT):
    if False:
        return 10
    "\n    Suppress any ``Exception`` raised in decorator function.\n\n    Main use-case when functional is optional, however any error on functions/methods might\n    raise any error which are subclass of ``Exception``.\n\n    .. note::\n        Decorator doesn't intend to catch ``BaseException``,\n        e.g. ``GeneratorExit``, ``KeyboardInterrupt``, ``SystemExit`` and others.\n\n    .. warning::\n        Only for internal usage, this decorator might be changed or removed in the future\n        without any further notice.\n\n    :param return_value: Return value if decorated function/method raise any ``Exception``.\n    :meta: private\n    "

    def decorator(func: Callable[PS, RT]) -> Callable[PS, RT]:
        if False:
            return 10

        @wraps(func)
        def wrapper(*args, **kwargs) -> RT:
            if False:
                print('Hello World!')
            try:
                return func(*args, **kwargs)
            except Exception:
                log.debug('Encountered error during execution function/method %r', func.__name__, exc_info=True)
                return return_value
        return wrapper
    return decorator