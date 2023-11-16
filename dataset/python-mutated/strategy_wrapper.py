import logging
from copy import deepcopy
from functools import wraps
from typing import Any, Callable, TypeVar, cast
from freqtrade.exceptions import StrategyError
logger = logging.getLogger(__name__)
F = TypeVar('F', bound=Callable[..., Any])

def strategy_safe_wrapper(f: F, message: str='', default_retval=None, supress_error=False) -> F:
    if False:
        while True:
            i = 10
    "\n    Wrapper around user-provided methods and functions.\n    Caches all exceptions and returns either the default_retval (if it's not None) or raises\n    a StrategyError exception, which then needs to be handled by the calling method.\n    "

    @wraps(f)
    def wrapper(*args, **kwargs):
        if False:
            print('Hello World!')
        try:
            if 'trade' in kwargs:
                kwargs['trade'] = deepcopy(kwargs['trade'])
            return f(*args, **kwargs)
        except ValueError as error:
            logger.warning(f'{message}Strategy caused the following exception: {error}{f}')
            if default_retval is None and (not supress_error):
                raise StrategyError(str(error)) from error
            return default_retval
        except Exception as error:
            logger.exception(f'{message}Unexpected error {error} calling {f}')
            if default_retval is None and (not supress_error):
                raise StrategyError(str(error)) from error
            return default_retval
    return cast(F, wrapper)