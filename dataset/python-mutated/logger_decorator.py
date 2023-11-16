"""
Module contains the functions designed for the enable/disable of logging.

``enable_logging`` is used for decorating individual Modin functions or classes.
"""
from functools import wraps
from logging import Logger
from types import FunctionType, MethodType
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union
from modin.config import LogMode
from .config import get_logger
_MODIN_LOGGER_NOWRAP = '__modin_logging_nowrap__'

def disable_logging(func: Callable) -> Callable:
    if False:
        i = 10
        return i + 15
    '\n    Disable logging of one particular function. Useful for decorated classes.\n\n    Parameters\n    ----------\n    func : callable\n        A method in a logger-decorated class for which logging should be disabled.\n\n    Returns\n    -------\n    func\n        A function with logging disabled.\n    '
    setattr(func, _MODIN_LOGGER_NOWRAP, True)
    return func

def enable_logging(modin_layer: Union[str, Callable, Type]='PANDAS-API', name: Optional[str]=None, log_level: str='info') -> Callable:
    if False:
        for i in range(10):
            print('nop')
    '\n    Log Decorator used on specific Modin functions or classes.\n\n    Parameters\n    ----------\n    modin_layer : str or object to decorate, default: "PANDAS-API"\n        Specified by the logger (e.g. PANDAS-API).\n        If it\'s an object to decorate, call logger_decorator() on it with default arguments.\n    name : str, optional\n        The name of the object the decorator is being applied to.\n        Composed from the decorated object name if not specified.\n    log_level : str, default: "info"\n        The log level (INFO, DEBUG, WARNING, etc.).\n\n    Returns\n    -------\n    func\n        A decorator function.\n    '
    if not isinstance(modin_layer, str):
        return enable_logging()(modin_layer)
    log_level = log_level.lower()
    assert hasattr(Logger, log_level.lower()), f'Invalid log level: {log_level}'

    def decorator(obj: Any) -> Any:
        if False:
            return 10
        'Decorate function or class to add logs to Modin API function(s).'
        if isinstance(obj, type):
            seen: Dict[Any, Any] = {}
            for (attr_name, attr_value) in vars(obj).items():
                if isinstance(attr_value, (FunctionType, MethodType, classmethod, staticmethod)) and (not hasattr(attr_value, _MODIN_LOGGER_NOWRAP)):
                    try:
                        wrapped = seen[attr_value]
                    except KeyError:
                        wrapped = seen[attr_value] = enable_logging(modin_layer, f'{name or obj.__name__}.{attr_name}', log_level)(attr_value)
                    setattr(obj, attr_name, wrapped)
            return obj
        elif isinstance(obj, classmethod):
            return classmethod(decorator(obj.__func__))
        elif isinstance(obj, staticmethod):
            return staticmethod(decorator(obj.__func__))
        assert isinstance(modin_layer, str), 'modin_layer is somehow not a string!'
        start_line = f'START::{modin_layer.upper()}::{name or obj.__name__}'
        stop_line = f'STOP::{modin_layer.upper()}::{name or obj.__name__}'

        @wraps(obj)
        def run_and_log(*args: Tuple, **kwargs: Dict) -> Any:
            if False:
                return 10
            '\n            Compute function with logging if Modin logging is enabled.\n\n            Parameters\n            ----------\n            *args : tuple\n                The function arguments.\n            **kwargs : dict\n                The function keyword arguments.\n\n            Returns\n            -------\n            Any\n            '
            if LogMode.get() == 'disable':
                return obj(*args, **kwargs)
            logger = get_logger()
            logger_level = getattr(logger, log_level)
            logger_level(start_line)
            try:
                result = obj(*args, **kwargs)
            except BaseException as e:
                if not hasattr(e, '_modin_logged'):
                    get_logger('modin.logger.errors').exception(stop_line, stack_info=True)
                    e._modin_logged = True
                raise
            finally:
                logger_level(stop_line)
            return result
        return disable_logging(run_and_log)
    return decorator