""" This module contains code that helps in maintaining the Ckan codebase. """
from __future__ import annotations
import inspect
import time
import logging
import re
import warnings
from typing import Any, Callable, Optional, TypeVar, Union
from typing_extensions import ParamSpec
from ckan.exceptions import CkanDeprecationWarning
P = ParamSpec('P')
RT = TypeVar('RT')
log = logging.getLogger(__name__)

def deprecated(message: Optional[str]='', since: Optional[str]=None):
    if False:
        print('Hello World!')
    ' This is a decorator used to mark functions as deprecated.\n\n    It logs a warning when the function is called. If a message is\n    passed it is also logged, this can be useful to indicate for example\n    that a different function should be used instead.\n\n    Additionally an exception is raised if the functions docstring does\n    not contain the word `deprecated`.'

    def decorator(fn: Callable[P, RT]) -> Callable[P, RT]:
        if False:
            return 10
        if not fn.__doc__ or not re.search('\\bdeprecated\\b', fn.__doc__, re.IGNORECASE):
            raise Exception('Function %s() in module %s has been deprecated but this is not mentioned in the docstring. Please update the docstring for the function. It must include the word `deprecated`.' % (fn.__name__, fn.__module__))

        def wrapped(*args: P.args, **kw: P.kwargs) -> RT:
            if False:
                print('Hello World!')
            since_msg = f'since CKAN v{since}' if since else ''
            msg = 'Function %s() in module %s has been deprecated %s and will be removed in a later release of ckan. %s' % (fn.__name__, fn.__module__, since_msg, message)
            log.warning(msg)
            warnings.warn(msg, CkanDeprecationWarning, stacklevel=2)
            return fn(*args, **kw)
        return wrapped
    return decorator

def timer(params: Union[Callable[..., Any], list[str]]) -> Callable[..., Any]:
    if False:
        print('Hello World!')
    ' Decorator function for basic performance testing. It logs the time\n    taken to call a function.  It can either be used as a basic decorator or an\n    array of parameter names can be passed. If parameter names are passed then\n    the logging will include the value of the parameter if it is passed to the\n    function. '
    if callable(params):
        fn = params
        fn_name = '%s.%s' % (fn.__module__, fn.__name__)

        def wrapped(*args: Any, **kw: Any):
            if False:
                print('Hello World!')
            start = time.time()
            result = fn(*args, **kw)
            log.info('Timer: %s %.4f' % (fn_name, time.time() - start))
            return result
        return wrapped

    def decorator(fn: Callable[..., Any]):
        if False:
            while True:
                i = 10
        assert isinstance(params, list)
        args_info = inspect.getargspec(fn)
        params_data = []
        for param in params:
            if param in args_info.args:
                params_data.append((param, args_info.args.index(param)))
            else:
                params_data.append(param)
        fn_name = '%s.%s' % (fn.__module__, fn.__name__)

        def wrapped(*args: Any, **kw: Any):
            if False:
                return 10
            params = []
            for param in params_data:
                value = None
                if param[0] in kw:
                    value = kw[param[0]]
                elif len(param) != 1 and len(args) >= param[1]:
                    value = args[param[1]]
                else:
                    continue
                params.append(u'%s=%r' % (param[0], value))
            p = ', '.join(params)
            start = time.time()
            result = fn(*args, **kw)
            log.info('Timer: %s %.4f %s' % (fn_name, time.time() - start, p))
            return result
        return wrapped
    return decorator