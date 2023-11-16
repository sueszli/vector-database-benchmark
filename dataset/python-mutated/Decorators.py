import functools
import re
from typing import Callable
SEMANTIC_VERSION_REGEX = re.compile('^[0-9]+\\.[0-9]+(\\.[0-9]+)?$')

def api(since_version: str) -> Callable:
    if False:
        for i in range(10):
            print('nop')
    "Decorator for functions that belong to a set of APIs. For now, this should only be used for officially supported\n\n    APIs, meaning that those APIs should be versioned and maintained.\n\n    :param since_version: The earliest version since when this API becomes supported. This means that since this version,\n        this API function is supposed to behave the same. This parameter is not used. It's just a\n        documentation.\n    "
    if not SEMANTIC_VERSION_REGEX.fullmatch(since_version):
        raise ValueError('API since_version [%s] is not a semantic version.' % since_version)

    def api_decorator(function):
        if False:
            for i in range(10):
                print('nop')

        @functools.wraps(function)
        def api_wrapper(*args, **kwargs):
            if False:
                i = 10
                return i + 15
            return function(*args, **kwargs)
        return api_wrapper
    return api_decorator