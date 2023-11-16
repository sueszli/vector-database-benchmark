"""
Some utility function decorators
"""
from typing import Callable

def run_once(func: Callable) -> Callable:
    if False:
        for i in range(10):
            print('nop')
    '\n    Decorator to run func only at its first invocation.\n\n    Set func.has_run to False to manually re-run.\n    '

    def wrapper(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        ' Returned function wrapper. '
        if wrapper.has_run:
            return None
        wrapper.has_run = True
        return func(*args, **kwargs)
    wrapper.has_run = False
    return wrapper