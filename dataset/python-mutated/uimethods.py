import importlib
from functools import wraps
import wifiphisher.common.constants

def uimethod(func):
    if False:
        while True:
            i = 10

    def _decorator(data, *args, **kwargs):
        if False:
            return 10
        response = func(data, *args, **kwargs)
        return response
    func.is_uimethod = True
    return wraps(func)(_decorator)