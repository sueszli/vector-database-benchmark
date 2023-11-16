import inspect
from .platform import PYPY

def is_init(method):
    if False:
        return 10
    if not method:
        return False
    if PYPY:
        return method is not object.__init__
    return inspect.isfunction(method)