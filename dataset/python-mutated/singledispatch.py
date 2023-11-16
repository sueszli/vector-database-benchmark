import inspect
from functools import singledispatch

def assign_signature(func):
    if False:
        return 10
    func.__signature__ = inspect.signature(func)
    return func

@singledispatch
def func(arg, kwarg=None):
    if False:
        for i in range(10):
            print('nop')
    'A function for general use.'
    pass

@func.register(int)
@func.register(float)
def _func_int(arg, kwarg=None):
    if False:
        i = 10
        return i + 15
    'A function for int.'
    pass

@func.register(str)
@assign_signature
def _func_str(arg, kwarg=None):
    if False:
        print('Hello World!')
    'A function for str.'
    pass

@func.register
def _func_dict(arg: dict, kwarg=None):
    if False:
        i = 10
        return i + 15
    'A function for dict.'
    pass