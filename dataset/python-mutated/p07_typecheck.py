"""
Topic: 函数参数强制类型检查
Desc : 
"""
from inspect import signature
from functools import wraps

def typeassert(*ty_args, **ty_kwargs):
    if False:
        i = 10
        return i + 15

    def decorate(func):
        if False:
            return 10
        if not __debug__:
            return func
        sig = signature(func)
        bound_types = sig.bind_partial(*ty_args, **ty_kwargs).arguments

        @wraps(func)
        def wrapper(*args, **kwargs):
            if False:
                return 10
            bound_values = sig.bind(*args, **kwargs)
            for (name, value) in bound_values.arguments.items():
                if name in bound_types:
                    if not isinstance(value, bound_types[name]):
                        raise TypeError('Argument {} must be {}'.format(name, bound_types[name]))
            return func(*args, **kwargs)
        return wrapper
    return decorate

@typeassert(int, int)
def add(x, y):
    if False:
        return 10
    return x + y
add(2, 3)

@typeassert(int, list)
def bar(x, items=None):
    if False:
        return 10
    if items is None:
        items = []
    items.append(x)
    return items
bar(2)
bar(2, [1, 2, 3])