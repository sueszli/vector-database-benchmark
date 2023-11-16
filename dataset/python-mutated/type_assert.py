from functools import wraps
from inspect import signature

def type_assert(*ty_args, **ty_kwargs):
    if False:
        i = 10
        return i + 15
    "a decorator which is used to check the types of arguments in a function or class\n    Examples:\n        >>> @type_assert(str)\n        ... def main(a: str, b: list):\n        ...     print(a, b)\n        >>> main(1)\n        Argument a must be a str\n\n        >>> @type_assert(str, (int, str))\n        ... def main(a: str, b: int | str):\n        ...     print(a, b)\n        >>> main('1', [1])\n        Argument b must be (<class 'int'>, <class 'str'>)\n\n        >>> @type_assert(str, (int, str))\n        ... class A:\n        ...     def __init__(self, a: str, b: int | str)\n        ...         print(a, b)\n        >>> a = A('1', [1])\n        Argument b must be (<class 'int'>, <class 'str'>)\n    "

    def decorate(func):
        if False:
            while True:
                i = 10
        if not __debug__:
            return func
        sig = signature(func)
        bound_types = sig.bind_partial(*ty_args, **ty_kwargs).arguments

        @wraps(func)
        def wrapper(*args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            bound_values = sig.bind(*args, **kwargs)
            for (name, value) in bound_values.arguments.items():
                if name in bound_types:
                    if not isinstance(value, bound_types[name]):
                        raise TypeError('Argument {} must be {}'.format(name, bound_types[name]))
            return func(*args, **kwargs)
        return wrapper
    return decorate