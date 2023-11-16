"""Tool to name unnamed arguments."""
import functools

def name_args(mapping, skip=0):
    if False:
        print('Hello World!')
    "Decorator to convert unnamed arguments to named ones.\n\n    Can be used to deprecate old signatures of a function, e.g.\n\n    .. code-block::\n\n        old_f(a: TypeA, b: TypeB, c: TypeC)\n        new_f(a: TypeA, d: TypeD, b: TypeB=None, c: TypeC=None)\n\n    Then, to support the old signature this decorator can be used as\n\n    .. code-block::\n\n        @name_args([\n            ('a'),  # stays the same\n            ('d', {TypeB: 'b'}),  # if arg is of type TypeB, call if 'b' else 'd'\n            ('b', {TypeC: 'c'})\n        ])\n        def new_f(a: TypeA, d: TypeD, b: TypeB=None, c: TypeC=None):\n            if b is not None:\n                # raise warning, this is deprecated!\n            if c is not None:\n                # raise warning, this is deprecated!\n\n    "

    def decorator(func):
        if False:
            for i in range(10):
                print('nop')

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            for (arg, replacement) in zip(args[skip:], mapping):
                default_name = replacement[0]
                if len(replacement) == 1:
                    if default_name in kwargs:
                        raise ValueError(f'Name collapse on {default_name}')
                    kwargs[default_name] = arg
                else:
                    name = None
                    for (special_type, special_name) in replacement[1].items():
                        if isinstance(arg, special_type):
                            name = special_name
                            break
                    if name is None:
                        name = default_name
                    if name in kwargs:
                        raise ValueError(f'Name collapse on {default_name}')
                    kwargs[name] = arg
            return func(*args[:skip], **kwargs)
        return wrapper
    return decorator