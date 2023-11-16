import functools
from inspect import signature
from .common_op_utils import _basic_validation
'\nCommon utilities to register ops on ShardedTensor\nand PartialTensor.\n'

def _register_op(op, func, op_table):
    if False:
        for i in range(10):
            print('nop')
    '\n    Performs basic validation and registers the provided op in the given\n    op_table.\n    '
    if len(signature(func).parameters) != 4:
        raise TypeError(f'Custom sharded op function expects signature: (types, args, kwargs, process_group), but received signature: {signature(func)}')
    op_table[op] = func

def _decorator_func(wrapped_func, op, op_table):
    if False:
        for i in range(10):
            print('nop')
    '\n    Decorator function to register the given ``op`` in the provided\n    ``op_table``\n    '

    @functools.wraps(wrapped_func)
    def wrapper(types, args, kwargs, process_group):
        if False:
            while True:
                i = 10
        _basic_validation(op, args, kwargs)
        return wrapped_func(types, args, kwargs, process_group)
    _register_op(op, wrapper, op_table)
    return wrapper