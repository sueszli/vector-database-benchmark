"""
Violation:

Prefer TypeError when relevant.
"""

def incorrect_basic(some_arg):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(some_arg, int):
        pass
    else:
        raise Exception('...')

def incorrect_multiple_type_check(some_arg):
    if False:
        while True:
            i = 10
    if isinstance(some_arg, (int, str)):
        pass
    else:
        raise Exception('...')

class MyClass:
    pass

def incorrect_with_issubclass(some_arg):
    if False:
        print('Hello World!')
    if issubclass(some_arg, MyClass):
        pass
    else:
        raise Exception('...')

def incorrect_with_callable(some_arg):
    if False:
        for i in range(10):
            print('nop')
    if callable(some_arg):
        pass
    else:
        raise Exception('...')

def incorrect_ArithmeticError(some_arg):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(some_arg, int):
        pass
    else:
        raise ArithmeticError('...')

def incorrect_AssertionError(some_arg):
    if False:
        while True:
            i = 10
    if isinstance(some_arg, int):
        pass
    else:
        raise AssertionError('...')

def incorrect_AttributeError(some_arg):
    if False:
        print('Hello World!')
    if isinstance(some_arg, int):
        pass
    else:
        raise AttributeError('...')

def incorrect_BufferError(some_arg):
    if False:
        i = 10
        return i + 15
    if isinstance(some_arg, int):
        pass
    else:
        raise BufferError

def incorrect_EOFError(some_arg):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(some_arg, int):
        pass
    else:
        raise EOFError('...')

def incorrect_ImportError(some_arg):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(some_arg, int):
        pass
    else:
        raise ImportError('...')

def incorrect_LookupError(some_arg):
    if False:
        while True:
            i = 10
    if isinstance(some_arg, int):
        pass
    else:
        raise LookupError('...')

def incorrect_MemoryError(some_arg):
    if False:
        print('Hello World!')
    if isinstance(some_arg, int):
        pass
    else:
        raise MemoryError('...')

def incorrect_NameError(some_arg):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(some_arg, int):
        pass
    else:
        raise NameError('...')

def incorrect_ReferenceError(some_arg):
    if False:
        return 10
    if isinstance(some_arg, int):
        pass
    else:
        raise ReferenceError('...')

def incorrect_RuntimeError(some_arg):
    if False:
        i = 10
        return i + 15
    if isinstance(some_arg, int):
        pass
    else:
        raise RuntimeError('...')

def incorrect_SyntaxError(some_arg):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(some_arg, int):
        pass
    else:
        raise SyntaxError('...')

def incorrect_SystemError(some_arg):
    if False:
        print('Hello World!')
    if isinstance(some_arg, int):
        pass
    else:
        raise SystemError('...')

def incorrect_ValueError(some_arg):
    if False:
        while True:
            i = 10
    if isinstance(some_arg, int):
        pass
    else:
        raise ValueError('...')

def incorrect_not_operator_isinstance(some_arg):
    if False:
        for i in range(10):
            print('nop')
    if not isinstance(some_arg):
        pass
    else:
        raise Exception('...')

def incorrect_and_operator_isinstance(arg1, arg2):
    if False:
        return 10
    if isinstance(some_arg) and isinstance(arg2):
        pass
    else:
        raise Exception('...')

def incorrect_or_operator_isinstance(arg1, arg2):
    if False:
        print('Hello World!')
    if isinstance(some_arg) or isinstance(arg2):
        pass
    else:
        raise Exception('...')

def incorrect_multiple_operators_isinstance(arg1, arg2, arg3):
    if False:
        i = 10
        return i + 15
    if not isinstance(arg1) and isinstance(arg2) or isinstance(arg3):
        pass
    else:
        raise Exception('...')

def incorrect_not_operator_callable(some_arg):
    if False:
        for i in range(10):
            print('nop')
    if not callable(some_arg):
        pass
    else:
        raise Exception('...')

def incorrect_and_operator_callable(arg1, arg2):
    if False:
        return 10
    if callable(some_arg) and callable(arg2):
        pass
    else:
        raise Exception('...')

def incorrect_or_operator_callable(arg1, arg2):
    if False:
        for i in range(10):
            print('nop')
    if callable(some_arg) or callable(arg2):
        pass
    else:
        raise Exception('...')

def incorrect_multiple_operators_callable(arg1, arg2, arg3):
    if False:
        i = 10
        return i + 15
    if not callable(arg1) and callable(arg2) or callable(arg3):
        pass
    else:
        raise Exception('...')

def incorrect_not_operator_issubclass(some_arg):
    if False:
        return 10
    if not issubclass(some_arg):
        pass
    else:
        raise Exception('...')

def incorrect_and_operator_issubclass(arg1, arg2):
    if False:
        for i in range(10):
            print('nop')
    if issubclass(some_arg) and issubclass(arg2):
        pass
    else:
        raise Exception('...')

def incorrect_or_operator_issubclass(arg1, arg2):
    if False:
        while True:
            i = 10
    if issubclass(some_arg) or issubclass(arg2):
        pass
    else:
        raise Exception('...')

def incorrect_multiple_operators_issubclass(arg1, arg2, arg3):
    if False:
        while True:
            i = 10
    if not issubclass(arg1) and issubclass(arg2) or issubclass(arg3):
        pass
    else:
        raise Exception('...')

def incorrect_multi_conditional(arg1, arg2):
    if False:
        i = 10
        return i + 15
    if isinstance(arg1, int):
        pass
    elif isinstance(arg2, int):
        raise Exception('...')

def multiple_is_instance_checks(some_arg):
    if False:
        return 10
    if isinstance(some_arg, str):
        pass
    elif isinstance(some_arg, int):
        pass
    else:
        raise Exception('...')

class MyCustomTypeValidation(Exception):
    pass

def correct_custom_exception(some_arg):
    if False:
        print('Hello World!')
    if isinstance(some_arg, int):
        pass
    else:
        raise MyCustomTypeValidation('...')

def correct_complex_conditional(val):
    if False:
        i = 10
        return i + 15
    if val is not None and (not isinstance(val, int) or val < 0):
        raise ValueError(...)

def correct_multi_conditional(some_arg):
    if False:
        print('Hello World!')
    if some_arg == 3:
        pass
    elif isinstance(some_arg, int):
        pass
    else:
        raise Exception('...')

def correct_should_ignore(some_arg):
    if False:
        i = 10
        return i + 15
    if isinstance(some_arg, int):
        pass
    else:
        raise TypeError('...')

def check_body(some_args):
    if False:
        while True:
            i = 10
    if isinstance(some_args, int):
        raise ValueError('...')

def check_body_correct(some_args):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(some_args, int):
        raise TypeError('...')

def multiple_elifs(some_args):
    if False:
        i = 10
        return i + 15
    if not isinstance(some_args, int):
        raise ValueError('...')
    elif some_args < 3:
        raise ValueError('...')
    elif some_args > 10:
        raise ValueError('...')
    else:
        pass

def multiple_ifs(some_args):
    if False:
        for i in range(10):
            print('nop')
    if not isinstance(some_args, int):
        raise ValueError('...')
    elif some_args < 3:
        raise ValueError('...')
    elif some_args > 10:
        raise ValueError('...')
    else:
        pass

def else_body(obj):
    if False:
        while True:
            i = 10
    if isinstance(obj, datetime.timedelta):
        return 'TimeDelta'
    elif isinstance(obj, relativedelta.relativedelta):
        return 'RelativeDelta'
    elif isinstance(obj, CronExpression):
        return 'CronExpression'
    else:
        raise Exception(f'Unknown object type: {obj.__class__.__name__}')

def early_return():
    if False:
        print('Hello World!')
    if isinstance(this, some_type):
        if x in this:
            return
        raise ValueError(f'{this} has a problem')

def early_break():
    if False:
        for i in range(10):
            print('nop')
    for x in this:
        if isinstance(this, some_type):
            if x in this:
                break
            raise ValueError(f'{this} has a problem')

def early_continue():
    if False:
        print('Hello World!')
    for x in this:
        if isinstance(this, some_type):
            if x in this:
                continue
            raise ValueError(f'{this} has a problem')

def early_return_else():
    if False:
        i = 10
        return i + 15
    if isinstance(this, some_type):
        pass
    else:
        if x in this:
            return
        raise ValueError(f'{this} has a problem')