from helper import pretty

def lib_mandatory_named_varargs_and_kwargs(a, b='default', *args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    return pretty(a, b, *args, **kwargs)

def lib_kwargs(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    return pretty(**kwargs)

def lib_mandatory_named_and_kwargs(a, b=2, **kwargs):
    if False:
        print('Hello World!')
    return pretty(a, b, **kwargs)

def lib_mandatory_named_and_varargs(a, b='default', *args):
    if False:
        for i in range(10):
            print('nop')
    return pretty(a, b, *args)

def lib_mandatory_and_named(a, b='default'):
    if False:
        return 10
    return pretty(a, b)

def lib_mandatory_and_named_2(a, b='default', c='default'):
    if False:
        print('Hello World!')
    return pretty(a, b, c)