from .autoparse import autoparse
from .automain import automain
try:
    from .autoasync import autoasync
except ImportError:
    pass

def autocommand(module, *, description=None, epilog=None, add_nos=False, parser=None, loop=None, forever=False, pass_loop=False):
    if False:
        i = 10
        return i + 15
    if callable(module):
        raise TypeError('autocommand requires a module name argument')

    def autocommand_decorator(func):
        if False:
            for i in range(10):
                print('nop')
        if loop is not None or forever or pass_loop:
            func = autoasync(func, loop=None if loop is True else loop, pass_loop=pass_loop, forever=forever)
        func = autoparse(func, description=description, epilog=epilog, add_nos=add_nos, parser=parser)
        func = automain(module)(func)
        return func
    return autocommand_decorator