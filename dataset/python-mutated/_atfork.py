import sys
__all__ = ['register_after_fork']
if sys.platform == 'win32':
    import multiprocessing.util as _util

    def _register(func):
        if False:
            print('Hello World!')

        def wrapper(arg):
            if False:
                for i in range(10):
                    print('nop')
            func()
        _util.register_after_fork(_register, wrapper)
else:
    import os

    def _register(func):
        if False:
            for i in range(10):
                print('nop')
        os.register_at_fork(after_in_child=func)

def register_after_fork(func):
    if False:
        print('Hello World!')
    'Register a callable to be executed in the child process after a fork.\n\n    Note:\n        In python < 3.7 this will only work with processes created using the\n        ``multiprocessing`` module. In python >= 3.7 it also works with\n        ``os.fork()``.\n\n    Args:\n        func (function): Function taking no arguments to be called in the child after fork\n\n    '
    _register(func)