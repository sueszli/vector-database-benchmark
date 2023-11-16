"""
>>> from pytype_function_ext import *

>>> print(func.__doc__.splitlines()[1])
func( (A)arg1) -> A :

>>> print(func.__module__)
pytype_function_ext

>>> print(func.__name__)
func
"""

def run(args=None):
    if False:
        return 10
    import sys
    import doctest
    if args is not None:
        sys.argv = args
    return doctest.testmod(sys.modules.get(__name__))
if __name__ == '__main__':
    print('running...')
    import sys
    status = run()[0]
    if status == 0:
        print('Done.')
    sys.exit(status)