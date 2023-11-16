"""
>>> from exception_translator_ext import *
>>> try:
...     throw_error();
... except RuntimeError as x:
...     print(x)
... else:
...     print('Expected a RuntimeError!')
!!!error!!!
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