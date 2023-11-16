"""
>>> from voidptr_ext import *


    Check for correct conversion

>>> use(get())

    Check that None is converted to a NULL void pointer

>>> useany(get())
1
>>> useany(None)
0

    Check that we don't lose type information by converting NULL
    opaque pointers to None

>>> assert getnull() is None
>>> useany(getnull())
0

   Check that there is no conversion from integers ...

>>> try: use(0)
... except TypeError: pass
... else: print('expected a TypeError')

   ... and from strings to opaque objects

>>> try: use("")
... except TypeError: pass
... else: print('expected a TypeError')
"""

def run(args=None):
    if False:
        i = 10
        return i + 15
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