"""
>>> from implicit_ext import *
>>> x_value(X(42))
42
>>> x_value(42)
42
>>> x = make_x(X(42))
>>> x.value()
42
>>> try: make_x('fool')
... except TypeError: pass
... else: print('no error')

>>> print(x_value.__doc__.splitlines()[1])
x_value( (X)arg1) -> int :

>>> print(make_x.__doc__.splitlines()[1])
make_x( (object)arg1) -> X :

>>> print(X.value.__doc__.splitlines()[1])
value( (X)arg1) -> int :

>>> print(X.set.__doc__.splitlines()[1])
set( (X)arg1, (object)arg2) -> None :

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