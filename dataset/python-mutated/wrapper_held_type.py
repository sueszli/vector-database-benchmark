"""
>>> from wrapper_held_type_ext import *
>>> d = data()
>>> print(d.id())
42
>>> do_nothing( d )
>>> print(d.id())
42
>>> d = create_data()
>>> print(d.id())
42
>>> do_nothing( d )
>>> print(d.id())
42
"""

def run(args=None):
    if False:
        while True:
            i = 10
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