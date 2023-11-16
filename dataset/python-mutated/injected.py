"""
>>> from injected_ext import *
>>> X(3,5).value() - (3+5)
0
>>> X(a=3,b=5,c=7).value() - (3*5*7)
0
>>> X().value()
1000
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