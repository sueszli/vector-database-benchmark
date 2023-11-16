"""
 >>> from andreas_beyer_ext import *
 >>> b=B()
 >>> a=b.get() # let b create an A
 >>> a2=b.get()
 >>> assert id(a) == id(a2)
"""

def run(args=None):
    if False:
        for i in range(10):
            print('nop')
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