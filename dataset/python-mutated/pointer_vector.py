"""

>>> import pointer_vector_ext
>>> d = pointer_vector_ext.DoesSomething()
>>> lst = d.returnList()
>>> lst[0].f();
'harru'

"""

def run(args=None):
    if False:
        print('Hello World!')
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