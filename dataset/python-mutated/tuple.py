from __future__ import print_function
'\n>>> from tuple_ext import *\n>>> def printer(*args):\n...     for x in args: print(x,)\n...     print(\'\')\n...\n>>> print(convert_to_tuple("this is a test string"))\n(\'t\', \'h\', \'i\', \'s\', \' \', \'i\', \'s\', \' \', \'a\', \' \', \'t\', \'e\', \'s\', \'t\', \' \', \'s\', \'t\', \'r\', \'i\', \'n\', \'g\')\n>>> t1 = convert_to_tuple("this is")\n>>> t2 = (1,2,3,4)\n>>> test_operators(t1,t2,printer) #doctest: +NORMALIZE_WHITESPACE\n(\'t\', \'h\', \'i\', \'s\', \' \', \'i\', \'s\', 1, 2, 3, 4)\n>>> make_tuple()\n()\n>>> make_tuple(42)\n(42,)\n>>> make_tuple(\'hello\', 42)\n(\'hello\', 42)\n'

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