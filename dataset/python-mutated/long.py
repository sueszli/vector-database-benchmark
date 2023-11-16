import sys
if sys.version_info.major >= 3:
    long = int
"\n>>> from long_ext import *\n>>> print(new_long())\n0\n>>> print(longify(42))\n42\n>>> print(longify_string('300'))\n300\n>>> is_long(long(20))\n'yes'\n>>> is_long('20')\n0\n\n>>> x = Y(long(4294967295))\n"

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