from __future__ import print_function
">>> import pickle3_ext\n    >>> import pickle\n    >>> pickle3_ext.world.__module__\n    'pickle3_ext'\n    >>> pickle3_ext.world.__safe_for_unpickling__\n    1\n    >>> pickle3_ext.world.__getstate_manages_dict__\n    1\n    >>> pickle3_ext.world.__name__\n    'world'\n    >>> pickle3_ext.world('Hello').__reduce__()\n    (<class 'pickle3_ext.world'>, ('Hello',), ({}, 0))\n    >>> for number in (24, 42):\n    ...   wd = pickle3_ext.world('California')\n    ...   wd.set_secret_number(number)\n    ...   wd.x = 2 * number\n    ...   wd.y = 'y' * number\n    ...   wd.z = 3. * number\n    ...   pstr = pickle.dumps(wd)\n    ...   wl = pickle.loads(pstr)\n    ...   print(wd.greet(), wd.get_secret_number(), wd.x, wd.y, wd.z)\n    ...   print(wl.greet(), wl.get_secret_number(), wl.x, wl.y, wl.z)\n    Hello from California! 24 48 yyyyyyyyyyyyyyyyyyyyyyyy 72.0\n    Hello from California! 24 48 yyyyyyyyyyyyyyyyyyyyyyyy 72.0\n    Hello from California! 42 84 yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy 126.0\n    Hello from California! 0 84 yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy 126.0\n"

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