from __future__ import print_function
">>> import pickle2_ext\n    >>> import pickle\n    >>> pickle2_ext.world.__module__\n    'pickle2_ext'\n    >>> pickle2_ext.world.__safe_for_unpickling__\n    1\n    >>> pickle2_ext.world.__name__\n    'world'\n    >>> pickle2_ext.world('Hello').__reduce__()\n    (<class 'pickle2_ext.world'>, ('Hello',), (0,))\n    >>> for number in (24, 42):\n    ...   wd = pickle2_ext.world('California')\n    ...   wd.set_secret_number(number)\n    ...   pstr = pickle.dumps(wd)\n    ...   wl = pickle.loads(pstr)\n    ...   print(wd.greet(), wd.get_secret_number())\n    ...   print(wl.greet(), wl.get_secret_number())\n    Hello from California! 24\n    Hello from California! 24\n    Hello from California! 42\n    Hello from California! 0\n\n# Now show that the __dict__ is not taken care of.\n    >>> wd = pickle2_ext.world('California')\n    >>> wd.x = 1\n    >>> wd.__dict__\n    {'x': 1}\n    >>> try: pstr = pickle.dumps(wd)\n    ... except RuntimeError as err: print(err)\n    ...\n    Incomplete pickle support (__getstate_manages_dict__ not set)\n"

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