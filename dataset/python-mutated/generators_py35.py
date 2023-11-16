from __future__ import generator_stop

def with_outer_raising(*args):
    if False:
        return 10
    '\n    >>> x = with_outer_raising(1, 2, 3)\n    >>> try:\n    ...     list(x())\n    ... except RuntimeError:\n    ...     print("OK!")\n    ... else:\n    ...     print("NOT RAISED!")\n    OK!\n    '

    def generator():
        if False:
            print('Hello World!')
        for i in args:
            yield i
        raise StopIteration
    return generator

def anno_gen(x: 'int') -> 'float':
    if False:
        while True:
            i = 10
    '\n    >>> gen = anno_gen(2)\n    >>> next(gen)\n    2.0\n    >>> ret, arg = sorted(anno_gen.__annotations__.items())\n    >>> print(ret[0]); print(str(ret[1]).strip("\'"))  # strip makes it pass with/without PEP563\n    return\n    float\n    >>> print(arg[0]); print(str(arg[1]).strip("\'"))\n    x\n    int\n    '
    yield float(x)