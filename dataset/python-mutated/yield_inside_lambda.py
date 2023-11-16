def test_inside_lambda():
    if False:
        print('Hello World!')
    '\n    >>> obj = test_inside_lambda()()\n    >>> next(obj)\n    1\n    >>> next(obj)\n    2\n    >>> try: next(obj)\n    ... except StopIteration: pass\n    '
    return lambda : ((yield 1), (yield 2))