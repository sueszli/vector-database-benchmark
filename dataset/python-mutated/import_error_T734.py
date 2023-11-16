def test_import_error():
    if False:
        print('Hello World!')
    '\n    >>> test_import_error()   # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    ImportError: cannot import name ...xxx...\n    '
    from sys import xxx