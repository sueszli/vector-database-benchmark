def f(**kwargs):
    if False:
        while True:
            i = 10
    return sorted(kwargs.items())

def test_call(kwargs):
    if False:
        print('Hello World!')
    "\n    >>> kwargs = {'b' : 2}\n    >>> f(a=1, **kwargs)\n    [('a', 1), ('b', 2)]\n    >>> test_call(kwargs)\n    [('a', 1), ('b', 2)]\n\n    >>> kwargs = {'a' : 2}\n    >>> f(a=1, **kwargs)    # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    TypeError: ...got multiple values for keyword argument 'a'\n\n    >>> test_call(kwargs)   # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    TypeError: ...got multiple values for keyword argument 'a'\n    "
    return f(a=1, **kwargs)