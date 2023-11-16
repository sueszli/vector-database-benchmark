def typechain(*args):
    if False:
        i = 10
        return i + 15
    '\n    Returns function which applies the first transformation it can from args\n    and returns transformed value, or the value itself if it is in args.\n\n    >>> function = typechain(int, \'a\', ord, None)\n    >>> function("10")\n    10\n    >>> function("b")\n    98\n    >>> function("a")\n    \'a\'\n    >>> function(int)\n    <class \'int\'>\n    >>> function(None) is None\n    True\n    >>> function("str")\n    Traceback (most recent call last):\n        ...\n    ValueError: Couldn\'t convert value \'str\' to any specified type or find it in specified values.\n\n    :raises TypeError:  Raises when either no functions are specified for\n                        checking.\n    '
    if len(args) == 0:
        raise TypeError('No arguments were provided.')

    def annotation(value):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns value either transformed with one of the function in args, or\n        casted to one of types in args, or the value itself if it is in the\n        args.\n\n        :raises ValueError: Raises when cannot transform value in any one of\n                            specified ways.\n        '
        for arg in args:
            if value == arg:
                return value
            if isinstance(arg, type) and isinstance(value, arg):
                return value
            try:
                return arg(value)
            except (ValueError, TypeError):
                pass
        raise ValueError(f"Couldn't convert value {value!r} to any specified type or find it in specified values.")
    return annotation