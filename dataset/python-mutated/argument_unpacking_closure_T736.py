def default_args_for_closure(a=1, b=2):
    if False:
        while True:
            i = 10
    '\n    >>> default_args_for_closure()()\n    (1, 2)\n    >>> default_args_for_closure(1, 2)()\n    (1, 2)\n    >>> default_args_for_closure(2)()\n    (2, 2)\n    >>> default_args_for_closure(8,9)()\n    (8, 9)\n    >>> default_args_for_closure(7, b=6)()\n    (7, 6)\n    >>> default_args_for_closure(a=5, b=4)()\n    (5, 4)\n    >>> default_args_for_closure(b=5, a=6)()\n    (6, 5)\n    '

    def func():
        if False:
            while True:
                i = 10
        return (a, b)
    return func