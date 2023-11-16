def f(x, y, z):
    if False:
        i = 10
        return i + 15
    'Do something.\n\n    Args:\n        x: the value\n            with a hanging indent\n\n    Returns:\n        the value\n    '
    return x

def f(x, y, z):
    if False:
        print('Hello World!')
    'Do something.\n\n    Args:\n        x:\n            The whole thing has a hanging indent.\n\n    Returns:\n        the value\n    '
    return x

def f(x, y, z):
    if False:
        return 10
    'Do something.\n\n    Args:\n        x:\n            The whole thing has a hanging indent.\n\n    Returns: the value\n    '
    return x

def f(x, y, z):
    if False:
        i = 10
        return i + 15
    'Do something.\n\n    Args:\n        x: the value def\n            ghi\n\n    Returns:\n        the value\n    '
    return x

def f(x, y, z):
    if False:
        for i in range(10):
            print('nop')
    'Do something.\n\n    Args:\n        x: the value\n        z: A final argument\n\n    Returns:\n        the value\n    '
    return x

def f(x, y, z):
    if False:
        for i in range(10):
            print('nop')
    'Do something.\n\n    Args:\n        x: the value\n        z: A final argument\n\n    Returns: the value\n    '
    return x

def f(x, y, z):
    if False:
        i = 10
        return i + 15
    'Do something.\n\n    Args:\n        x: the value\n        z: A final argument\n    '
    return x

def f(x, *args, **kwargs):
    if False:
        i = 10
        return i + 15
    'Do something.\n\n    Args:\n        x: the value\n        *args: variable arguments\n        **kwargs: keyword arguments\n    '
    return x

def f(x, *args, **kwargs):
    if False:
        i = 10
        return i + 15
    'Do something.\n\n    Args:\n        *args: variable arguments\n        **kwargs: keyword arguments\n    '
    return x

def f(x, *args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Do something.\n\n    Args:\n        x: the value\n        **kwargs: keyword arguments\n    '
    return x

def f(x, *, y, z):
    if False:
        for i in range(10):
            print('nop')
    'Do something.\n\n    Args:\n        x: some first value\n\n    Keyword Args:\n        y (int): the other value\n        z (int): the last value\n\n    '
    return (x, y, z)

def f(x):
    if False:
        i = 10
        return i + 15
    'Do something with valid description.\n\n    Args:\n    ----\n        x: the value\n\n    Returns:\n    -------\n        the value\n    '
    return x

class Test:

    def f(self, /, arg1: int) -> None:
        if False:
            while True:
                i = 10
        '\n        Some beauty description.\n\n        Args:\n            arg1: some description of arg\n        '