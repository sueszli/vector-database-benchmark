"""
Python polyfills for common builtins.
"""

def all(iterator):
    if False:
        return 10
    for elem in iterator:
        if not elem:
            return False
    return True

def index(iterator, item, start=0, end=-1):
    if False:
        return 10
    for (i, elem) in enumerate(list(iterator))[start:end]:
        if item == elem:
            return i
    raise ValueError(f'{item} is not in {type(iterator)}')

def repeat(item, count):
    if False:
        while True:
            i = 10
    for i in range(count):
        yield item