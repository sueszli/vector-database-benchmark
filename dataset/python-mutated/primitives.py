__all__ = ['isnone', 'notnone', 'inc', 'dec', 'even', 'odd']

class EmptyType:

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return 'EMPTY'
EMPTY = EmptyType()

def isnone(x):
    if False:
        return 10
    return x is None

def notnone(x):
    if False:
        i = 10
        return i + 15
    return x is not None

def inc(x):
    if False:
        return 10
    return x + 1

def dec(x):
    if False:
        return 10
    return x - 1

def even(x):
    if False:
        print('Hello World!')
    return x % 2 == 0

def odd(x):
    if False:
        i = 10
        return i + 15
    return x % 2 == 1