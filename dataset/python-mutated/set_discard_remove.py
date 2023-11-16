def set_discard():
    if False:
        for i in range(10):
            print('nop')
    '\n    >>> sorted(set_discard())\n    [1, 2]\n    '
    s = set([1, 2, 3])
    s.discard(3)
    return s

def set_discard_missing():
    if False:
        return 10
    '\n    >>> sorted(set_discard_missing())\n    [1, 2, 3]\n    '
    s = set([1, 2, 3])
    s.discard(4)
    return s

def set_discard_set():
    if False:
        print('Hello World!')
    '\n    >>> s = set_discard_set()\n    >>> len(s)\n    1\n    >>> sorted(s.pop())\n    [1, 2]\n    '
    s = set([frozenset([1, 2]), frozenset([2, 3])])
    s.discard(set([2, 3]))
    return s

def set_remove():
    if False:
        i = 10
        return i + 15
    '\n    >>> sorted(set_remove())\n    [1, 2]\n    '
    s = set([1, 2, 3])
    s.remove(3)
    return s

def set_remove_missing():
    if False:
        print('Hello World!')
    '\n    >>> sorted(set_remove_missing())\n    Traceback (most recent call last):\n    KeyError: 4\n    '
    s = set([1, 2, 3])
    s.remove(4)
    return s

def set_remove_set():
    if False:
        while True:
            i = 10
    '\n    >>> s = set_remove_set()\n    >>> len(s)\n    1\n    >>> sorted(s.pop())\n    [1, 2]\n    '
    s = set([frozenset([1, 2]), frozenset([2, 3])])
    s.remove(set([2, 3]))
    return s