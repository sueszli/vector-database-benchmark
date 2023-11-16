from __future__ import annotations
import math

def issubset(a, b):
    if False:
        i = 10
        return i + 15
    return set(a) <= set(b)

def issuperset(a, b):
    if False:
        return 10
    return set(a) >= set(b)

def isnotanumber(x):
    if False:
        print('Hello World!')
    try:
        return math.isnan(x)
    except TypeError:
        return False

def contains(seq, value):
    if False:
        i = 10
        return i + 15
    'Opposite of the ``in`` test, allowing use as a test in filters like ``selectattr``\n\n    .. versionadded:: 2.8\n    '
    return value in seq

class TestModule:
    """ Ansible math jinja2 tests """

    def tests(self):
        if False:
            return 10
        return {'subset': issubset, 'issubset': issubset, 'superset': issuperset, 'issuperset': issuperset, 'contains': contains, 'nan': isnotanumber, 'isnan': isnotanumber}