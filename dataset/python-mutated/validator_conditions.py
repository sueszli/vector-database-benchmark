"""
Implement basic assertions to be used in assertion action
"""
from __future__ import annotations

def eq(value, other):
    if False:
        while True:
            i = 10
    'Equal'
    return value == other

def ne(value, other):
    if False:
        print('Hello World!')
    'Not equal'
    return value != other

def gt(value, other):
    if False:
        while True:
            i = 10
    'Greater than'
    return value > other

def lt(value, other):
    if False:
        while True:
            i = 10
    'Lower than'
    return value < other

def gte(value, other):
    if False:
        i = 10
        return i + 15
    'Greater than or equal'
    return value >= other

def lte(value, other):
    if False:
        while True:
            i = 10
    'Lower than or equal'
    return value <= other

def identity(value, other):
    if False:
        return 10
    'Identity check using ID'
    return value is other

def is_type_of(value, other):
    if False:
        return 10
    'Type check'
    return isinstance(value, other)

def is_in(value, other):
    if False:
        print('Hello World!')
    'Existence'
    return value in other

def is_not_in(value, other):
    if False:
        for i in range(10):
            print('nop')
    'Inexistence'
    return value not in other

def cont(value, other):
    if False:
        print('Hello World!')
    'Contains'
    return other in value

def len_eq(value, other):
    if False:
        while True:
            i = 10
    'Length Equal'
    return len(value) == other

def len_ne(value, other):
    if False:
        i = 10
        return i + 15
    'Length Not equal'
    return len(value) != other

def len_min(value, other):
    if False:
        return 10
    'Minimum length'
    return len(value) >= other

def len_max(value, other):
    if False:
        i = 10
        return i + 15
    'Maximum length'
    return len(value) <= other

def startswith(value, term):
    if False:
        while True:
            i = 10
    'returns value.startswith(term) result'
    return value.startswith(term)

def endswith(value, term):
    if False:
        i = 10
        return i + 15
    'returns value.endswith(term) result'
    return value.endswith(term)