"""
>>> X = make_class_with_new(cynew)
>>> X.__new__ is cynew
True
>>> X().__new__ is cynew
True
>>> def pynew(cls): return object.__new__(cls)
>>> X = make_class_with_new(pynew)
>>> X.__new__ is pynew
True
>>> X().__new__ is pynew
True
"""

def make_class_with_new(n):
    if False:
        return 10

    class X(object):
        __new__ = n
    return X

def cynew(cls):
    if False:
        i = 10
        return i + 15
    return object.__new__(cls)