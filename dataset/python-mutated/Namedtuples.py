""" This module is only an abstraction of namedtuple.

It works around bugs present in some version of Python, and provides extra
methods like "asDict".
"""
from collections import namedtuple

def makeNamedtupleClass(name, element_names):
    if False:
        i = 10
        return i + 15
    namedtuple_class = namedtuple(name, element_names)

    class DynamicNamedtuple(namedtuple_class):
        __qualname__ = name
        __slots__ = ()

        def asDict(self):
            if False:
                return 10
            return self._asdict()
    DynamicNamedtuple.__name__ = name
    return DynamicNamedtuple