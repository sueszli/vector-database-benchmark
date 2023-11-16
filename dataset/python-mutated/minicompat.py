"""Python version compatibility support for minidom.

This module contains internal implementation details and
should not be imported; use xml.dom.minidom instead.
"""
__all__ = ['NodeList', 'EmptyNodeList', 'StringTypes', 'defproperty']
import xml.dom
StringTypes = (str,)

class NodeList(list):
    __slots__ = ()

    def item(self, index):
        if False:
            while True:
                i = 10
        if 0 <= index < len(self):
            return self[index]

    def _get_length(self):
        if False:
            print('Hello World!')
        return len(self)

    def _set_length(self, value):
        if False:
            return 10
        raise xml.dom.NoModificationAllowedErr("attempt to modify read-only attribute 'length'")
    length = property(_get_length, _set_length, doc='The number of nodes in the NodeList.')

    def __setstate__(self, state):
        if False:
            while True:
                i = 10
        if state is None:
            state = []
        self[:] = state

class EmptyNodeList(tuple):
    __slots__ = ()

    def __add__(self, other):
        if False:
            for i in range(10):
                print('nop')
        NL = NodeList()
        NL.extend(other)
        return NL

    def __radd__(self, other):
        if False:
            i = 10
            return i + 15
        NL = NodeList()
        NL.extend(other)
        return NL

    def item(self, index):
        if False:
            i = 10
            return i + 15
        return None

    def _get_length(self):
        if False:
            print('Hello World!')
        return 0

    def _set_length(self, value):
        if False:
            i = 10
            return i + 15
        raise xml.dom.NoModificationAllowedErr("attempt to modify read-only attribute 'length'")
    length = property(_get_length, _set_length, doc='The number of nodes in the NodeList.')

def defproperty(klass, name, doc):
    if False:
        print('Hello World!')
    get = getattr(klass, '_get_' + name)

    def set(self, value, name=name):
        if False:
            print('Hello World!')
        raise xml.dom.NoModificationAllowedErr('attempt to modify read-only attribute ' + repr(name))
    assert not hasattr(klass, '_set_' + name), 'expected not to find _set_' + name
    prop = property(get, set, doc=doc)
    setattr(klass, name, prop)