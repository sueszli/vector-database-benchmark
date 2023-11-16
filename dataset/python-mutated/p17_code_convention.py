"""
Topic: 强制编码规约
Desc : 
"""

class MyMeta(type):

    def __init__(self, clsname, bases, clsdict):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(clsname, bases, clsdict)

class Root(metaclass=MyMeta):
    pass

class A(Root):
    pass

class B(Root):
    pass

class NoMixedCaseMeta(type):

    def __new__(cls, clsname, bases, clsdict):
        if False:
            i = 10
            return i + 15
        for name in clsdict:
            if name.lower() != name:
                raise TypeError('Bad attribute name: ' + name)
        return super().__new__(cls, clsname, bases, clsdict)

class Root(metaclass=NoMixedCaseMeta):
    pass

class A(Root):

    def foo_bar(self):
        if False:
            i = 10
            return i + 15
        pass

class B(Root):

    def fooBar(self):
        if False:
            for i in range(10):
                print('nop')
        pass
from inspect import signature
import logging

class MatchSignaturesMeta(type):

    def __init__(self, clsname, bases, clsdict):
        if False:
            i = 10
            return i + 15
        super().__init__(clsname, bases, clsdict)
        sup = super(self, self)
        for (name, value) in clsdict.items():
            if name.startswith('_') or not callable(value):
                continue
            prev_dfn = getattr(sup, name, None)
            if prev_dfn:
                prev_sig = signature(prev_dfn)
                val_sig = signature(value)
                if prev_sig != val_sig:
                    logging.warning('Signature mismatch in %s. %s != %s', value.__qualname__, prev_sig, val_sig)

class Root(metaclass=MatchSignaturesMeta):
    pass

class A(Root):

    def foo(self, x, y):
        if False:
            for i in range(10):
                print('nop')
        pass

    def spam(self, x, *, z):
        if False:
            i = 10
            return i + 15
        pass

class B(A):

    def foo(self, a, b):
        if False:
            for i in range(10):
                print('nop')
        pass

    def spam(self, x, z):
        if False:
            return 10
        pass