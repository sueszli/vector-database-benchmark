from __future__ import annotations
from typing import Literal
from zope.interface import Interface, implementer
from twisted.python import components

def foo() -> Literal[2]:
    if False:
        return 10
    return 2

class X:

    def __init__(self, x: str) -> None:
        if False:
            i = 10
            return i + 15
        self.x = x

    def do(self) -> None:
        if False:
            while True:
                i = 10
        pass

class XComponent(components.Componentized):
    pass

class IX(Interface):
    pass

@implementer(IX)
class XA(components.Adapter):

    def method(self) -> None:
        if False:
            while True:
                i = 10
        pass
components.registerAdapter(XA, X, IX)