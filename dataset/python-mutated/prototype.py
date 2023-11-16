"""
*What is this pattern about?
This patterns aims to reduce the number of classes required by an
application. Instead of relying on subclasses it creates objects by
copying a prototypical instance at run-time.

This is useful as it makes it easier to derive new kinds of objects,
when instances of the class have only a few different combinations of
state, and when instantiation is expensive.

*What does this example do?
When the number of prototypes in an application can vary, it can be
useful to keep a Dispatcher (aka, Registry or Manager). This allows
clients to query the Dispatcher for a prototype before cloning a new
instance.

Below provides an example of such Dispatcher, which contains three
copies of the prototype: 'default', 'objecta' and 'objectb'.

*TL;DR
Creates new object instances by cloning prototype.
"""
from __future__ import annotations
from typing import Any

class Prototype:

    def __init__(self, value: str='default', **attrs: Any) -> None:
        if False:
            print('Hello World!')
        self.value = value
        self.__dict__.update(attrs)

    def clone(self, **attrs: Any) -> Prototype:
        if False:
            for i in range(10):
                print('nop')
        'Clone a prototype and update inner attributes dictionary'
        obj = self.__class__(**self.__dict__)
        obj.__dict__.update(attrs)
        return obj

class PrototypeDispatcher:

    def __init__(self):
        if False:
            return 10
        self._objects = {}

    def get_objects(self) -> dict[str, Prototype]:
        if False:
            print('Hello World!')
        'Get all objects'
        return self._objects

    def register_object(self, name: str, obj: Prototype) -> None:
        if False:
            print('Hello World!')
        'Register an object'
        self._objects[name] = obj

    def unregister_object(self, name: str) -> None:
        if False:
            print('Hello World!')
        'Unregister an object'
        del self._objects[name]

def main() -> None:
    if False:
        while True:
            i = 10
    "\n    >>> dispatcher = PrototypeDispatcher()\n    >>> prototype = Prototype()\n\n    >>> d = prototype.clone()\n    >>> a = prototype.clone(value='a-value', category='a')\n    >>> b = a.clone(value='b-value', is_checked=True)\n    >>> dispatcher.register_object('objecta', a)\n    >>> dispatcher.register_object('objectb', b)\n    >>> dispatcher.register_object('default', d)\n\n    >>> [{n: p.value} for n, p in dispatcher.get_objects().items()]\n    [{'objecta': 'a-value'}, {'objectb': 'b-value'}, {'default': 'default'}]\n\n    >>> print(b.category, b.is_checked)\n    a True\n    "
if __name__ == '__main__':
    import doctest
    doctest.testmod()