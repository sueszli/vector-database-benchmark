"""
*What is this pattern about?
This pattern aims to minimise the number of objects that are needed by
a program at run-time. A Flyweight is an object shared by multiple
contexts, and is indistinguishable from an object that is not shared.

The state of a Flyweight should not be affected by it's context, this
is known as its intrinsic state. The decoupling of the objects state
from the object's context, allows the Flyweight to be shared.

*What does this example do?
The example below sets-up an 'object pool' which stores initialised
objects. When a 'Card' is created it first checks to see if it already
exists instead of creating a new one. This aims to reduce the number of
objects initialised by the program.

*References:
http://codesnipers.com/?q=python-flyweights
https://python-patterns.guide/gang-of-four/flyweight/

*Examples in Python ecosystem:
https://docs.python.org/3/library/sys.html#sys.intern

*TL;DR
Minimizes memory usage by sharing data with other similar objects.
"""
import weakref

class Card:
    """The Flyweight"""
    _pool: weakref.WeakValueDictionary = weakref.WeakValueDictionary()

    def __new__(cls, value, suit):
        if False:
            print('Hello World!')
        obj = cls._pool.get(value + suit)
        if obj is None:
            obj = object.__new__(Card)
            cls._pool[value + suit] = obj
            (obj.value, obj.suit) = (value, suit)
        return obj

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return f'<Card: {self.value}{self.suit}>'

def main():
    if False:
        print('Hello World!')
    "\n    >>> c1 = Card('9', 'h')\n    >>> c2 = Card('9', 'h')\n    >>> c1, c2\n    (<Card: 9h>, <Card: 9h>)\n    >>> c1 == c2\n    True\n    >>> c1 is c2\n    True\n\n    >>> c1.new_attr = 'temp'\n    >>> c3 = Card('9', 'h')\n    >>> hasattr(c3, 'new_attr')\n    True\n\n    >>> Card._pool.clear()\n    >>> c4 = Card('9', 'h')\n    >>> hasattr(c4, 'new_attr')\n    False\n    "
if __name__ == '__main__':
    import doctest
    doctest.testmod()