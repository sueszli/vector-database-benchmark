"""
*What is this pattern about?
The Adapter pattern provides a different interface for a class. We can
think about it as a cable adapter that allows you to charge a phone
somewhere that has outlets in a different shape. Following this idea,
the Adapter pattern is useful to integrate classes that couldn't be
integrated due to their incompatible interfaces.

*What does this example do?

The example has classes that represent entities (Dog, Cat, Human, Car)
that make different noises. The Adapter class provides a different
interface to the original methods that make such noises. So the
original interfaces (e.g., bark and meow) are available under a
different name: make_noise.

*Where is the pattern used practically?
The Grok framework uses adapters to make objects work with a
particular API without modifying the objects themselves:
http://grok.zope.org/doc/current/grok_overview.html#adapters

*References:
http://ginstrom.com/scribbles/2008/11/06/generic-adapter-class-in-python/
https://sourcemaking.com/design_patterns/adapter
http://python-3-patterns-idioms-test.readthedocs.io/en/latest/ChangeInterface.html#adapter

*TL;DR
Allows the interface of an existing class to be used as another interface.
"""
from typing import Callable, TypeVar
T = TypeVar('T')

class Dog:

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        self.name = 'Dog'

    def bark(self) -> str:
        if False:
            print('Hello World!')
        return 'woof!'

class Cat:

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        self.name = 'Cat'

    def meow(self) -> str:
        if False:
            print('Hello World!')
        return 'meow!'

class Human:

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        self.name = 'Human'

    def speak(self) -> str:
        if False:
            return 10
        return "'hello'"

class Car:

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        self.name = 'Car'

    def make_noise(self, octane_level: int) -> str:
        if False:
            print('Hello World!')
        return f"vroom{'!' * octane_level}"

class Adapter:
    """Adapts an object by replacing methods.

    Usage
    ------
    dog = Dog()
    dog = Adapter(dog, make_noise=dog.bark)
    """

    def __init__(self, obj: T, **adapted_methods: Callable):
        if False:
            for i in range(10):
                print('nop')
        "We set the adapted methods in the object's dict."
        self.obj = obj
        self.__dict__.update(adapted_methods)

    def __getattr__(self, attr):
        if False:
            for i in range(10):
                print('nop')
        'All non-adapted calls are passed to the object.'
        return getattr(self.obj, attr)

    def original_dict(self):
        if False:
            i = 10
            return i + 15
        'Print original object dict.'
        return self.obj.__dict__

def main():
    if False:
        for i in range(10):
            print('nop')
    '\n    >>> objects = []\n    >>> dog = Dog()\n    >>> print(dog.__dict__)\n    {\'name\': \'Dog\'}\n\n    >>> objects.append(Adapter(dog, make_noise=dog.bark))\n\n    >>> objects[0].__dict__[\'obj\'], objects[0].__dict__[\'make_noise\']\n    (<...Dog object at 0x...>, <bound method Dog.bark of <...Dog object at 0x...>>)\n\n    >>> print(objects[0].original_dict())\n    {\'name\': \'Dog\'}\n\n    >>> cat = Cat()\n    >>> objects.append(Adapter(cat, make_noise=cat.meow))\n    >>> human = Human()\n    >>> objects.append(Adapter(human, make_noise=human.speak))\n    >>> car = Car()\n    >>> objects.append(Adapter(car, make_noise=lambda: car.make_noise(3)))\n\n    >>> for obj in objects:\n    ...    print("A {0} goes {1}".format(obj.name, obj.make_noise()))\n    A Dog goes woof!\n    A Cat goes meow!\n    A Human goes \'hello\'\n    A Car goes vroom!!!\n    '
if __name__ == '__main__':
    import doctest
    doctest.testmod(optionflags=doctest.ELLIPSIS)