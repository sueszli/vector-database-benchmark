"""
*What is this pattern about?

In Java and other languages, the Abstract Factory Pattern serves to provide an interface for
creating related/dependent objects without need to specify their
actual class.

The idea is to abstract the creation of objects depending on business
logic, platform choice, etc.

In Python, the interface we use is simply a callable, which is "builtin" interface
in Python, and in normal circumstances we can simply use the class itself as
that callable, because classes are first class objects in Python.

*What does this example do?
This particular implementation abstracts the creation of a pet and
does so depending on the factory we chose (Dog or Cat, or random_animal)
This works because both Dog/Cat and random_animal respect a common
interface (callable for creation and .speak()).
Now my application can create pets abstractly and decide later,
based on my own criteria, dogs over cats.

*Where is the pattern used practically?

*References:
https://sourcemaking.com/design_patterns/abstract_factory
http://ginstrom.com/scribbles/2007/10/08/design-patterns-python-style/

*TL;DR
Provides a way to encapsulate a group of individual factories.
"""
import random
from typing import Type

class Pet:

    def __init__(self, name: str) -> None:
        if False:
            i = 10
            return i + 15
        self.name = name

    def speak(self) -> None:
        if False:
            print('Hello World!')
        raise NotImplementedError

    def __str__(self) -> str:
        if False:
            print('Hello World!')
        raise NotImplementedError

class Dog(Pet):

    def speak(self) -> None:
        if False:
            print('Hello World!')
        print('woof')

    def __str__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return f'Dog<{self.name}>'

class Cat(Pet):

    def speak(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        print('meow')

    def __str__(self) -> str:
        if False:
            i = 10
            return i + 15
        return f'Cat<{self.name}>'

class PetShop:
    """A pet shop"""

    def __init__(self, animal_factory: Type[Pet]) -> None:
        if False:
            print('Hello World!')
        'pet_factory is our abstract factory.  We can set it at will.'
        self.pet_factory = animal_factory

    def buy_pet(self, name: str) -> Pet:
        if False:
            while True:
                i = 10
        'Creates and shows a pet using the abstract factory'
        pet = self.pet_factory(name)
        print(f'Here is your lovely {pet}')
        return pet

def random_animal(name: str) -> Pet:
    if False:
        while True:
            i = 10
    "Let's be dynamic!"
    return random.choice([Dog, Cat])(name)

def main() -> None:
    if False:
        i = 10
        return i + 15
    '\n    # A Shop that sells only cats\n    >>> cat_shop = PetShop(Cat)\n    >>> pet = cat_shop.buy_pet("Lucy")\n    Here is your lovely Cat<Lucy>\n    >>> pet.speak()\n    meow\n\n    # A shop that sells random animals\n    >>> shop = PetShop(random_animal)\n    >>> for name in ["Max", "Jack", "Buddy"]:\n    ...    pet = shop.buy_pet(name)\n    ...    pet.speak()\n    ...    print("=" * 20)\n    Here is your lovely Cat<Max>\n    meow\n    ====================\n    Here is your lovely Dog<Jack>\n    woof\n    ====================\n    Here is your lovely Dog<Buddy>\n    woof\n    ====================\n    '
if __name__ == '__main__':
    random.seed(1234)
    shop = PetShop(random_animal)
    import doctest
    doctest.testmod()