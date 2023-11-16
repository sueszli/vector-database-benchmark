"""
*What is this pattern about?
It decouples the creation of a complex object and its representation,
so that the same process can be reused to build objects from the same
family.
This is useful when you must separate the specification of an object
from its actual representation (generally for abstraction).

*What does this example do?

The first example achieves this by using an abstract base
class for a building, where the initializer (__init__ method) specifies the
steps needed, and the concrete subclasses implement these steps.

In other programming languages, a more complex arrangement is sometimes
necessary. In particular, you cannot have polymorphic behaviour in a constructor in C++ -
see https://stackoverflow.com/questions/1453131/how-can-i-get-polymorphic-behavior-in-a-c-constructor
- which means this Python technique will not work. The polymorphism
required has to be provided by an external, already constructed
instance of a different class.

In general, in Python this won't be necessary, but a second example showing
this kind of arrangement is also included.

*Where is the pattern used practically?

*References:
https://sourcemaking.com/design_patterns/builder

*TL;DR
Decouples the creation of a complex object and its representation.
"""

class Building:

    def __init__(self) -> None:
        if False:
            return 10
        self.build_floor()
        self.build_size()

    def build_floor(self):
        if False:
            while True:
                i = 10
        raise NotImplementedError

    def build_size(self):
        if False:
            return 10
        raise NotImplementedError

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return 'Floor: {0.floor} | Size: {0.size}'.format(self)

class House(Building):

    def build_floor(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.floor = 'One'

    def build_size(self) -> None:
        if False:
            print('Hello World!')
        self.size = 'Big'

class Flat(Building):

    def build_floor(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.floor = 'More than One'

    def build_size(self) -> None:
        if False:
            return 10
        self.size = 'Small'

class ComplexBuilding:

    def __repr__(self) -> str:
        if False:
            return 10
        return 'Floor: {0.floor} | Size: {0.size}'.format(self)

class ComplexHouse(ComplexBuilding):

    def build_floor(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.floor = 'One'

    def build_size(self) -> None:
        if False:
            i = 10
            return i + 15
        self.size = 'Big and fancy'

def construct_building(cls) -> Building:
    if False:
        return 10
    building = cls()
    building.build_floor()
    building.build_size()
    return building

def main():
    if False:
        print('Hello World!')
    '\n    >>> house = House()\n    >>> house\n    Floor: One | Size: Big\n\n    >>> flat = Flat()\n    >>> flat\n    Floor: More than One | Size: Small\n\n    # Using an external constructor function:\n    >>> complex_house = construct_building(ComplexHouse)\n    >>> complex_house\n    Floor: One | Size: Big and fancy\n    '
if __name__ == '__main__':
    import doctest
    doctest.testmod()