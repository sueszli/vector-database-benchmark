"""
*What is this pattern about?
The composite pattern describes a group of objects that is treated the
same way as a single instance of the same type of object. The intent of
a composite is to "compose" objects into tree structures to represent
part-whole hierarchies. Implementing the composite pattern lets clients
treat individual objects and compositions uniformly.

*What does this example do?
The example implements a graphic class，which can be either an ellipse
or a composition of several graphics. Every graphic can be printed.

*Where is the pattern used practically?
In graphics editors a shape can be basic or complex. An example of a
simple shape is a line, where a complex shape is a rectangle which is
made of four line objects. Since shapes have many operations in common
such as rendering the shape to screen, and since shapes follow a
part-whole hierarchy, composite pattern can be used to enable the
program to deal with all shapes uniformly.

*References:
https://en.wikipedia.org/wiki/Composite_pattern
https://infinitescript.com/2014/10/the-23-gang-of-three-design-patterns/

*TL;DR
Describes a group of objects that is treated as a single instance.
"""
from abc import ABC, abstractmethod
from typing import List

class Graphic(ABC):

    @abstractmethod
    def render(self) -> None:
        if False:
            return 10
        raise NotImplementedError('You should implement this!')

class CompositeGraphic(Graphic):

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.graphics: List[Graphic] = []

    def render(self) -> None:
        if False:
            return 10
        for graphic in self.graphics:
            graphic.render()

    def add(self, graphic: Graphic) -> None:
        if False:
            while True:
                i = 10
        self.graphics.append(graphic)

    def remove(self, graphic: Graphic) -> None:
        if False:
            while True:
                i = 10
        self.graphics.remove(graphic)

class Ellipse(Graphic):

    def __init__(self, name: str) -> None:
        if False:
            return 10
        self.name = name

    def render(self) -> None:
        if False:
            while True:
                i = 10
        print(f'Ellipse: {self.name}')

def main():
    if False:
        for i in range(10):
            print('nop')
    '\n    >>> ellipse1 = Ellipse("1")\n    >>> ellipse2 = Ellipse("2")\n    >>> ellipse3 = Ellipse("3")\n    >>> ellipse4 = Ellipse("4")\n\n    >>> graphic1 = CompositeGraphic()\n    >>> graphic2 = CompositeGraphic()\n\n    >>> graphic1.add(ellipse1)\n    >>> graphic1.add(ellipse2)\n    >>> graphic1.add(ellipse3)\n    >>> graphic2.add(ellipse4)\n\n    >>> graphic = CompositeGraphic()\n\n    >>> graphic.add(graphic1)\n    >>> graphic.add(graphic2)\n\n    >>> graphic.render()\n    Ellipse: 1\n    Ellipse: 2\n    Ellipse: 3\n    Ellipse: 4\n    '
if __name__ == '__main__':
    import doctest
    doctest.testmod()