"""
*References:
http://en.wikibooks.org/wiki/Computer_Science_Design_Patterns/Bridge_Pattern#Python

*TL;DR
Decouples an abstraction from its implementation.
"""

class DrawingAPI1:

    def draw_circle(self, x, y, radius):
        if False:
            print('Hello World!')
        print(f'API1.circle at {x}:{y} radius {radius}')

class DrawingAPI2:

    def draw_circle(self, x, y, radius):
        if False:
            while True:
                i = 10
        print(f'API2.circle at {x}:{y} radius {radius}')

class CircleShape:

    def __init__(self, x, y, radius, drawing_api):
        if False:
            print('Hello World!')
        self._x = x
        self._y = y
        self._radius = radius
        self._drawing_api = drawing_api

    def draw(self):
        if False:
            for i in range(10):
                print('nop')
        self._drawing_api.draw_circle(self._x, self._y, self._radius)

    def scale(self, pct):
        if False:
            i = 10
            return i + 15
        self._radius *= pct

def main():
    if False:
        i = 10
        return i + 15
    '\n    >>> shapes = (CircleShape(1, 2, 3, DrawingAPI1()), CircleShape(5, 7, 11, DrawingAPI2()))\n\n    >>> for shape in shapes:\n    ...    shape.scale(2.5)\n    ...    shape.draw()\n    API1.circle at 1:2 radius 7.5\n    API2.circle at 5:7 radius 27.5\n    '
if __name__ == '__main__':
    import doctest
    doctest.testmod()