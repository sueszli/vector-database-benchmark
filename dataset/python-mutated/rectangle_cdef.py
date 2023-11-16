@cython.cclass
class Rectangle:
    x0: cython.int
    y0: cython.int
    x1: cython.int
    y1: cython.int

    def __init__(self, x0: cython.int, y0: cython.int, x1: cython.int, y1: cython.int):
        if False:
            i = 10
            return i + 15
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1

    @cython.cfunc
    def _area(self) -> cython.int:
        if False:
            for i in range(10):
                print('nop')
        area: cython.int = (self.x1 - self.x0) * (self.y1 - self.y0)
        if area < 0:
            area = -area
        return area

    def area(self):
        if False:
            return 10
        return self._area()

def rectArea(x0, y0, x1, y1):
    if False:
        print('Hello World!')
    rect: Rectangle = Rectangle(x0, y0, x1, y1)
    return rect._area()