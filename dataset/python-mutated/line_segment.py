"""
This class represents a line segment
"""
import typing
from decimal import Decimal
from math import sqrt
from borb.pdf.canvas.geometry.matrix import Matrix

class LineSegment:
    """
    This class represents a line segment
    """

    def __init__(self, x0: Decimal, y0: Decimal, x1: Decimal, y1: Decimal):
        if False:
            for i in range(10):
                print('nop')
        self.x0: Decimal = x0
        self.y0: Decimal = y0
        self.x1: Decimal = x1
        self.y1: Decimal = y1

    def get_end(self) -> typing.Tuple[Decimal, Decimal]:
        if False:
            return 10
        '\n        This function returns the end of this LineSegment\n        '
        return (self.x1, self.y1)

    def get_start(self) -> typing.Tuple[Decimal, Decimal]:
        if False:
            while True:
                i = 10
        '\n        This function returns the start of this LineSegment\n        '
        return (self.x0, self.y0)

    def length(self) -> Decimal:
        if False:
            print('Hello World!')
        '\n        This function returns the length of this LineSegment\n        '
        return Decimal(sqrt((self.x0 - self.x1) ** 2 + (self.y0 - self.y1) ** 2))

    def transform_by(self, matrix: Matrix) -> 'LineSegment':
        if False:
            i = 10
            return i + 15
        '\n        This function transforms the start and end of this LineSegment by a given Matrix,\n        it returns the transformed LineSegment\n        '
        p0 = matrix.cross(self.x0, self.y0, Decimal(1))
        p1 = matrix.cross(self.x1, self.y1, Decimal(1))
        return LineSegment(p0[0], p0[1], p1[0], p1[1])