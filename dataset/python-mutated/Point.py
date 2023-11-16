"""
Point.py -  Extension of QPointF which adds a few missing methods.
Copyright 2010  Luke Campagnola
Distributed under MIT/X11 license. See license.txt for more information.
"""
from math import atan2, degrees, hypot
from .Qt import QtCore

class Point(QtCore.QPointF):
    """Extension of QPointF which adds a few missing methods."""
    __slots__ = ()

    def __init__(self, *args):
        if False:
            for i in range(10):
                print('nop')
        if len(args) == 1:
            if isinstance(args[0], (QtCore.QSize, QtCore.QSizeF)):
                super().__init__(float(args[0].width()), float(args[0].height()))
                return
            elif isinstance(args[0], (int, float)):
                super().__init__(float(args[0]), float(args[0]))
                return
            elif hasattr(args[0], '__getitem__'):
                super().__init__(float(args[0][0]), float(args[0][1]))
                return
        elif len(args) == 2:
            super().__init__(args[0], args[1])
            return
        super().__init__(*args)

    def __len__(self):
        if False:
            i = 10
            return i + 15
        return 2

    def __reduce__(self):
        if False:
            while True:
                i = 10
        return (Point, (self.x(), self.y()))

    def __getitem__(self, i):
        if False:
            return 10
        if i == 0:
            return self.x()
        elif i == 1:
            return self.y()
        else:
            raise IndexError('Point has no index %s' % str(i))

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        yield self.x()
        yield self.y()

    def __setitem__(self, i, x):
        if False:
            for i in range(10):
                print('nop')
        if i == 0:
            return self.setX(x)
        elif i == 1:
            return self.setY(x)
        else:
            raise IndexError('Point has no index %s' % str(i))

    def __radd__(self, a):
        if False:
            return 10
        return self._math_('__radd__', a)

    def __add__(self, a):
        if False:
            while True:
                i = 10
        return self._math_('__add__', a)

    def __rsub__(self, a):
        if False:
            print('Hello World!')
        return self._math_('__rsub__', a)

    def __sub__(self, a):
        if False:
            while True:
                i = 10
        return self._math_('__sub__', a)

    def __rmul__(self, a):
        if False:
            while True:
                i = 10
        return self._math_('__rmul__', a)

    def __mul__(self, a):
        if False:
            while True:
                i = 10
        return self._math_('__mul__', a)

    def __rdiv__(self, a):
        if False:
            while True:
                i = 10
        return self._math_('__rdiv__', a)

    def __div__(self, a):
        if False:
            i = 10
            return i + 15
        return self._math_('__div__', a)

    def __truediv__(self, a):
        if False:
            while True:
                i = 10
        return self._math_('__truediv__', a)

    def __rtruediv__(self, a):
        if False:
            for i in range(10):
                print('nop')
        return self._math_('__rtruediv__', a)

    def __rpow__(self, a):
        if False:
            print('Hello World!')
        return self._math_('__rpow__', a)

    def __pow__(self, a):
        if False:
            print('Hello World!')
        return self._math_('__pow__', a)

    def _math_(self, op, x):
        if False:
            i = 10
            return i + 15
        if not isinstance(x, QtCore.QPointF):
            x = Point(x)
        return Point(getattr(self.x(), op)(x.x()), getattr(self.y(), op)(x.y()))

    def length(self):
        if False:
            print('Hello World!')
        'Returns the vector length of this Point.'
        return hypot(self.x(), self.y())

    def norm(self):
        if False:
            while True:
                i = 10
        'Returns a vector in the same direction with unit length.'
        return self / self.length()

    def angle(self, a, units='degrees'):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the angle in degrees from the vector a to self.\n        \n        Parameters\n        ----------\n        a : Point, QPointF or QPoint\n            The Point to return the angle with\n        units : str, optional\n            The units with which to compute the angle with, "degrees" or "radians",\n            default "degrees"\n        \n        Returns\n        -------\n        float\n            The angle between two vectors\n        '
        rads = atan2(self.y(), self.x()) - atan2(a.y(), a.x())
        if units == 'radians':
            return rads
        return degrees(rads)

    def dot(self, a):
        if False:
            print('Hello World!')
        'Returns the dot product of a and this Point.'
        if not isinstance(a, QtCore.QPointF):
            a = Point(a)
        return Point.dotProduct(self, a)

    def cross(self, a):
        if False:
            print('Hello World!')
        'Returns the cross product of a and this Point'
        if not isinstance(a, QtCore.QPointF):
            a = Point(a)
        return self.x() * a.y() - self.y() * a.x()

    def proj(self, b):
        if False:
            print('Hello World!')
        'Return the projection of this vector onto the vector b'
        b1 = b.norm()
        return self.dot(b1) * b1

    def __repr__(self):
        if False:
            return 10
        return 'Point(%f, %f)' % (self.x(), self.y())

    def min(self):
        if False:
            for i in range(10):
                print('nop')
        return min(self.x(), self.y())

    def max(self):
        if False:
            return 10
        return max(self.x(), self.y())

    def copy(self):
        if False:
            for i in range(10):
                print('nop')
        return Point(self)

    def toQPoint(self):
        if False:
            for i in range(10):
                print('nop')
        return self.toPoint()