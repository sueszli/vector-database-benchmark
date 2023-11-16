"""
Vector.py -  Extension of QVector3D which adds a few missing methods.
Copyright 2010  Luke Campagnola
Distributed under MIT/X11 license. See license.txt for more information.
"""
from math import acos, degrees
from . import functions as fn
from .Qt import QT_LIB, QtCore, QtGui

class Vector(QtGui.QVector3D):
    """Extension of QVector3D which adds a few helpful methods."""

    def __init__(self, *args):
        if False:
            while True:
                i = 10
        '\n        Handle additional constructions of a Vector\n\n        ==============  ================================================================================================\n        **Arguments:**\n        *args*          Could be any of:\n\n                         * 3 numerics (x, y, and z)\n                         * 2 numerics (x, y, and `0` assumed for z)\n                         * Either of the previous in a list-like collection\n                         * 1 QSizeF (`0` assumed for z)\n                         * 1 QPointF (`0` assumed for z)\n                         * Any other valid QVector3D init args.\n        ==============  ================================================================================================\n        '
        initArgs = args
        if len(args) == 1:
            if isinstance(args[0], QtCore.QSizeF):
                initArgs = (float(args[0].width()), float(args[0].height()), 0)
            elif isinstance(args[0], QtCore.QPoint) or isinstance(args[0], QtCore.QPointF):
                initArgs = (float(args[0].x()), float(args[0].y()), 0)
            elif hasattr(args[0], '__getitem__') and (not isinstance(args[0], QtGui.QVector3D)):
                vals = list(args[0])
                if len(vals) == 2:
                    vals.append(0)
                if len(vals) != 3:
                    raise Exception('Cannot init Vector with sequence of length %d' % len(args[0]))
                initArgs = vals
            elif isinstance(args[0], QtGui.QVector3D):
                initArgs = (args[0].x(), args[0].y(), args[0].z())
        elif len(args) == 2:
            initArgs = (args[0], args[1], 0)
        QtGui.QVector3D.__init__(self, *initArgs)

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return 3

    def __getitem__(self, i):
        if False:
            print('Hello World!')
        if i == 0:
            return self.x()
        elif i == 1:
            return self.y()
        elif i == 2:
            return self.z()
        else:
            raise IndexError('Point has no index %s' % str(i))

    def __setitem__(self, i, x):
        if False:
            print('Hello World!')
        if i == 0:
            return self.setX(x)
        elif i == 1:
            return self.setY(x)
        elif i == 2:
            return self.setZ(x)
        else:
            raise IndexError('Point has no index %s' % str(i))

    def __iter__(self):
        if False:
            return 10
        yield self.x()
        yield self.y()
        yield self.z()

    def angle(self, a):
        if False:
            i = 10
            return i + 15
        'Returns the angle in degrees between this vector and the vector a.'
        n1 = self.length()
        n2 = a.length()
        if n1 == 0.0 or n2 == 0.0:
            return None
        rads = acos(fn.clip_scalar(QtGui.QVector3D.dotProduct(self, a) / (n1 * n2), -1.0, 1.0))
        return degrees(rads)

    def __abs__(self):
        if False:
            while True:
                i = 10
        return Vector(abs(self.x()), abs(self.y()), abs(self.z()))