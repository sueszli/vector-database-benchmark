from math import atan2, degrees
import numpy as np
from .Point import Point
from .Qt import QtGui
from . import SRTTransform3D

class SRTTransform(QtGui.QTransform):
    """Transform that can always be represented as a combination of 3 matrices: scale * rotate * translate
    This transform has no shear; angles are always preserved.
    """

    def __init__(self, init=None):
        if False:
            i = 10
            return i + 15
        QtGui.QTransform.__init__(self)
        self.reset()
        if init is None:
            return
        elif isinstance(init, dict):
            self.restoreState(init)
        elif isinstance(init, SRTTransform):
            self._state = {'pos': Point(init._state['pos']), 'scale': Point(init._state['scale']), 'angle': init._state['angle']}
            self.update()
        elif isinstance(init, QtGui.QTransform):
            self.setFromQTransform(init)
        elif isinstance(init, QtGui.QMatrix4x4):
            self.setFromMatrix4x4(init)
        else:
            raise Exception('Cannot create SRTTransform from input type: %s' % str(type(init)))

    def getScale(self):
        if False:
            return 10
        return self._state['scale']

    def getRotation(self):
        if False:
            for i in range(10):
                print('nop')
        return self._state['angle']

    def getTranslation(self):
        if False:
            i = 10
            return i + 15
        return self._state['pos']

    def reset(self):
        if False:
            for i in range(10):
                print('nop')
        self._state = {'pos': Point(0, 0), 'scale': Point(1, 1), 'angle': 0.0}
        self.update()

    def setFromQTransform(self, tr):
        if False:
            print('Hello World!')
        p1 = Point(tr.map(0.0, 0.0))
        p2 = Point(tr.map(1.0, 0.0))
        p3 = Point(tr.map(0.0, 1.0))
        dp2 = Point(p2 - p1)
        dp3 = Point(p3 - p1)
        if dp2.angle(dp3, units='radians') > 0:
            da = 0
            sy = -1.0
        else:
            da = 0
            sy = 1.0
        self._state = {'pos': Point(p1), 'scale': Point(dp2.length(), dp3.length() * sy), 'angle': degrees(atan2(dp2[1], dp2[0])) + da}
        self.update()

    def setFromMatrix4x4(self, m):
        if False:
            while True:
                i = 10
        m = SRTTransform3D.SRTTransform3D(m)
        (angle, axis) = m.getRotation()
        if angle != 0 and (axis[0] != 0 or axis[1] != 0 or axis[2] != 1):
            print('angle: %s  axis: %s' % (str(angle), str(axis)))
            raise Exception('Can only convert 4x4 matrix to 3x3 if rotation is around Z-axis.')
        self._state = {'pos': Point(m.getTranslation()), 'scale': Point(m.getScale()), 'angle': angle}
        self.update()

    def translate(self, *args):
        if False:
            while True:
                i = 10
        'Acceptable arguments are: \n           x, y\n           [x, y]\n           Point(x,y)'
        t = Point(*args)
        self.setTranslate(self._state['pos'] + t)

    def setTranslate(self, *args):
        if False:
            for i in range(10):
                print('nop')
        'Acceptable arguments are: \n           x, y\n           [x, y]\n           Point(x,y)'
        self._state['pos'] = Point(*args)
        self.update()

    def scale(self, *args):
        if False:
            for i in range(10):
                print('nop')
        'Acceptable arguments are: \n           x, y\n           [x, y]\n           Point(x,y)'
        s = Point(*args)
        self.setScale(self._state['scale'] * s)

    def setScale(self, *args):
        if False:
            while True:
                i = 10
        'Acceptable arguments are: \n           x, y\n           [x, y]\n           Point(x,y)'
        self._state['scale'] = Point(*args)
        self.update()

    def rotate(self, angle):
        if False:
            return 10
        'Rotate the transformation by angle (in degrees)'
        self.setRotate(self._state['angle'] + angle)

    def setRotate(self, angle):
        if False:
            i = 10
            return i + 15
        'Set the transformation rotation to angle (in degrees)'
        self._state['angle'] = angle
        self.update()

    def __truediv__(self, t):
        if False:
            i = 10
            return i + 15
        'A / B  ==  B^-1 * A'
        dt = t.inverted()[0] * self
        return SRTTransform(dt)

    def __div__(self, t):
        if False:
            i = 10
            return i + 15
        return self.__truediv__(t)

    def __mul__(self, t):
        if False:
            return 10
        return SRTTransform(QtGui.QTransform.__mul__(self, t))

    def saveState(self):
        if False:
            for i in range(10):
                print('nop')
        p = self._state['pos']
        s = self._state['scale']
        return {'pos': (p[0], p[1]), 'scale': (s[0], s[1]), 'angle': self._state['angle']}

    def restoreState(self, state):
        if False:
            for i in range(10):
                print('nop')
        self._state['pos'] = Point(state.get('pos', (0, 0)))
        self._state['scale'] = Point(state.get('scale', (1.0, 1.0)))
        self._state['angle'] = state.get('angle', 0)
        self.update()

    def update(self):
        if False:
            while True:
                i = 10
        QtGui.QTransform.reset(self)
        QtGui.QTransform.translate(self, *self._state['pos'])
        QtGui.QTransform.rotate(self, self._state['angle'])
        QtGui.QTransform.scale(self, *self._state['scale'])

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return str(self.saveState())

    def matrix(self):
        if False:
            return 10
        return np.array([[self.m11(), self.m12(), self.m13()], [self.m21(), self.m22(), self.m23()], [self.m31(), self.m32(), self.m33()]])