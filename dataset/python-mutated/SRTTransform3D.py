from math import atan2, degrees
import numpy as np
from .Qt import QtGui
from .Transform3D import Transform3D
from .Vector import Vector
from . import SRTTransform

class SRTTransform3D(Transform3D):
    """4x4 Transform matrix that can always be represented as a combination of 3 matrices: scale * rotate * translate
    This transform has no shear; angles are always preserved.
    """

    def __init__(self, init=None):
        if False:
            print('Hello World!')
        Transform3D.__init__(self)
        self.reset()
        if init is None:
            return
        if init.__class__ is QtGui.QTransform:
            init = SRTTransform.SRTTransform(init)
        if isinstance(init, dict):
            self.restoreState(init)
        elif isinstance(init, SRTTransform3D):
            self._state = {'pos': Vector(init._state['pos']), 'scale': Vector(init._state['scale']), 'angle': init._state['angle'], 'axis': Vector(init._state['axis'])}
            self.update()
        elif isinstance(init, SRTTransform.SRTTransform):
            self._state = {'pos': Vector(init._state['pos']), 'scale': Vector(init._state['scale']), 'angle': init._state['angle'], 'axis': Vector(0, 0, 1)}
            self._state['scale'][2] = 1.0
            self.update()
        elif isinstance(init, QtGui.QMatrix4x4):
            self.setFromMatrix(init)
        else:
            raise Exception('Cannot build SRTTransform3D from argument type:', type(init))

    def getScale(self):
        if False:
            i = 10
            return i + 15
        return Vector(self._state['scale'])

    def getRotation(self):
        if False:
            for i in range(10):
                print('nop')
        'Return (angle, axis) of rotation'
        return (self._state['angle'], Vector(self._state['axis']))

    def getTranslation(self):
        if False:
            return 10
        return Vector(self._state['pos'])

    def reset(self):
        if False:
            print('Hello World!')
        self._state = {'pos': Vector(0, 0, 0), 'scale': Vector(1, 1, 1), 'angle': 0.0, 'axis': (0, 0, 1)}
        self.update()

    def translate(self, *args):
        if False:
            for i in range(10):
                print('nop')
        'Adjust the translation of this transform'
        t = Vector(*args)
        self.setTranslate(self._state['pos'] + t)

    def setTranslate(self, *args):
        if False:
            i = 10
            return i + 15
        'Set the translation of this transform'
        self._state['pos'] = Vector(*args)
        self.update()

    def scale(self, *args):
        if False:
            for i in range(10):
                print('nop')
        'adjust the scale of this transform'
        if len(args) == 1 and hasattr(args[0], '__len__'):
            args = args[0]
        if len(args) == 2:
            args = args + (1,)
        s = Vector(*args)
        self.setScale(self._state['scale'] * s)

    def setScale(self, *args):
        if False:
            i = 10
            return i + 15
        'Set the scale of this transform'
        if len(args) == 1 and hasattr(args[0], '__len__'):
            args = args[0]
        if len(args) == 2:
            args = args + (1,)
        self._state['scale'] = Vector(*args)
        self.update()

    def rotate(self, angle, axis=(0, 0, 1)):
        if False:
            while True:
                i = 10
        'Adjust the rotation of this transform'
        origAxis = self._state['axis']
        if axis[0] == origAxis[0] and axis[1] == origAxis[1] and (axis[2] == origAxis[2]):
            self.setRotate(self._state['angle'] + angle)
        else:
            m = QtGui.QMatrix4x4()
            m.translate(*self._state['pos'])
            m.rotate(self._state['angle'], *self._state['axis'])
            m.rotate(angle, *axis)
            m.scale(*self._state['scale'])
            self.setFromMatrix(m)

    def setRotate(self, angle, axis=(0, 0, 1)):
        if False:
            return 10
        'Set the transformation rotation to angle (in degrees)'
        self._state['angle'] = angle
        self._state['axis'] = Vector(axis)
        self.update()

    def setFromMatrix(self, m):
        if False:
            print('Hello World!')
        '\n        Set this transform based on the elements of *m*\n        The input matrix must be affine AND have no shear,\n        otherwise the conversion will most likely fail.\n        '
        import numpy.linalg
        for i in range(4):
            self.setRow(i, m.row(i))
        m = self.matrix().reshape(4, 4)
        self._state['pos'] = m[:3, 3]
        scale = (m[:3, :3] ** 2).sum(axis=0) ** 0.5
        z = np.cross(m[0, :3], m[1, :3])
        if np.dot(z, m[2, :3]) < 0:
            scale[1] *= -1
        self._state['scale'] = scale
        r = m[:3, :3] / scale[np.newaxis, :]
        try:
            (evals, evecs) = numpy.linalg.eig(r)
        except:
            print('Rotation matrix: %s' % str(r))
            print('Scale: %s' % str(scale))
            print('Original matrix: %s' % str(m))
            raise
        eigIndex = np.argwhere(np.abs(evals - 1) < 1e-06)
        if len(eigIndex) < 1:
            print('eigenvalues: %s' % str(evals))
            print('eigenvectors: %s' % str(evecs))
            print('index: %s, %s' % (str(eigIndex), str(evals - 1)))
            raise Exception('Could not determine rotation axis.')
        axis = evecs[:, eigIndex[0, 0]].real
        axis /= (axis ** 2).sum() ** 0.5
        self._state['axis'] = axis
        cos = (r.trace() - 1) * 0.5
        axisInd = np.argmax(np.abs(axis))
        (rInd, sign) = [((1, 2), -1), ((0, 2), 1), ((0, 1), -1)][axisInd]
        sin = (r - r.T)[rInd] / (2.0 * sign * axis[axisInd])
        self._state['angle'] = degrees(atan2(sin, cos))
        if self._state['angle'] == 0:
            self._state['axis'] = (0, 0, 1)

    def as2D(self):
        if False:
            while True:
                i = 10
        'Return a QTransform representing the x,y portion of this transform (if possible)'
        return SRTTransform.SRTTransform(self)

    def saveState(self):
        if False:
            return 10
        p = self._state['pos']
        s = self._state['scale']
        ax = self._state['axis']
        return {'pos': (p[0], p[1], p[2]), 'scale': (s[0], s[1], s[2]), 'angle': self._state['angle'], 'axis': (ax[0], ax[1], ax[2])}

    def restoreState(self, state):
        if False:
            return 10
        self._state['pos'] = Vector(state.get('pos', (0.0, 0.0, 0.0)))
        scale = state.get('scale', (1.0, 1.0, 1.0))
        scale = tuple(scale) + (1.0,) * (3 - len(scale))
        self._state['scale'] = Vector(scale)
        self._state['angle'] = state.get('angle', 0.0)
        self._state['axis'] = state.get('axis', (0, 0, 1))
        self.update()

    def update(self):
        if False:
            for i in range(10):
                print('nop')
        Transform3D.setToIdentity(self)
        Transform3D.translate(self, *self._state['pos'])
        Transform3D.rotate(self, self._state['angle'], *self._state['axis'])
        Transform3D.scale(self, *self._state['scale'])

    def __repr__(self):
        if False:
            return 10
        return str(self.saveState())

    def matrix(self, nd=3):
        if False:
            i = 10
            return i + 15
        if nd == 3:
            return np.array(self.copyDataTo()).reshape(4, 4)
        elif nd == 2:
            m = np.array(self.copyDataTo()).reshape(4, 4)
            m[2] = m[3]
            m[:, 2] = m[:, 3]
            return m[:3, :3]
        else:
            raise Exception("Argument 'nd' must be 2 or 3")