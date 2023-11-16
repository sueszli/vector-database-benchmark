"""
RawImageWidget.py
Copyright 2010-2016 Luke Campagnola
Distributed under MIT/X11 license. See license.txt for more information.
"""
from .. import functions as fn
from .. import getConfigOption, getCupy
from ..Qt import QtCore, QtGui, QtWidgets
try:
    QOpenGLWidget = QtWidgets.QOpenGLWidget
    from OpenGL.GL import *
    HAVE_OPENGL = True
except (ImportError, AttributeError):
    HAVE_OPENGL = False
__all__ = ['RawImageWidget']

class RawImageWidget(QtWidgets.QWidget):
    """
    Widget optimized for very fast video display.
    Generally using an ImageItem inside GraphicsView is fast enough.
    On some systems this may provide faster video. See the VideoSpeedTest example for benchmarking.
    """

    def __init__(self, parent=None, scaled=False):
        if False:
            return 10
        '\n        Setting scaled=True will cause the entire image to be displayed within the boundaries of the widget.\n        This also greatly reduces the speed at which it will draw frames.\n        '
        QtWidgets.QWidget.__init__(self, parent)
        self.setSizePolicy(QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding))
        self.scaled = scaled
        self.opts = None
        self.image = None
        self._cp = getCupy()

    def setImage(self, img, *args, **kargs):
        if False:
            return 10
        '\n        img must be ndarray of shape (x,y), (x,y,3), or (x,y,4).\n        Extra arguments are sent to functions.makeARGB\n        '
        if getConfigOption('imageAxisOrder') == 'col-major':
            img = img.swapaxes(0, 1)
        self.opts = (img, args, kargs)
        self.image = None
        self.update()

    def paintEvent(self, ev):
        if False:
            print('Hello World!')
        if self.opts is None:
            return
        if self.image is None:
            (argb, alpha) = fn.makeARGB(self.opts[0], *self.opts[1], **self.opts[2])
            if self._cp and self._cp.get_array_module(argb) == self._cp:
                argb = argb.get()
            self.image = fn.makeQImage(argb, alpha, copy=False, transpose=False)
            self.opts = ()
        p = QtGui.QPainter(self)
        if self.scaled:
            rect = self.rect()
            ar = rect.width() / float(rect.height())
            imar = self.image.width() / float(self.image.height())
            if ar > imar:
                rect.setWidth(int(rect.width() * imar / ar))
            else:
                rect.setHeight(int(rect.height() * ar / imar))
            p.drawImage(rect, self.image)
        else:
            p.drawImage(QtCore.QPointF(), self.image)
        p.end()
if HAVE_OPENGL:
    __all__.append('RawImageGLWidget')

    class RawImageGLWidget(QOpenGLWidget):
        """
        Similar to RawImageWidget, but uses a GL widget to do all drawing.
        Performance varies between platforms; see examples/VideoSpeedTest for benchmarking.

        Checks if setConfigOptions(imageAxisOrder='row-major') was set.
        """

        def __init__(self, parent=None, scaled=False):
            if False:
                return 10
            QOpenGLWidget.__init__(self, parent)
            self.scaled = scaled
            self.image = None
            self.uploaded = False
            self.smooth = False
            self.opts = None

        def setImage(self, img, *args, **kargs):
            if False:
                for i in range(10):
                    print('nop')
            '\n            img must be ndarray of shape (x,y), (x,y,3), or (x,y,4).\n            Extra arguments are sent to functions.makeARGB\n            '
            if getConfigOption('imageAxisOrder') == 'col-major':
                img = img.swapaxes(0, 1)
            self.opts = (img, args, kargs)
            self.image = None
            self.uploaded = False
            self.update()

        def initializeGL(self):
            if False:
                for i in range(10):
                    print('nop')
            self.texture = glGenTextures(1)

        def uploadTexture(self):
            if False:
                print('Hello World!')
            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, self.texture)
            if self.smooth:
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            else:
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
            (h, w) = self.image.shape[:2]
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, self.image)
            glDisable(GL_TEXTURE_2D)
            self.uploaded = True

        def paintGL(self):
            if False:
                return 10
            glClear(GL_COLOR_BUFFER_BIT)
            if self.image is None:
                if self.opts is None:
                    return
                (img, args, kwds) = self.opts
                kwds['useRGBA'] = True
                (self.image, _) = fn.makeARGB(img, *args, **kwds)
            if not self.uploaded:
                self.uploadTexture()
            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, self.texture)
            glColor4f(1, 1, 1, 1)
            glBegin(GL_QUADS)
            glTexCoord2f(0, 1)
            glVertex3f(-1, -1, 0)
            glTexCoord2f(1, 1)
            glVertex3f(1, -1, 0)
            glTexCoord2f(1, 0)
            glVertex3f(1, 1, 0)
            glTexCoord2f(0, 0)
            glVertex3f(-1, 1, 0)
            glEnd()
            glDisable(GL_TEXTURE_2D)