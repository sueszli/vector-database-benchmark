from OpenGL.GL import *
from ... import QtGui
from ..GLGraphicsItem import GLGraphicsItem
__all__ = ['GLAxisItem']

class GLAxisItem(GLGraphicsItem):
    """
    **Bases:** :class:`GLGraphicsItem <pyqtgraph.opengl.GLGraphicsItem.GLGraphicsItem>`
    
    Displays three lines indicating origin and orientation of local coordinate system. 
    
    """

    def __init__(self, size=None, antialias=True, glOptions='translucent', parentItem=None):
        if False:
            print('Hello World!')
        super().__init__(parentItem=parentItem)
        if size is None:
            size = QtGui.QVector3D(1, 1, 1)
        self.antialias = antialias
        self.setSize(size=size)
        self.setGLOptions(glOptions)

    def setSize(self, x=None, y=None, z=None, size=None):
        if False:
            print('Hello World!')
        '\n        Set the size of the axes (in its local coordinate system; this does not affect the transform)\n        Arguments can be x,y,z or size=QVector3D().\n        '
        if size is not None:
            x = size.x()
            y = size.y()
            z = size.z()
        self.__size = [x, y, z]
        self.update()

    def size(self):
        if False:
            i = 10
            return i + 15
        return self.__size[:]

    def paint(self):
        if False:
            i = 10
            return i + 15
        self.setupGLState()
        if self.antialias:
            glEnable(GL_LINE_SMOOTH)
            glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glBegin(GL_LINES)
        (x, y, z) = self.size()
        glColor4f(0, 1, 0, 0.6)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, z)
        glColor4f(1, 1, 0, 0.6)
        glVertex3f(0, 0, 0)
        glVertex3f(0, y, 0)
        glColor4f(0, 0, 1, 0.6)
        glVertex3f(0, 0, 0)
        glVertex3f(x, 0, 0)
        glEnd()