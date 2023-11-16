from OpenGL.GL import *
import numpy as np
from ..MeshData import MeshData
from .GLMeshItem import GLMeshItem
__all__ = ['GLSurfacePlotItem']

class GLSurfacePlotItem(GLMeshItem):
    """
    **Bases:** :class:`GLMeshItem <pyqtgraph.opengl.GLMeshItem>`
    
    Displays a surface plot on a regular x,y grid
    """

    def __init__(self, x=None, y=None, z=None, colors=None, parentItem=None, **kwds):
        if False:
            for i in range(10):
                print('nop')
        '\n        The x, y, z, and colors arguments are passed to setData().\n        All other keyword arguments are passed to GLMeshItem.__init__().\n        '
        self._x = None
        self._y = None
        self._z = None
        self._color = None
        self._vertexes = None
        self._meshdata = MeshData()
        super().__init__(parentItem=parentItem, meshdata=self._meshdata, **kwds)
        self.setData(x, y, z, colors)

    def setData(self, x=None, y=None, z=None, colors=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Update the data in this surface plot. \n        \n        ==============  =====================================================================\n        **Arguments:**\n        x,y             1D arrays of values specifying the x,y positions of vertexes in the\n                        grid. If these are omitted, then the values will be assumed to be\n                        integers.\n        z               2D array of height values for each grid vertex.\n        colors          (width, height, 4) array of vertex colors.\n        ==============  =====================================================================\n        \n        All arguments are optional.\n        \n        Note that if vertex positions are updated, the normal vectors for each triangle must \n        be recomputed. This is somewhat expensive if the surface was initialized with smooth=False\n        and very expensive if smooth=True. For faster performance, initialize with \n        computeNormals=False and use per-vertex colors or a normal-independent shader program.\n        '
        if x is not None:
            if self._x is None or len(x) != len(self._x):
                self._vertexes = None
            self._x = x
        if y is not None:
            if self._y is None or len(y) != len(self._y):
                self._vertexes = None
            self._y = y
        if z is not None:
            if self._x is not None and z.shape[0] != len(self._x):
                raise Exception('Z values must have shape (len(x), len(y))')
            if self._y is not None and z.shape[1] != len(self._y):
                raise Exception('Z values must have shape (len(x), len(y))')
            self._z = z
            if self._vertexes is not None and self._z.shape != self._vertexes.shape[:2]:
                self._vertexes = None
        if colors is not None:
            self._colors = colors
            self._meshdata.setVertexColors(colors)
        if self._z is None:
            return
        updateMesh = False
        newVertexes = False
        if self._vertexes is None:
            newVertexes = True
            self._vertexes = np.empty((self._z.shape[0], self._z.shape[1], 3), dtype=float)
            self.generateFaces()
            self._meshdata.setFaces(self._faces)
            updateMesh = True
        if newVertexes or x is not None:
            if x is None:
                if self._x is None:
                    x = np.arange(self._z.shape[0])
                else:
                    x = self._x
            self._vertexes[:, :, 0] = x.reshape(len(x), 1)
            updateMesh = True
        if newVertexes or y is not None:
            if y is None:
                if self._y is None:
                    y = np.arange(self._z.shape[1])
                else:
                    y = self._y
            self._vertexes[:, :, 1] = y.reshape(1, len(y))
            updateMesh = True
        if newVertexes or z is not None:
            self._vertexes[..., 2] = self._z
            updateMesh = True
        if updateMesh:
            self._meshdata.setVertexes(self._vertexes.reshape(self._vertexes.shape[0] * self._vertexes.shape[1], 3))
            self.meshDataChanged()

    def generateFaces(self):
        if False:
            i = 10
            return i + 15
        cols = self._z.shape[1] - 1
        rows = self._z.shape[0] - 1
        faces = np.empty((cols * rows * 2, 3), dtype=np.uint)
        rowtemplate1 = np.arange(cols).reshape(cols, 1) + np.array([[0, 1, cols + 1]])
        rowtemplate2 = np.arange(cols).reshape(cols, 1) + np.array([[cols + 1, 1, cols + 2]])
        for row in range(rows):
            start = row * cols * 2
            faces[start:start + cols] = rowtemplate1 + row * (cols + 1)
            faces[start + cols:start + cols * 2] = rowtemplate2 + row * (cols + 1)
        self._faces = faces