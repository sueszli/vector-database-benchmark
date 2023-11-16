from typing import Optional, TYPE_CHECKING
from UM.Qt.QtApplication import QtApplication
from UM.Logger import Logger
from UM.Math.Vector import Vector
from UM.Resources import Resources
from UM.View.RenderPass import RenderPass
from UM.View.GL.OpenGL import OpenGL
from UM.View.GL.ShaderProgram import InvalidShaderProgramError
from UM.View.RenderBatch import RenderBatch
from UM.Scene.Iterator.DepthFirstIterator import DepthFirstIterator
if TYPE_CHECKING:
    from UM.View.GL.ShaderProgram import ShaderProgram

class PickingPass(RenderPass):
    """A :py:class:`Uranium.UM.View.RenderPass` subclass that renders a the distance of selectable objects from the
    active camera to a texture.

    The texture is used to map a 2d location (eg the mouse location) to a world space position

    .. note:: that in order to increase precision, the 24 bit depth value is encoded into all three of the R,G & B channels
    """

    def __init__(self, width: int, height: int) -> None:
        if False:
            print('Hello World!')
        super().__init__('picking', width, height)
        self._renderer = QtApplication.getInstance().getRenderer()
        self._shader = None
        self._scene = QtApplication.getInstance().getController().getScene()

    def render(self) -> None:
        if False:
            print('Hello World!')
        if not self._shader:
            try:
                self._shader = OpenGL.getInstance().createShaderProgram(Resources.getPath(Resources.Shaders, 'camera_distance.shader'))
            except InvalidShaderProgramError:
                Logger.error('Unable to compile shader program: camera_distance.shader')
                return
        (width, height) = self.getSize()
        self._gl.glViewport(0, 0, width, height)
        self._gl.glClearColor(1.0, 1.0, 1.0, 0.0)
        self._gl.glClear(self._gl.GL_COLOR_BUFFER_BIT | self._gl.GL_DEPTH_BUFFER_BIT)
        batch = RenderBatch(self._shader)
        for node in DepthFirstIterator(self._scene.getRoot()):
            if node.callDecoration('isSliceable') and node.getMeshData() and node.isVisible():
                batch.addItem(node.getWorldTransformation(copy=False), node.getMeshData(), normal_transformation=node.getCachedNormalMatrix())
        self.bind()
        batch.render(self._scene.getActiveCamera())
        self.release()

    def getPickedDepth(self, x: int, y: int) -> float:
        if False:
            for i in range(10):
                print('nop')
        'Get the distance in mm from the camera to at a certain pixel coordinate.\n\n        :param x: x component of coordinate vector in pixels\n        :param y: y component of coordinate vector in pixels\n        :return: distance in mm from the camera to pixel coordinate\n        '
        output = self.getOutput()
        window_size = self._renderer.getWindowSize()
        px = int((0.5 + x / 2.0) * window_size[0])
        py = int((0.5 + y / 2.0) * window_size[1])
        if px < 0 or px > output.width() - 1 or py < 0 or (py > output.height() - 1):
            return -1
        distance = output.pixel(px, py)
        distance = (distance & 16777215) / 1000.0
        return distance

    def getPickedPosition(self, x: int, y: int) -> Vector:
        if False:
            for i in range(10):
                print('nop')
        'Get the world coordinates of a picked point\n\n        :param x: x component of coordinate vector in pixels\n        :param y: y component of coordinate vector in pixels\n        :return: vector of the world coordinate\n        '
        distance = self.getPickedDepth(x, y)
        camera = self._scene.getActiveCamera()
        if camera:
            return camera.getRay(x, y).getPointAlongRay(distance)
        return Vector()