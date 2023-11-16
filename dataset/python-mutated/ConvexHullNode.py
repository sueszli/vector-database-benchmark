from typing import Optional, TYPE_CHECKING
from UM.Application import Application
from UM.Math.Polygon import Polygon
from UM.Qt.QtApplication import QtApplication
from UM.Scene.SceneNode import SceneNode
from UM.Resources import Resources
from UM.Math.Color import Color
from UM.Mesh.MeshBuilder import MeshBuilder
from UM.View.GL.OpenGL import OpenGL
if TYPE_CHECKING:
    from UM.Mesh.MeshData import MeshData

class ConvexHullNode(SceneNode):
    shader = None

    def __init__(self, node: SceneNode, hull: Optional[Polygon], thickness: float, parent: Optional[SceneNode]=None) -> None:
        if False:
            while True:
                i = 10
        "Convex hull node is a special type of scene node that is used to display an area, to indicate the\n\n        location an object uses on the buildplate. This area (or area's in case of one at a time printing) is\n        then displayed as a transparent shadow. If the adhesion type is set to raft, the area is extruded\n        to represent the raft as well.\n        "
        super().__init__(parent)
        self.setCalculateBoundingBox(False)
        self._original_parent = parent
        if not Application.getInstance().getIsHeadLess():
            theme = QtApplication.getInstance().getTheme()
            if theme:
                self._color = Color(*theme.getColor('convex_hull').getRgb())
            else:
                self._color = Color(0, 0, 0)
        else:
            self._color = Color(0, 0, 0)
        self._mesh_height = 0.1
        self._thickness = thickness
        self._node = node
        self._convex_hull_head_mesh = None
        self._node.decoratorsChanged.connect(self._onNodeDecoratorsChanged)
        self._onNodeDecoratorsChanged(self._node)
        self._hull = hull
        if self._hull:
            hull_mesh_builder = MeshBuilder()
            if self._thickness == 0:
                if hull_mesh_builder.addConvexPolygon(self._hull.getPoints()[:], self._mesh_height, color=self._color):
                    hull_mesh_builder.resetNormals()
                    hull_mesh = hull_mesh_builder.build()
                    self.setMeshData(hull_mesh)
            elif hull_mesh_builder.addConvexPolygonExtrusion(self._hull.getPoints()[::-1], self._mesh_height - thickness, self._mesh_height, color=self._color):
                hull_mesh_builder.resetNormals()
                hull_mesh = hull_mesh_builder.build()
                self.setMeshData(hull_mesh)

    def getHull(self):
        if False:
            print('Hello World!')
        return self._hull

    def getThickness(self):
        if False:
            for i in range(10):
                print('nop')
        return self._thickness

    def getWatchedNode(self):
        if False:
            for i in range(10):
                print('nop')
        return self._node

    def render(self, renderer):
        if False:
            print('Hello World!')
        if not ConvexHullNode.shader:
            ConvexHullNode.shader = OpenGL.getInstance().createShaderProgram(Resources.getPath(Resources.Shaders, 'transparent_object.shader'))
            ConvexHullNode.shader.setUniformValue('u_diffuseColor', self._color)
            ConvexHullNode.shader.setUniformValue('u_opacity', 0.6)
        batch = renderer.getNamedBatch('convex_hull_node')
        if not batch:
            batch = renderer.createRenderBatch(transparent=True, shader=ConvexHullNode.shader, backface_cull=True, sort=-8)
            renderer.addRenderBatch(batch, name='convex_hull_node')
        batch.addItem(self.getWorldTransformation(copy=False), self.getMeshData())
        if self._convex_hull_head_mesh:
            renderer.queueNode(self, shader=ConvexHullNode.shader, transparent=True, mesh=self._convex_hull_head_mesh, backface_cull=True, sort=-8)
        return True

    def _onNodeDecoratorsChanged(self, node: SceneNode) -> None:
        if False:
            for i in range(10):
                print('nop')
        convex_hull_head = self._node.callDecoration('getConvexHullHeadFull')
        if convex_hull_head:
            convex_hull_head_builder = MeshBuilder()
            convex_hull_head_builder.addConvexPolygon(convex_hull_head.getPoints(), self._mesh_height - self._thickness)
            self._convex_hull_head_mesh = convex_hull_head_builder.build()