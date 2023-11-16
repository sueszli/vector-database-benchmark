from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QApplication
from UM.Application import Application
from UM.Math.Vector import Vector
from UM.Operations.TranslateOperation import TranslateOperation
from UM.Tool import Tool
from UM.Event import Event, MouseEvent
from UM.Mesh.MeshBuilder import MeshBuilder
from UM.Scene.Selection import Selection
from cura.CuraApplication import CuraApplication
from cura.Scene.CuraSceneNode import CuraSceneNode
from cura.PickingPass import PickingPass
from UM.Operations.GroupedOperation import GroupedOperation
from UM.Operations.AddSceneNodeOperation import AddSceneNodeOperation
from UM.Operations.RemoveSceneNodeOperation import RemoveSceneNodeOperation
from cura.Operations.SetParentOperation import SetParentOperation
from cura.Scene.SliceableObjectDecorator import SliceableObjectDecorator
from cura.Scene.BuildPlateDecorator import BuildPlateDecorator
from UM.Settings.SettingInstance import SettingInstance
import numpy

class SupportEraser(Tool):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self._shortcut_key = Qt.Key.Key_E
        self._controller = self.getController()
        self._selection_pass = None
        CuraApplication.getInstance().globalContainerStackChanged.connect(self._updateEnabled)
        Selection.selectionChanged.connect(self._onSelectionChanged)
        self._had_selection = False
        self._skip_press = False
        self._had_selection_timer = QTimer()
        self._had_selection_timer.setInterval(0)
        self._had_selection_timer.setSingleShot(True)
        self._had_selection_timer.timeout.connect(self._selectionChangeDelay)

    def event(self, event):
        if False:
            return 10
        super().event(event)
        modifiers = QApplication.keyboardModifiers()
        ctrl_is_active = modifiers & Qt.KeyboardModifier.ControlModifier
        if event.type == Event.MousePressEvent and MouseEvent.LeftButton in event.buttons and self._controller.getToolsEnabled():
            if ctrl_is_active:
                self._controller.setActiveTool('TranslateTool')
                return
            if self._skip_press:
                self._skip_press = False
                return
            if self._selection_pass is None:
                self._selection_pass = Application.getInstance().getRenderer().getRenderPass('selection')
            picked_node = self._controller.getScene().findObject(self._selection_pass.getIdAtPosition(event.x, event.y))
            if not picked_node:
                return
            node_stack = picked_node.callDecoration('getStack')
            if node_stack:
                if node_stack.getProperty('anti_overhang_mesh', 'value'):
                    self._removeEraserMesh(picked_node)
                    return
                elif node_stack.getProperty('support_mesh', 'value') or node_stack.getProperty('infill_mesh', 'value') or node_stack.getProperty('cutting_mesh', 'value'):
                    return
            active_camera = self._controller.getScene().getActiveCamera()
            picking_pass = PickingPass(active_camera.getViewportWidth(), active_camera.getViewportHeight())
            picking_pass.render()
            picked_position = picking_pass.getPickedPosition(event.x, event.y)
            self._createEraserMesh(picked_node, picked_position)

    def _createEraserMesh(self, parent: CuraSceneNode, position: Vector):
        if False:
            return 10
        node = CuraSceneNode()
        node.setName('Eraser')
        node.setSelectable(True)
        node.setCalculateBoundingBox(True)
        mesh = self._createCube(10)
        node.setMeshData(mesh.build())
        node.calculateBoundingBoxMesh()
        active_build_plate = CuraApplication.getInstance().getMultiBuildPlateModel().activeBuildPlate
        node.addDecorator(BuildPlateDecorator(active_build_plate))
        node.addDecorator(SliceableObjectDecorator())
        stack = node.callDecoration('getStack')
        settings = stack.getTop()
        definition = stack.getSettingDefinition('anti_overhang_mesh')
        new_instance = SettingInstance(definition, settings)
        new_instance.setProperty('value', True)
        new_instance.resetState()
        settings.addInstance(new_instance)
        op = GroupedOperation()
        op.addOperation(AddSceneNodeOperation(node, self._controller.getScene().getRoot()))
        op.addOperation(SetParentOperation(node, parent))
        op.addOperation(TranslateOperation(node, position, set_position=True))
        op.push()
        CuraApplication.getInstance().getController().getScene().sceneChanged.emit(node)

    def _removeEraserMesh(self, node: CuraSceneNode):
        if False:
            return 10
        parent = node.getParent()
        if parent == self._controller.getScene().getRoot():
            parent = None
        op = RemoveSceneNodeOperation(node)
        op.push()
        if parent and (not Selection.isSelected(parent)):
            Selection.add(parent)
        CuraApplication.getInstance().getController().getScene().sceneChanged.emit(node)

    def _updateEnabled(self):
        if False:
            print('Hello World!')
        plugin_enabled = False
        global_container_stack = CuraApplication.getInstance().getGlobalContainerStack()
        if global_container_stack:
            plugin_enabled = global_container_stack.getProperty('anti_overhang_mesh', 'enabled')
        CuraApplication.getInstance().getController().toolEnabledChanged.emit(self._plugin_id, plugin_enabled)

    def _onSelectionChanged(self):
        if False:
            while True:
                i = 10
        if Selection.hasSelection() != self._had_selection:
            self._had_selection_timer.start()

    def _selectionChangeDelay(self):
        if False:
            for i in range(10):
                print('nop')
        has_selection = Selection.hasSelection()
        if not has_selection and self._had_selection:
            self._skip_press = True
        else:
            self._skip_press = False
        self._had_selection = has_selection

    def _createCube(self, size):
        if False:
            return 10
        mesh = MeshBuilder()
        s = size / 2
        verts = [[-s, -s, s], [-s, s, s], [s, s, s], [s, -s, s], [-s, s, -s], [-s, -s, -s], [s, -s, -s], [s, s, -s], [s, -s, -s], [-s, -s, -s], [-s, -s, s], [s, -s, s], [-s, s, -s], [s, s, -s], [s, s, s], [-s, s, s], [-s, -s, s], [-s, -s, -s], [-s, s, -s], [-s, s, s], [s, -s, -s], [s, -s, s], [s, s, s], [s, s, -s]]
        mesh.setVertices(numpy.asarray(verts, dtype=numpy.float32))
        indices = []
        for i in range(0, 24, 4):
            indices.append([i, i + 2, i + 1])
            indices.append([i, i + 3, i + 2])
        mesh.setIndices(numpy.asarray(indices, dtype=numpy.int32))
        mesh.calculateNormals()
        return mesh