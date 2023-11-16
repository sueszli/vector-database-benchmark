from PyQt6.QtCore import QTimer
from UM.Application import Application
from UM.Logger import Logger
from UM.Scene.SceneNode import SceneNode
from UM.Scene.Iterator.BreadthFirstIterator import BreadthFirstIterator
from UM.Math.Vector import Vector
from UM.Scene.Selection import Selection
from UM.Scene.SceneNodeSettings import SceneNodeSettings
from cura.Scene.ConvexHullDecorator import ConvexHullDecorator
from cura.Operations import PlatformPhysicsOperation
from cura.Scene import ZOffsetDecorator
import random

class PlatformPhysics:

    def __init__(self, controller, volume):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self._controller = controller
        self._controller.getScene().sceneChanged.connect(self._onSceneChanged)
        self._controller.toolOperationStarted.connect(self._onToolOperationStarted)
        self._controller.toolOperationStopped.connect(self._onToolOperationStopped)
        self._build_volume = volume
        self._enabled = True
        self._change_timer = QTimer()
        self._change_timer.setInterval(100)
        self._change_timer.setSingleShot(True)
        self._change_timer.timeout.connect(self._onChangeTimerFinished)
        self._move_factor = 1.1
        self._max_overlap_checks = 10
        self._minimum_gap = 2
        Application.getInstance().getPreferences().addPreference('physics/automatic_push_free', False)
        Application.getInstance().getPreferences().addPreference('physics/automatic_drop_down', True)

    def _onSceneChanged(self, source):
        if False:
            while True:
                i = 10
        if not source.callDecoration('isSliceable'):
            return
        self._change_timer.start()

    def _onChangeTimerFinished(self):
        if False:
            return 10
        if not self._enabled:
            return
        app_instance = Application.getInstance()
        app_preferences = app_instance.getPreferences()
        app_automatic_drop_down = app_preferences.getValue('physics/automatic_drop_down')
        app_automatic_push_free = app_preferences.getValue('physics/automatic_push_free')
        root = self._controller.getScene().getRoot()
        build_volume = app_instance.getBuildVolume()
        build_volume.updateNodeBoundaryCheck()
        transformed_nodes = []
        nodes = list(BreadthFirstIterator(root))
        nodes = [node for node in nodes if hasattr(node, '_outside_buildarea') and (not node._outside_buildarea)]
        random.shuffle(nodes)
        for node in nodes:
            if node is root or not isinstance(node, SceneNode) or node.getBoundingBox() is None:
                continue
            bbox = node.getBoundingBox()
            move_vector = Vector()
            if node.getSetting(SceneNodeSettings.AutoDropDown, app_automatic_drop_down) and (not (node.getParent() and node.getParent().callDecoration('isGroup') or node.getParent() != root)) and node.isEnabled():
                z_offset = node.callDecoration('getZOffset') if node.getDecorator(ZOffsetDecorator.ZOffsetDecorator) else 0
                move_vector = move_vector.set(y=-bbox.bottom + z_offset)
            if not node.getDecorator(ConvexHullDecorator) and (not node.callDecoration('isNonPrintingMesh')) and (node.callDecoration('getLayerData') is None):
                node.addDecorator(ConvexHullDecorator())
            if not node.callDecoration('isNonPrintingMesh') and app_automatic_push_free:
                if node.getSetting(SceneNodeSettings.LockPosition):
                    continue
                for other_node in BreadthFirstIterator(root):
                    if other_node is root or not issubclass(type(other_node), SceneNode) or other_node is node or (other_node.callDecoration('getBuildPlateNumber') != node.callDecoration('getBuildPlateNumber')):
                        continue
                    if other_node in node.getAllChildren() or node in other_node.getAllChildren():
                        continue
                    if other_node.getParent() and node.getParent() and (other_node.getParent().callDecoration('isGroup') is not None or node.getParent().callDecoration('isGroup') is not None):
                        continue
                    if not other_node.callDecoration('getConvexHull') or not other_node.getBoundingBox():
                        continue
                    if other_node in transformed_nodes:
                        continue
                    if other_node.callDecoration('isNonPrintingMesh'):
                        continue
                    overlap = (0, 0)
                    current_overlap_checks = 0
                    while overlap and current_overlap_checks < self._max_overlap_checks:
                        current_overlap_checks += 1
                        head_hull = node.callDecoration('getConvexHullHead')
                        if head_hull:
                            overlap = head_hull.translate(move_vector.x, move_vector.z).intersectsPolygon(other_node.callDecoration('getConvexHull'))
                            if not overlap:
                                other_head_hull = other_node.callDecoration('getConvexHullHead')
                                if other_head_hull:
                                    overlap = node.callDecoration('getConvexHull').translate(move_vector.x, move_vector.z).intersectsPolygon(other_head_hull)
                                    if overlap:
                                        move_vector = move_vector.set(x=move_vector.x + overlap[0] * self._move_factor, z=move_vector.z + overlap[1] * self._move_factor)
                            else:
                                move_vector = move_vector.set(x=move_vector.x + overlap[0] * self._move_factor, z=move_vector.z + overlap[1] * self._move_factor)
                        else:
                            own_convex_hull = node.callDecoration('getConvexHull')
                            other_convex_hull = other_node.callDecoration('getConvexHull')
                            if own_convex_hull and other_convex_hull:
                                overlap = own_convex_hull.translate(move_vector.x, move_vector.z).intersectsPolygon(other_convex_hull)
                                if overlap:
                                    temp_move_vector = move_vector.set(x=move_vector.x + overlap[0] * self._move_factor, z=move_vector.z + overlap[1] * self._move_factor)
                                    if abs(temp_move_vector.x - overlap[0]) < self._minimum_gap and abs(temp_move_vector.y - overlap[1]) < self._minimum_gap:
                                        temp_x_factor = (abs(overlap[0]) + self._minimum_gap) / overlap[0] if overlap[0] != 0 else 0
                                        temp_y_factor = (abs(overlap[1]) + self._minimum_gap) / overlap[1] if overlap[1] != 0 else 0
                                        temp_scale_factor = temp_x_factor if abs(temp_x_factor) > abs(temp_y_factor) else temp_y_factor
                                        move_vector = move_vector.set(x=move_vector.x + overlap[0] * temp_scale_factor, z=move_vector.z + overlap[1] * temp_scale_factor)
                                    else:
                                        move_vector = temp_move_vector
                            else:
                                overlap = None
            if not Vector.Null.equals(move_vector, epsilon=1e-05):
                transformed_nodes.append(node)
                op = PlatformPhysicsOperation.PlatformPhysicsOperation(node, move_vector)
                op.push()
        build_volume.updateNodeBoundaryCheck()

    def _onToolOperationStarted(self, tool):
        if False:
            for i in range(10):
                print('nop')
        self._enabled = False

    def _onToolOperationStopped(self, tool):
        if False:
            i = 10
            return i + 15
        if tool.getPluginId() == 'SelectionTool':
            return
        if tool.getPluginId() == 'TranslateTool':
            for node in Selection.getAllSelectedObjects():
                if node.getBoundingBox() and node.getBoundingBox().bottom < 0:
                    if not node.getDecorator(ZOffsetDecorator.ZOffsetDecorator):
                        node.addDecorator(ZOffsetDecorator.ZOffsetDecorator())
                    node.callDecoration('setZOffset', node.getBoundingBox().bottom)
                elif node.getDecorator(ZOffsetDecorator.ZOffsetDecorator):
                    node.removeDecorator(ZOffsetDecorator.ZOffsetDecorator)
        self._enabled = True
        self._onChangeTimerFinished()