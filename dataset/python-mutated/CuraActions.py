from typing import List, cast
from PyQt6.QtCore import QObject, QUrl, QMimeData
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import QApplication
from UM.Event import CallFunctionEvent
from UM.FlameProfiler import pyqtSlot
from UM.Math.Vector import Vector
from UM.Scene.Selection import Selection
from UM.Scene.Iterator.BreadthFirstIterator import BreadthFirstIterator
from UM.Scene.Iterator.DepthFirstIterator import DepthFirstIterator
from UM.Operations.GroupedOperation import GroupedOperation
from UM.Operations.RemoveSceneNodeOperation import RemoveSceneNodeOperation
from UM.Operations.TranslateOperation import TranslateOperation
import cura.CuraApplication
from cura.Operations.SetParentOperation import SetParentOperation
from cura.MultiplyObjectsJob import MultiplyObjectsJob
from cura.Settings.SetObjectExtruderOperation import SetObjectExtruderOperation
from cura.Settings.ExtruderManager import ExtruderManager
from cura.Arranging.GridArrange import GridArrange
from cura.Arranging.Nest2DArrange import Nest2DArrange
from cura.Operations.SetBuildPlateNumberOperation import SetBuildPlateNumberOperation
from UM.Logger import Logger
from UM.Scene.SceneNode import SceneNode

class CuraActions(QObject):

    def __init__(self, parent: QObject=None) -> None:
        if False:
            while True:
                i = 10
        super().__init__(parent)

    @pyqtSlot()
    def openDocumentation(self) -> None:
        if False:
            print('Hello World!')
        event = CallFunctionEvent(self._openUrl, [QUrl('https://ultimaker.com/en/resources/manuals/software?utm_source=cura&utm_medium=software&utm_campaign=dropdown-documentation')], {})
        cura.CuraApplication.CuraApplication.getInstance().functionEvent(event)

    @pyqtSlot()
    def openBugReportPage(self) -> None:
        if False:
            print('Hello World!')
        event = CallFunctionEvent(self._openUrl, [QUrl('https://github.com/Ultimaker/Cura/issues/new/choose')], {})
        cura.CuraApplication.CuraApplication.getInstance().functionEvent(event)

    @pyqtSlot()
    def homeCamera(self) -> None:
        if False:
            i = 10
            return i + 15
        'Reset camera position and direction to default'
        scene = cura.CuraApplication.CuraApplication.getInstance().getController().getScene()
        camera = scene.getActiveCamera()
        if camera:
            diagonal_size = cura.CuraApplication.CuraApplication.getInstance().getBuildVolume().getDiagonalSize()
            camera.setPosition(Vector(-80, 250, 700) * diagonal_size / 375)
            camera.setPerspective(True)
            camera.lookAt(Vector(0, 0, 0))

    @pyqtSlot()
    def centerSelection(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Center all objects in the selection'
        operation = GroupedOperation()
        for node in Selection.getAllSelectedObjects():
            current_node = node
            parent_node = current_node.getParent()
            while parent_node and parent_node.callDecoration('isGroup'):
                current_node = parent_node
                parent_node = current_node.getParent()
            bbox = current_node.getBoundingBox()
            if bbox:
                center_y = current_node.getWorldPosition().y - bbox.bottom
            else:
                center_y = 0
            center_operation = TranslateOperation(current_node, Vector(0, center_y, 0), set_position=True)
            operation.addOperation(center_operation)
        operation.push()

    @pyqtSlot(int)
    def multiplySelection(self, count: int) -> None:
        if False:
            while True:
                i = 10
        'Multiply all objects in the selection\n        :param count: The number of times to multiply the selection.\n        '
        min_offset = cura.CuraApplication.CuraApplication.getInstance().getBuildVolume().getEdgeDisallowedSize() + 2
        job = MultiplyObjectsJob(Selection.getAllSelectedObjects(), count, min_offset=max(min_offset, 8))
        job.start()

    @pyqtSlot(int)
    def multiplySelectionToGrid(self, count: int) -> None:
        if False:
            return 10
        'Multiply all objects in the selection\n\n        :param count: The number of times to multiply the selection.\n        '
        min_offset = cura.CuraApplication.CuraApplication.getInstance().getBuildVolume().getEdgeDisallowedSize() + 2
        job = MultiplyObjectsJob(Selection.getAllSelectedObjects(), count, min_offset=max(min_offset, 8), grid_arrange=True)
        job.start()

    @pyqtSlot()
    def deleteSelection(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Delete all selected objects.'
        if not cura.CuraApplication.CuraApplication.getInstance().getController().getToolsEnabled():
            return
        removed_group_nodes = []
        op = GroupedOperation()
        nodes = Selection.getAllSelectedObjects()
        for node in nodes:
            op.addOperation(RemoveSceneNodeOperation(node))
            group_node = node.getParent()
            if group_node and group_node.callDecoration('isGroup') and (group_node not in removed_group_nodes):
                remaining_nodes_in_group = list(set(group_node.getChildren()) - set(nodes))
                if len(remaining_nodes_in_group) == 1:
                    removed_group_nodes.append(group_node)
                    op.addOperation(SetParentOperation(remaining_nodes_in_group[0], group_node.getParent()))
                    op.addOperation(RemoveSceneNodeOperation(group_node))
            cura.CuraApplication.CuraApplication.getInstance().getController().getScene().sceneChanged.emit(node)
        op.push()

    @pyqtSlot(str)
    def setExtruderForSelection(self, extruder_id: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Set the extruder that should be used to print the selection.\n\n        :param extruder_id: The ID of the extruder stack to use for the selected objects.\n        '
        operation = GroupedOperation()
        nodes_to_change = []
        for node in Selection.getAllSelectedObjects():
            if node.callDecoration('isGroup'):
                for grouped_node in BreadthFirstIterator(node):
                    if grouped_node.callDecoration('getActiveExtruder') == extruder_id:
                        continue
                    if grouped_node.callDecoration('isGroup'):
                        continue
                    nodes_to_change.append(grouped_node)
                continue
            if node.callDecoration('getActiveExtruder') == extruder_id:
                continue
            nodes_to_change.append(node)
        if not nodes_to_change:
            ExtruderManager.getInstance().resetSelectedObjectExtruders()
            return
        for node in nodes_to_change:
            operation.addOperation(SetObjectExtruderOperation(node, extruder_id))
        operation.push()

    @pyqtSlot(int)
    def setBuildPlateForSelection(self, build_plate_nr: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        Logger.log('d', 'Setting build plate number... %d' % build_plate_nr)
        operation = GroupedOperation()
        root = cura.CuraApplication.CuraApplication.getInstance().getController().getScene().getRoot()
        nodes_to_change = []
        for node in Selection.getAllSelectedObjects():
            parent_node = node
            while parent_node.getParent() != root:
                parent_node = cast(SceneNode, parent_node.getParent())
            for single_node in BreadthFirstIterator(parent_node):
                nodes_to_change.append(single_node)
        if not nodes_to_change:
            Logger.log('d', 'Nothing to change.')
            return
        for node in nodes_to_change:
            operation.addOperation(SetBuildPlateNumberOperation(node, build_plate_nr))
        operation.push()
        Selection.clear()

    @pyqtSlot()
    def cut(self) -> None:
        if False:
            print('Hello World!')
        self.copy()
        self.deleteSelection()

    @pyqtSlot()
    def copy(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        mesh_writer = cura.CuraApplication.CuraApplication.getInstance().getMeshFileHandler().getWriter('3MFWriter')
        if not mesh_writer:
            Logger.log('e', 'No 3MF writer found, unable to copy.')
            return
        selected_objects = Selection.getAllSelectedObjects()
        scene_string = mesh_writer.sceneNodesToString(selected_objects)
        QApplication.clipboard().setText(scene_string)

    @pyqtSlot()
    def paste(self) -> None:
        if False:
            return 10
        application = cura.CuraApplication.CuraApplication.getInstance()
        mesh_reader = application.getMeshFileHandler().getReaderForFile('.3mf')
        if not mesh_reader:
            Logger.log('e', 'No 3MF reader found, unable to paste.')
            return
        scene_string = QApplication.clipboard().text()
        nodes = mesh_reader.stringToSceneNodes(scene_string)
        if not nodes:
            return
        fixed_nodes = []
        root = application.getController().getScene().getRoot()
        for node in DepthFirstIterator(root):
            if node.callDecoration('isSliceable'):
                fixed_nodes.append(node)
        arranger = GridArrange(nodes, application.getBuildVolume(), fixed_nodes)
        (group_operation, not_fit_count) = arranger.createGroupOperationForArrange(add_new_nodes_in_scene=True)
        group_operation.push()
        for node in Selection.getAllSelectedObjects():
            Selection.remove(node)
        for node in nodes:
            Selection.add(node)

    def _openUrl(self, url: QUrl) -> None:
        if False:
            print('Hello World!')
        QDesktopServices.openUrl(url)