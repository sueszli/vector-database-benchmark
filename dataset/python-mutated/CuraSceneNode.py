from copy import deepcopy
from typing import cast, Dict, List, Optional
from UM.Application import Application
from UM.Math.AxisAlignedBox import AxisAlignedBox
from UM.Math.Polygon import Polygon
from UM.Scene.SceneNode import SceneNode
from UM.Scene.SceneNodeDecorator import SceneNodeDecorator
import cura.CuraApplication
from cura.Settings.ExtruderStack import ExtruderStack
from cura.Settings.SettingOverrideDecorator import SettingOverrideDecorator

class CuraSceneNode(SceneNode):
    """Scene nodes that are models are only seen when selecting the corresponding build plate

    Note that many other nodes can just be UM SceneNode objects.
    """

    def __init__(self, parent: Optional['SceneNode']=None, visible: bool=True, name: str='', no_setting_override: bool=False) -> None:
        if False:
            while True:
                i = 10
        super().__init__(parent=parent, visible=visible, name=name)
        if not no_setting_override:
            self.addDecorator(SettingOverrideDecorator())
        self._outside_buildarea = False

    def setOutsideBuildArea(self, new_value: bool) -> None:
        if False:
            return 10
        self._outside_buildarea = new_value

    def isOutsideBuildArea(self) -> bool:
        if False:
            print('Hello World!')
        return self._outside_buildarea or self.callDecoration('getBuildPlateNumber') < 0

    def isVisible(self) -> bool:
        if False:
            return 10
        return super().isVisible() and self.callDecoration('getBuildPlateNumber') == cura.CuraApplication.CuraApplication.getInstance().getMultiBuildPlateModel().activeBuildPlate

    def isSelectable(self) -> bool:
        if False:
            while True:
                i = 10
        return super().isSelectable() and self.callDecoration('getBuildPlateNumber') == cura.CuraApplication.CuraApplication.getInstance().getMultiBuildPlateModel().activeBuildPlate

    def isSupportMesh(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        per_mesh_stack = self.callDecoration('getStack')
        if not per_mesh_stack:
            return False
        return per_mesh_stack.getProperty('support_mesh', 'value')

    def getPrintingExtruder(self) -> Optional[ExtruderStack]:
        if False:
            return 10
        'Get the extruder used to print this node. If there is no active node, then the extruder in position zero is returned\n\n        TODO The best way to do it is by adding the setActiveExtruder decorator to every node when is loaded\n        '
        global_container_stack = Application.getInstance().getGlobalContainerStack()
        if global_container_stack is None:
            return None
        per_mesh_stack = self.callDecoration('getStack')
        extruders = global_container_stack.extruderList
        if per_mesh_stack:
            if per_mesh_stack.getProperty('support_mesh', 'value'):
                return extruders[int(global_container_stack.getExtruderPositionValueWithDefault('support_extruder_nr'))]
        extruder_id = self.callDecoration('getActiveExtruder')
        for extruder in extruders:
            if extruder_id is not None:
                if extruder_id == extruder.getId():
                    return extruder
            else:
                try:
                    if extruder.getMetaDataEntry('position', default='0') == '0':
                        return extruder
                except ValueError:
                    continue
        return None

    def getDiffuseColor(self) -> List[float]:
        if False:
            while True:
                i = 10
        'Return the color of the material used to print this model'
        printing_extruder = self.getPrintingExtruder()
        material_color = '#808080'
        if printing_extruder is not None and printing_extruder.material:
            material_color = printing_extruder.material.getMetaDataEntry('color_code', default=material_color)
        return [int(material_color[1:3], 16) / 255, int(material_color[3:5], 16) / 255, int(material_color[5:7], 16) / 255, 1.0]

    def collidesWithAreas(self, areas: List[Polygon]) -> bool:
        if False:
            while True:
                i = 10
        'Return if any area collides with the convex hull of this scene node'
        convex_hull = self.callDecoration('getPrintingArea')
        if convex_hull:
            if not convex_hull.isValid():
                return False
            for area in areas:
                overlap = convex_hull.intersectsPolygon(area)
                if overlap is None:
                    continue
                return True
        return False

    def _calculateAABB(self) -> None:
        if False:
            i = 10
            return i + 15
        'Override of SceneNode._calculateAABB to exclude non-printing-meshes from bounding box'
        self._aabb = None
        if self._mesh_data:
            self._aabb = self._mesh_data.getExtents(self.getWorldTransformation(copy=False))
        for child in self.getAllChildren():
            if child.callDecoration('isNonPrintingMesh'):
                continue
            child_bb = child.getBoundingBox()
            if child_bb is None or child_bb.minimum == child_bb.maximum:
                continue
            if self._aabb is None:
                self._aabb = child_bb
            else:
                self._aabb = self._aabb + child_bb
        if self._aabb is None:
            position = self.getWorldPosition()
            self._aabb = AxisAlignedBox(minimum=position, maximum=position)

    def __deepcopy__(self, memo: Dict[int, object]) -> 'CuraSceneNode':
        if False:
            print('Hello World!')
        'Taken from SceneNode, but replaced SceneNode with CuraSceneNode'
        copy = CuraSceneNode(no_setting_override=True)
        copy.setTransformation(self.getLocalTransformation(copy=False))
        copy.setMeshData(self._mesh_data)
        copy.setVisible(cast(bool, deepcopy(self._visible, memo)))
        copy.source_mime_type = cast(str, deepcopy(self.source_mime_type, memo))
        copy._selectable = cast(bool, deepcopy(self._selectable, memo))
        copy._name = cast(str, deepcopy(self._name, memo))
        for decorator in self._decorators:
            copy.addDecorator(cast(SceneNodeDecorator, deepcopy(decorator, memo)))
        for child in self._children:
            copy.addChild(cast(SceneNode, deepcopy(child, memo)))
        self.calculateBoundingBoxMesh()
        return copy

    def transformChanged(self) -> None:
        if False:
            while True:
                i = 10
        self._transformChanged()