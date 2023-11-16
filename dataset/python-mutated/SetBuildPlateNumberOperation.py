from UM.Scene.SceneNode import SceneNode
from UM.Operations.Operation import Operation
from cura.Settings.SettingOverrideDecorator import SettingOverrideDecorator

class SetBuildPlateNumberOperation(Operation):
    """Simple operation to set the buildplate number of a scenenode."""

    def __init__(self, node: SceneNode, build_plate_nr: int) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__()
        self._node = node
        self._build_plate_nr = build_plate_nr
        self._previous_build_plate_nr = None
        self._decorator_added = False

    def undo(self) -> None:
        if False:
            i = 10
            return i + 15
        if self._previous_build_plate_nr:
            self._node.callDecoration('setBuildPlateNumber', self._previous_build_plate_nr)

    def redo(self) -> None:
        if False:
            print('Hello World!')
        stack = self._node.callDecoration('getStack')
        if not stack:
            self._node.addDecorator(SettingOverrideDecorator())
        self._previous_build_plate_nr = self._node.callDecoration('getBuildPlateNumber')
        self._node.callDecoration('setBuildPlateNumber', self._build_plate_nr)