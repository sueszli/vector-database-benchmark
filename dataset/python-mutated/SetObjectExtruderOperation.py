from UM.Scene.SceneNode import SceneNode
from UM.Operations.Operation import Operation
from cura.Settings.SettingOverrideDecorator import SettingOverrideDecorator

class SetObjectExtruderOperation(Operation):
    """Simple operation to set the extruder a certain object should be printed with."""

    def __init__(self, node: SceneNode, extruder_id: str) -> None:
        if False:
            return 10
        self._node = node
        self._extruder_id = extruder_id
        self._previous_extruder_id = None
        self._decorator_added = False

    def undo(self):
        if False:
            return 10
        if self._previous_extruder_id:
            self._node.callDecoration('setActiveExtruder', self._previous_extruder_id)

    def redo(self):
        if False:
            for i in range(10):
                print('nop')
        stack = self._node.callDecoration('getStack')
        if not stack:
            self._node.addDecorator(SettingOverrideDecorator())
        self._previous_extruder_id = self._node.callDecoration('getActiveExtruder')
        self._node.callDecoration('setActiveExtruder', self._extruder_id)