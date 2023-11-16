from PyQt6.QtCore import pyqtProperty
from UM.FlameProfiler import pyqtSlot
from UM.Application import Application
from UM.PluginRegistry import PluginRegistry
from UM.Settings.ContainerRegistry import ContainerRegistry
from UM.Settings.SettingInstance import SettingInstance
from UM.Logger import Logger
import UM.Settings.Models.SettingVisibilityHandler
from cura.Settings.ExtruderManager import ExtruderManager
from cura.Settings.SettingOverrideDecorator import SettingOverrideDecorator

class PerObjectSettingVisibilityHandler(UM.Settings.Models.SettingVisibilityHandler.SettingVisibilityHandler):
    """The per object setting visibility handler ensures that only setting

    definitions that have a matching instance Container are returned as visible.
    """

    def __init__(self, parent=None, *args, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(*args, parent=parent, **kwargs)
        self._selected_object_id = None
        self._node = None
        self._stack = None
        PluginRegistry.getInstance().getPluginObject('PerObjectSettingsTool').visibility_handler = self
        self._skip_reset_setting_set = set()

    def setSelectedObjectId(self, id):
        if False:
            for i in range(10):
                print('nop')
        if id != self._selected_object_id:
            self._selected_object_id = id
            self._node = Application.getInstance().getController().getScene().findObject(self._selected_object_id)
            if self._node:
                self._stack = self._node.callDecoration('getStack')
            self.visibilityChanged.emit()

    @pyqtProperty('quint64', fset=setSelectedObjectId)
    def selectedObjectId(self):
        if False:
            for i in range(10):
                print('nop')
        return self._selected_object_id

    @pyqtSlot(str)
    def addSkipResetSetting(self, setting_name):
        if False:
            i = 10
            return i + 15
        self._skip_reset_setting_set.add(setting_name)

    def setVisible(self, visible):
        if False:
            return 10
        if not self._node:
            return
        if not self._stack:
            self._node.addDecorator(SettingOverrideDecorator())
            self._stack = self._node.callDecoration('getStack')
        settings = self._stack.getTop()
        all_instances = settings.findInstances()
        visibility_changed = False
        for instance in all_instances:
            if instance.definition.key in self._skip_reset_setting_set:
                continue
            if instance.definition.key not in visible:
                settings.removeInstance(instance.definition.key)
                visibility_changed = True
        for item in visible:
            if settings.getInstance(item) is not None:
                continue
            definition = self._stack.getSettingDefinition(item)
            if not definition:
                Logger.log('w', f"Unable to add SettingInstance ({item}) to the per-object visibility because we couldn't find the matching SettingDefinition.")
                continue
            new_instance = SettingInstance(definition, settings)
            stack_nr = -1
            stack = None
            if self._stack.getProperty('machine_extruder_count', 'value') > 1:
                if definition.limit_to_extruder != '-1':
                    stack_nr = str(int(round(float(self._stack.getProperty(item, 'limit_to_extruder')))))
                if stack_nr not in ExtruderManager.getInstance().extruderIds and self._stack.getProperty('extruder_nr', 'value') is not None:
                    stack_nr = -1
                if stack_nr in ExtruderManager.getInstance().extruderIds:
                    stack = ContainerRegistry.getInstance().findContainerStacks(id=ExtruderManager.getInstance().extruderIds[stack_nr])[0]
            else:
                stack = self._stack
            if stack is not None:
                new_instance.setProperty('value', stack.getRawProperty(item, 'value'))
            else:
                new_instance.setProperty('value', None)
            new_instance.resetState()
            settings.addInstance(new_instance)
            visibility_changed = True
        if visibility_changed:
            self.visibilityChanged.emit()

    def getVisible(self):
        if False:
            print('Hello World!')
        visible_settings = set()
        if not self._node:
            return visible_settings
        if not self._stack:
            return visible_settings
        settings = self._stack.getTop()
        if not settings:
            return visible_settings
        visible_settings = set(map(lambda i: i.definition.key, settings.findInstances()))
        return visible_settings