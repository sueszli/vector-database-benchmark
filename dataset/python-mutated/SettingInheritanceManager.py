from typing import List, Optional, TYPE_CHECKING
from PyQt6.QtCore import QObject, QTimer, pyqtProperty, pyqtSignal
from UM.FlameProfiler import pyqtSlot
from UM.Application import Application
from UM.Logger import Logger
from UM.Settings.ContainerStack import ContainerStack
from UM.Settings.Interfaces import ContainerInterface
from UM.Settings.SettingFunction import SettingFunction
from UM.Settings.SettingInstance import InstanceState
from cura.Settings.ExtruderManager import ExtruderManager
if TYPE_CHECKING:
    from cura.Settings.ExtruderStack import ExtruderStack
    from UM.Settings.SettingDefinition import SettingDefinition

class SettingInheritanceManager(QObject):

    def __init__(self, parent=None) -> None:
        if False:
            while True:
                i = 10
        super().__init__(parent)
        self._global_container_stack = None
        self._settings_with_inheritance_warning = []
        self._active_container_stack = None
        self._update_timer = QTimer()
        self._update_timer.setInterval(500)
        self._update_timer.setSingleShot(True)
        self._update_timer.timeout.connect(self._update)
        Application.getInstance().globalContainerStackChanged.connect(self._onGlobalContainerChanged)
        ExtruderManager.getInstance().activeExtruderChanged.connect(self._onActiveExtruderChanged)
        self._onGlobalContainerChanged()
        self._onActiveExtruderChanged()
    settingsWithIntheritanceChanged = pyqtSignal()

    @pyqtSlot(str, result='QStringList')
    def getChildrenKeysWithOverride(self, key: str) -> List[str]:
        if False:
            i = 10
            return i + 15
        'Get the keys of all children settings with an override.'
        if self._global_container_stack is None:
            return []
        definitions = self._global_container_stack.definition.findDefinitions(key=key)
        if not definitions:
            Logger.log('w', 'Could not find definition for key [%s]', key)
            return []
        result = []
        for key in definitions[0].getAllKeys():
            if key in self._settings_with_inheritance_warning:
                result.append(key)
        return result

    @pyqtSlot(str, str, result=bool)
    def hasOverrides(self, key: str, extruder_index: str):
        if False:
            print('Hello World!')
        return key in self.getOverridesForExtruder(key, extruder_index)

    @pyqtSlot(str, str, result='QStringList')
    def getOverridesForExtruder(self, key: str, extruder_index: str) -> List[str]:
        if False:
            print('Hello World!')
        if self._global_container_stack is None:
            return []
        result = []
        extruder_stack = ExtruderManager.getInstance().getExtruderStack(extruder_index)
        if not extruder_stack:
            Logger.log('w', 'Unable to find extruder for current machine with index %s', extruder_index)
            return result
        definitions = self._global_container_stack.definition.findDefinitions(key=key)
        if not definitions:
            Logger.log('w', 'Could not find definition for key [%s] (2)', key)
            return result
        for key in definitions[0].getAllKeys():
            if self._settingIsOverwritingInheritance(key, extruder_stack):
                result.append(key)
        return result

    @pyqtSlot(str)
    def manualRemoveOverride(self, key: str) -> None:
        if False:
            return 10
        if key in self._settings_with_inheritance_warning:
            self._settings_with_inheritance_warning.remove(key)
            self.settingsWithIntheritanceChanged.emit()

    @pyqtSlot()
    def scheduleUpdate(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._update_timer.start()

    def _onActiveExtruderChanged(self) -> None:
        if False:
            return 10
        new_active_stack = ExtruderManager.getInstance().getActiveExtruderStack()
        if not new_active_stack:
            self._active_container_stack = None
            return
        if new_active_stack != self._active_container_stack:
            if self._active_container_stack:
                self._active_container_stack.propertyChanged.disconnect(self._onPropertyChanged)
                self._active_container_stack.containersChanged.disconnect(self._onContainersChanged)
            self._active_container_stack = new_active_stack
            if self._active_container_stack is not None:
                self._active_container_stack.propertyChanged.connect(self._onPropertyChanged)
                self._active_container_stack.containersChanged.connect(self._onContainersChanged)
            self._update_timer.start()

    def _onPropertyChanged(self, key: str, property_name: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        if (property_name == 'value' or property_name == 'enabled') and self._global_container_stack:
            definitions = self._global_container_stack.definition.findDefinitions(key=key)
            if not definitions:
                return
            has_overwritten_inheritance = self._settingIsOverwritingInheritance(key)
            settings_with_inheritance_warning_changed = False
            if key not in self._settings_with_inheritance_warning and has_overwritten_inheritance:
                self._settings_with_inheritance_warning.append(key)
                settings_with_inheritance_warning_changed = True
            elif key in self._settings_with_inheritance_warning and (not has_overwritten_inheritance):
                self._settings_with_inheritance_warning.remove(key)
                settings_with_inheritance_warning_changed = True
            parent = definitions[0].parent
            if parent is not None:
                while parent.parent is not None:
                    parent = parent.parent
            else:
                parent = definitions[0]
            if parent.key not in self._settings_with_inheritance_warning and has_overwritten_inheritance:
                self._settings_with_inheritance_warning.append(parent.key)
                settings_with_inheritance_warning_changed = True
            elif parent.key in self._settings_with_inheritance_warning and (not has_overwritten_inheritance):
                if not self._recursiveCheck(parent):
                    self._settings_with_inheritance_warning.remove(parent.key)
                    settings_with_inheritance_warning_changed = True
            if settings_with_inheritance_warning_changed:
                self.settingsWithIntheritanceChanged.emit()

    def _recursiveCheck(self, definition: 'SettingDefinition') -> bool:
        if False:
            print('Hello World!')
        for child in definition.children:
            if child.key in self._settings_with_inheritance_warning:
                return True
            if child.children:
                if self._recursiveCheck(child):
                    return True
        return False

    @pyqtProperty('QVariantList', notify=settingsWithIntheritanceChanged)
    def settingsWithInheritanceWarning(self) -> List[str]:
        if False:
            i = 10
            return i + 15
        return self._settings_with_inheritance_warning

    def _settingIsOverwritingInheritance(self, key: str, stack: ContainerStack=None) -> bool:
        if False:
            return 10
        'Check if a setting has an inheritance function that is overwritten'
        has_setting_function = False
        if not stack:
            stack = self._active_container_stack
        if not stack:
            return False
        if self._active_container_stack is None:
            return False
        all_keys = self._active_container_stack.getAllKeys()
        containers = []
        has_user_state = stack.getProperty(key, 'state') == InstanceState.User
        'Check if the setting has a user state. If not, it is never overwritten.'
        if not has_user_state:
            return False
        if not stack.getProperty(key, 'enabled'):
            return False
        user_container = stack.getTop()
        'Also check if the top container is not a setting function (this happens if the inheritance is restored).'
        if user_container and isinstance(user_container.getProperty(key, 'value'), SettingFunction):
            return False
        while stack:
            containers.extend(stack.getContainers())
            stack = stack.getNextStack()
        has_non_function_value = False
        for container in containers:
            try:
                value = container.getProperty(key, 'value')
            except AttributeError:
                continue
            if value is not None:
                has_setting_function = isinstance(value, SettingFunction)
                if has_setting_function:
                    for setting_key in value.getUsedSettingKeys():
                        if setting_key in all_keys:
                            break
                    else:
                        has_setting_function = False
                if has_setting_function is False:
                    has_non_function_value = True
                    continue
            if has_setting_function:
                break
        return has_setting_function and has_non_function_value

    def _update(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._settings_with_inheritance_warning = []
        if self._global_container_stack is None:
            return
        for setting_key in self._global_container_stack.getAllKeys():
            override = self._settingIsOverwritingInheritance(setting_key)
            if override:
                self._settings_with_inheritance_warning.append(setting_key)
        for category in self._global_container_stack.definition.findDefinitions(type='category'):
            if self._recursiveCheck(category):
                self._settings_with_inheritance_warning.append(category.key)
        self.settingsWithIntheritanceChanged.emit()

    def _onGlobalContainerChanged(self) -> None:
        if False:
            return 10
        if self._global_container_stack:
            self._global_container_stack.propertyChanged.disconnect(self._onPropertyChanged)
            self._global_container_stack.containersChanged.disconnect(self._onContainersChanged)
        self._global_container_stack = Application.getInstance().getGlobalContainerStack()
        if self._global_container_stack:
            self._global_container_stack.containersChanged.connect(self._onContainersChanged)
            self._global_container_stack.propertyChanged.connect(self._onPropertyChanged)
        self._onActiveExtruderChanged()

    def _onContainersChanged(self, container):
        if False:
            return 10
        self._update_timer.start()

    @staticmethod
    def createSettingInheritanceManager(engine=None, script_engine=None):
        if False:
            return 10
        return SettingInheritanceManager()