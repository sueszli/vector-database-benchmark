from PyQt6.QtCore import QObject, pyqtSignal, pyqtProperty
from UM.Application import Application

class SimpleModeSettingsManager(QObject):

    def __init__(self, parent=None):
        if False:
            print('Hello World!')
        super().__init__(parent)
        self._machine_manager = Application.getInstance().getMachineManager()
        self._is_profile_customized = False
        self._is_profile_user_created = False
        self._machine_manager.activeStackValueChanged.connect(self._updateIsProfileCustomized)
        self._updateIsProfileCustomized()
    isProfileCustomizedChanged = pyqtSignal()

    @pyqtProperty(bool, notify=isProfileCustomizedChanged)
    def isProfileCustomized(self):
        if False:
            for i in range(10):
                print('nop')
        return self._is_profile_customized

    def _updateIsProfileCustomized(self):
        if False:
            i = 10
            return i + 15
        user_setting_keys = set()
        if not self._machine_manager.activeMachine:
            return False
        global_stack = self._machine_manager.activeMachine
        user_setting_keys.update(global_stack.userChanges.getAllKeys())
        if global_stack.extruderList:
            for extruder_stack in global_stack.extruderList:
                user_setting_keys.update(extruder_stack.userChanges.getAllKeys())
        has_customized_user_settings = len(user_setting_keys) > 0
        if has_customized_user_settings != self._is_profile_customized:
            self._is_profile_customized = has_customized_user_settings
            self.isProfileCustomizedChanged.emit()