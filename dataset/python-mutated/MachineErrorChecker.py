import time
from collections import deque
from PyQt6.QtCore import QObject, QTimer, pyqtSignal, pyqtProperty
from typing import Optional, Any, Set
from UM.Logger import Logger
from UM.Settings.SettingDefinition import SettingDefinition
from UM.Settings.Validator import ValidatorState
import cura.CuraApplication

class MachineErrorChecker(QObject):
    """This class performs setting error checks for the currently active machine.

    The whole error checking process is pretty heavy which can take ~0.5 secs, so it can cause GUI to lag. The idea
    here is to split the whole error check into small tasks, each of which only checks a single setting key in a
    stack. According to my profiling results, the maximal runtime for such a sub-task is <0.03 secs, which should be
    good enough. Moreover, if any changes happened to the machine, we can cancel the check in progress without wait
    for it to finish the complete work.
    """

    def __init__(self, parent: Optional[QObject]=None) -> None:
        if False:
            print('Hello World!')
        super().__init__(parent)
        self._global_stack = None
        self._has_errors = True
        self._error_keys = set()
        self._error_keys_in_progress = set()
        self._stacks_and_keys_to_check = None
        self._need_to_check = False
        self._check_in_progress = False
        self._application = cura.CuraApplication.CuraApplication.getInstance()
        self._machine_manager = self._application.getMachineManager()
        self._check_start_time = time.time()
        self._setCheckTimer()
        self._keys_to_check = set()
        self._num_keys_to_check_per_update = 10

    def initialize(self) -> None:
        if False:
            while True:
                i = 10
        self._error_check_timer.timeout.connect(self._rescheduleCheck)
        self._machine_manager.globalContainerChanged.connect(self._onMachineChanged)
        self._machine_manager.globalContainerChanged.connect(self.startErrorCheck)
        self._onMachineChanged()

    def _setCheckTimer(self) -> None:
        if False:
            print('Hello World!')
        'A QTimer to regulate error check frequency\n\n        This timer delays the starting of error check\n        so we can react less frequently if the user is frequently\n        changing settings.\n        '
        self._error_check_timer = QTimer(self)
        self._error_check_timer.setInterval(100)
        self._error_check_timer.setSingleShot(True)

    def _onMachineChanged(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        if self._global_stack:
            self._global_stack.propertyChanged.disconnect(self.startErrorCheckPropertyChanged)
            self._global_stack.containersChanged.disconnect(self.startErrorCheck)
            for extruder in self._global_stack.extruderList:
                extruder.propertyChanged.disconnect(self.startErrorCheckPropertyChanged)
                extruder.containersChanged.disconnect(self.startErrorCheck)
        self._global_stack = self._machine_manager.activeMachine
        if self._global_stack:
            self._global_stack.propertyChanged.connect(self.startErrorCheckPropertyChanged)
            self._global_stack.containersChanged.connect(self.startErrorCheck)
            for extruder in self._global_stack.extruderList:
                extruder.propertyChanged.connect(self.startErrorCheckPropertyChanged)
                extruder.containersChanged.connect(self.startErrorCheck)
    hasErrorUpdated = pyqtSignal()
    needToWaitForResultChanged = pyqtSignal()
    errorCheckFinished = pyqtSignal()

    @pyqtProperty(bool, notify=hasErrorUpdated)
    def hasError(self) -> bool:
        if False:
            i = 10
            return i + 15
        return self._has_errors

    @pyqtProperty(bool, notify=needToWaitForResultChanged)
    def needToWaitForResult(self) -> bool:
        if False:
            while True:
                i = 10
        return self._need_to_check or self._check_in_progress

    def startErrorCheckPropertyChanged(self, key: str, property_name: str) -> None:
        if False:
            while True:
                i = 10
        'Start the error check for property changed\n        this is separate from the startErrorCheck because it ignores a number property types\n\n        :param key:\n        :param property_name:\n        '
        if property_name != 'value':
            return
        self._keys_to_check.add(key)
        self.startErrorCheck()

    def startErrorCheck(self, *args: Any) -> None:
        if False:
            return 10
        'Starts the error check timer to schedule a new error check.\n\n        :param args:\n        '
        if not self._check_in_progress:
            self._need_to_check = True
            self.needToWaitForResultChanged.emit()
        self._error_check_timer.start()

    def _rescheduleCheck(self) -> None:
        if False:
            while True:
                i = 10
        'This function is called by the timer to reschedule a new error check.\n\n        If there is no check in progress, it will start a new one. If there is any, it sets the "_need_to_check" flag\n        to notify the current check to stop and start a new one.\n        '
        if self._check_in_progress and (not self._need_to_check):
            self._need_to_check = True
            self.needToWaitForResultChanged.emit()
            return
        self._error_keys_in_progress = set()
        self._need_to_check = False
        self.needToWaitForResultChanged.emit()
        global_stack = self._machine_manager.activeMachine
        if global_stack is None:
            Logger.log('i', 'No active machine, nothing to check.')
            return
        self._stacks_and_keys_to_check = deque()
        for stack in global_stack.extruderList:
            if not self._keys_to_check:
                self._keys_to_check = stack.getAllKeys()
            for key in self._keys_to_check:
                self._stacks_and_keys_to_check.append((stack, key))
        self._application.callLater(self._checkStack)
        self._check_start_time = time.time()
        Logger.log('d', 'New error check scheduled.')

    def _checkStack(self) -> None:
        if False:
            return 10
        if self._need_to_check:
            Logger.log('d', 'Need to check for errors again. Discard the current progress and reschedule a check.')
            self._check_in_progress = False
            self._application.callLater(self.startErrorCheck)
            return
        self._check_in_progress = True
        for i in range(self._num_keys_to_check_per_update):
            if not self._stacks_and_keys_to_check:
                self._setResult(False)
                return
            (stack, key) = self._stacks_and_keys_to_check.popleft()
            enabled = stack.getProperty(key, 'enabled')
            if not enabled:
                continue
            validation_state = stack.getProperty(key, 'validationState')
            if validation_state is None:
                definition = stack.getSettingDefinition(key)
                validator_type = SettingDefinition.getValidatorForType(definition.type)
                if validator_type:
                    validator = validator_type(key)
                    validation_state = validator(stack)
            if validation_state in (ValidatorState.Exception, ValidatorState.MaximumError, ValidatorState.MinimumError, ValidatorState.Invalid):
                keys_to_recheck = {setting_key for (stack, setting_key) in self._stacks_and_keys_to_check}
                keys_to_recheck.add(key)
                self._setResult(True, keys_to_recheck=keys_to_recheck)
                return
        self._application.callLater(self._checkStack)

    def _setResult(self, result: bool, keys_to_recheck=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        if result != self._has_errors:
            self._has_errors = result
            self.hasErrorUpdated.emit()
            self._machine_manager.stacksValidationChanged.emit()
        self._keys_to_check = keys_to_recheck if keys_to_recheck else set()
        self._need_to_check = False
        self._check_in_progress = False
        self.needToWaitForResultChanged.emit()
        self.errorCheckFinished.emit()
        execution_time = time.time() - self._check_start_time
        Logger.info(f'Error check finished, result = {result}, time = {execution_time:.2f}s')