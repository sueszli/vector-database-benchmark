from PyQt6.QtCore import pyqtSlot, pyqtProperty, QObject, pyqtSignal
from PyQt6.QtGui import QValidator
import os
import urllib
from UM.Resources import Resources
from UM.Settings.ContainerRegistry import ContainerRegistry
from UM.Settings.InstanceContainer import InstanceContainer

class MachineNameValidator(QObject):
    """Are machine names valid?

    Performs checks based on the length of the name.
    """

    def __init__(self, parent=None):
        if False:
            return 10
        super().__init__(parent)
        try:
            filename_max_length = os.statvfs(Resources.getDataStoragePath()).f_namemax
        except (AttributeError, EnvironmentError):
            filename_max_length = 255
        machine_name_max_length = filename_max_length - len('_current_settings.') - len(ContainerRegistry.getMimeTypeForContainer(InstanceContainer).preferredSuffix)
        maximum_special_characters = int(machine_name_max_length / 12)
        unescaped = '[a-zA-Z0-9_\\-\\.\\/]'
        self.machine_name_regex = '^[^\\.]((' + unescaped + '){0,12}|.){0,' + str(maximum_special_characters) + '}$'
    validationChanged = pyqtSignal()

    def validate(self, name):
        if False:
            print('Hello World!')
        "Check if a specified machine name is allowed.\n\n        :param name: The machine name to check.\n        :return: ``QValidator.Invalid`` if it's disallowed, or ``QValidator.Acceptable`` if it's allowed.\n        "
        try:
            filename_max_length = os.statvfs(Resources.getDataStoragePath()).f_namemax
        except AttributeError:
            filename_max_length = 255
        escaped_name = urllib.parse.quote_plus(name)
        current_settings_filename = escaped_name + '_current_settings.' + ContainerRegistry.getMimeTypeForContainer(InstanceContainer).preferredSuffix
        if len(current_settings_filename) > filename_max_length:
            return QValidator.Invalid
        return QValidator.Acceptable

    @pyqtSlot(str)
    def updateValidation(self, new_name):
        if False:
            i = 10
            return i + 15
        'Updates the validation state of a machine name text field.'
        is_valid = self.validate(new_name)
        if is_valid == QValidator.Acceptable:
            self.validation_regex = '^.*$'
        else:
            self.validation_regex = 'a^'
        self.validationChanged.emit()

    @pyqtProperty(str, notify=validationChanged)
    def machineNameRegex(self):
        if False:
            print('Hello World!')
        return str(self.machine_name_regex)