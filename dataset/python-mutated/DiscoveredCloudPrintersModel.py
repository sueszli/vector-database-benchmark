from typing import Optional, TYPE_CHECKING, List, Dict
from PyQt6.QtCore import QObject, pyqtSlot, Qt, pyqtSignal, pyqtProperty
from UM.Qt.ListModel import ListModel
if TYPE_CHECKING:
    from cura.CuraApplication import CuraApplication

class DiscoveredCloudPrintersModel(ListModel):
    """Model used to inform the application about newly added cloud printers, which are discovered from the user's
     account """
    DeviceKeyRole = Qt.ItemDataRole.UserRole + 1
    DeviceNameRole = Qt.ItemDataRole.UserRole + 2
    DeviceTypeRole = Qt.ItemDataRole.UserRole + 3
    DeviceFirmwareVersionRole = Qt.ItemDataRole.UserRole + 4
    cloudPrintersDetectedChanged = pyqtSignal(bool)

    def __init__(self, application: 'CuraApplication', parent: Optional['QObject']=None) -> None:
        if False:
            while True:
                i = 10
        super().__init__(parent)
        self.addRoleName(self.DeviceKeyRole, 'key')
        self.addRoleName(self.DeviceNameRole, 'name')
        self.addRoleName(self.DeviceTypeRole, 'machine_type')
        self.addRoleName(self.DeviceFirmwareVersionRole, 'firmware_version')
        self._discovered_cloud_printers_list = []
        self._application = application

    def addDiscoveredCloudPrinters(self, new_devices: List[Dict[str, str]]) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Adds all the newly discovered cloud printers into the DiscoveredCloudPrintersModel.\n\n        Example new_devices entry:\n\n        .. code-block:: python\n\n        {\n            "key": "YjW8pwGYcaUvaa0YgVyWeFkX3z",\n            "name": "NG 001",\n            "machine_type": "Ultimaker S5",\n            "firmware_version": "5.5.12.202001"\n        }\n\n        :param new_devices: List of dictionaries which contain information about added cloud printers.\n\n        :return: None\n        '
        self._discovered_cloud_printers_list.extend(new_devices)
        self._update()
        self.cloudPrintersDetectedChanged.emit(len(new_devices) > 0)

    @pyqtSlot()
    def clear(self) -> None:
        if False:
            i = 10
            return i + 15
        'Clears the contents of the DiscoveredCloudPrintersModel.\n\n        :return: None\n        '
        self._discovered_cloud_printers_list = []
        self._update()
        self.cloudPrintersDetectedChanged.emit(False)

    def _update(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Sorts the newly discovered cloud printers by name and then updates the ListModel.\n\n        :return: None\n        '
        items = self._discovered_cloud_printers_list[:]
        items.sort(key=lambda k: k['name'])
        self.setItems(items)