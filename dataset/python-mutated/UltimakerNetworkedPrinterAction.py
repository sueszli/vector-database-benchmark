from typing import Optional, cast
from PyQt6.QtCore import pyqtSlot, pyqtSignal, pyqtProperty, QObject
from UM import i18nCatalog
from cura.CuraApplication import CuraApplication
from cura.MachineAction import MachineAction
from .UM3OutputDevicePlugin import UM3OutputDevicePlugin
from .Network.LocalClusterOutputDevice import LocalClusterOutputDevice
I18N_CATALOG = i18nCatalog('cura')

class UltimakerNetworkedPrinterAction(MachineAction):
    """Machine action that allows to connect the active machine to a networked devices.

    TODO: in the future this should be part of the new discovery workflow baked into Cura.
    """
    discoveredDevicesChanged = pyqtSignal()

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        super().__init__('DiscoverUM3Action', I18N_CATALOG.i18nc('@action', 'Connect via Network'))
        self._qml_url = 'resources/qml/DiscoverUM3Action.qml'
        self._network_plugin = None

    def needsUserInteraction(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Override the default value.'
        return False

    @pyqtSlot(name='startDiscovery')
    def startDiscovery(self) -> None:
        if False:
            return 10
        'Start listening to network discovery events via the plugin.'
        self._networkPlugin.discoveredDevicesChanged.connect(self._onDeviceDiscoveryChanged)
        self.discoveredDevicesChanged.emit()

    @pyqtSlot(name='reset')
    def reset(self) -> None:
        if False:
            return 10
        'Reset the discovered devices.'
        self.discoveredDevicesChanged.emit()

    @pyqtSlot(name='restartDiscovery')
    def restartDiscovery(self) -> None:
        if False:
            print('Hello World!')
        'Reset the discovered devices.'
        self._networkPlugin.startDiscovery()
        self.discoveredDevicesChanged.emit()

    @pyqtSlot(str, str, name='removeManualDevice')
    def removeManualDevice(self, key: str, address: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Remove a manually added device.'
        self._networkPlugin.removeManualDevice(key, address)

    @pyqtSlot(str, str, name='setManualDevice')
    def setManualDevice(self, key: str, address: str) -> None:
        if False:
            print('Hello World!')
        'Add a new manual device. Can replace an existing one by key.'
        if key != '':
            self._networkPlugin.removeManualDevice(key)
        if address != '':
            self._networkPlugin.addManualDevice(address)

    @pyqtProperty('QVariantList', notify=discoveredDevicesChanged)
    def foundDevices(self):
        if False:
            print('Hello World!')
        'Get the devices discovered in the local network sorted by name.'
        discovered_devices = list(self._networkPlugin.getDiscoveredDevices().values())
        discovered_devices.sort(key=lambda d: d.name)
        return discovered_devices

    @pyqtSlot(QObject, name='associateActiveMachineWithPrinterDevice')
    def associateActiveMachineWithPrinterDevice(self, device: LocalClusterOutputDevice) -> None:
        if False:
            i = 10
            return i + 15
        'Connect a device selected in the list with the active machine.'
        self._networkPlugin.associateActiveMachineWithPrinterDevice(device)

    def _onDeviceDiscoveryChanged(self) -> None:
        if False:
            i = 10
            return i + 15
        'Callback for when the list of discovered devices in the plugin was changed.'
        self.discoveredDevicesChanged.emit()

    @property
    def _networkPlugin(self) -> UM3OutputDevicePlugin:
        if False:
            while True:
                i = 10
        'Get the network manager from the plugin.'
        if not self._network_plugin:
            output_device_manager = CuraApplication.getInstance().getOutputDeviceManager()
            network_plugin = output_device_manager.getOutputDevicePlugin('UM3NetworkPrinting')
            self._network_plugin = cast(UM3OutputDevicePlugin, network_plugin)
        return self._network_plugin